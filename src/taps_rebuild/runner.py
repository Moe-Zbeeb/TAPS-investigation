import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import stat
import sys
import time
import traceback


MODES = ("speculative", "target_only")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_json(handle, value):
    handle.seek(0, os.SEEK_END)
    handle.write(canonical(value) + b"\n")
    handle.flush()
    os.fsync(handle.fileno())


def read_json(path):
    return json.loads(Path(path).read_bytes(), parse_constant=reject_nonfinite)


def reject_nonfinite(value):
    raise ValueError("Nonfinite JSON value: " + value)


def require_gpu_slurm():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Inference workers require a Slurm GPU allocation")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 1 or not devices[0].strip():
        raise RuntimeError("Expose exactly one allocated GPU to each inference worker")


def checked_relative(root, value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Authored source paths must be relative to the project")
    result = root / path
    if not result.resolve().is_relative_to(root):
        raise ValueError("Authored source path escapes the project")
    return result


def verify_provenance(root, config_path, manifest_path, method):
    manifest = read_json(manifest_path)
    if digest(config_path) != manifest["config_sha256"]:
        raise ValueError("Configuration hash differs from the frozen manifest")
    config = manifest["effective_config"]
    if method not in config["method_order"] or method not in config["methods"]:
        raise ValueError("The worker method is outside the effective configuration")
    if manifest["scope"] not in {"smoke", "reference"} or config["scope"] != manifest["scope"]:
        raise ValueError("The manifest and effective configuration scopes differ")
    hashes = manifest["authored_source_hashes"]
    authored_root = Path(manifest.get("authored_source_root", root)).resolve()
    if not hashes or any(name not in hashes for name in ("src/taps_rebuild/runner.py", "src/taps_rebuild/backends/official.py")):
        raise ValueError("The manifest must pin the runner and official backend")
    for name, expected in hashes.items():
        path = checked_relative(authored_root, name)
        if Path(name).parts[0] != "src" or digest(path) != expected:
            raise ValueError("Authored source hash differs: " + name)
    assets_path = Path(manifest.get("asset_receipt_path", root / "outputs/preparation/assets.json"))
    sources_path = Path(manifest.get("source_receipt_path", root / "outputs/preparation/sources.json"))
    if digest(assets_path) != manifest["asset_receipt_sha256"] or digest(sources_path) != manifest["source_receipt_sha256"]:
        raise ValueError("Preparation receipt hash differs from the manifest")
    assets, sources = read_json(assets_path), read_json(sources_path)
    for name, receipt in (("assets", assets), ("sources", sources)):
        if receipt.get("status") != "complete" or Path(receipt["root"]).resolve() != root:
            raise ValueError("A complete preparation receipt for this project is required: " + name)
        registry_key = "asset_registry_path" if name == "assets" else "source_registry_path"
        registry_path = manifest.get(registry_key, receipt["registry"])
        if digest(registry_path) != receipt["registry_sha256"] or digest(authored_root / f"src/taps_rebuild/{name}.py") != receipt["source_sha256"]:
            raise ValueError("Preparation registry or builder changed: " + name)
    selected = config["methods"][method]
    source = sources["sources"][selected["source"]]
    directory = Path(source["path"])
    for name, record in source["tracked_files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid tracked source path")
        path = directory / relative
        metadata = path.lstat()
        if record["kind"] == "symlink":
            if not stat.S_ISLNK(metadata.st_mode):
                raise ValueError("Official source file type changed")
            contents = os.fsencode(os.readlink(path))
            actual, size = hashlib.sha256(contents).hexdigest(), len(contents)
        else:
            if record["kind"] != "regular_file" or not stat.S_ISREG(metadata.st_mode):
                raise ValueError("Official source file type changed")
            actual, size = digest(path), metadata.st_size
        if actual != record["sha256"] or size != record["size_bytes"]:
            raise ValueError("Official source content changed: " + name)
    if identity(source["tracked_files"]) != source["source_identity_sha256"]:
        raise ValueError("Official source inventory identity differs")
    prompt = source["system_prompt"]
    if hashlib.sha256(prompt["text"].encode("utf-8")).hexdigest() != prompt["sha256"]:
        raise ValueError("System prompt hash differs")
    models = {}
    for role in ("target", "drafter"):
        model = assets["models"][selected[role]]
        if model["status"] != "verified":
            raise ValueError("A verified model preparation record is required")
        path = Path(model["path"])
        for name, metadata in model["files"].items():
            file = path / name
            if not file.is_file() or file.stat().st_size != metadata["size_bytes"]:
                raise ValueError("Prepared model file is missing or changed in size: " + name)
        if digest(path / "config.json") != model["files"]["config.json"]["sha256"]:
            raise ValueError("Prepared model configuration changed")
        models[role] = model
    frozen = {
        "run_id": manifest["run_id"],
        "scope": manifest["scope"],
        "scientific_identity_sha256": manifest["scientific_identity_sha256"],
        "config_sha256": manifest["config_sha256"],
        "effective_config_sha256": identity(config),
        "manifest_sha256": digest(manifest_path),
        "asset_receipt_sha256": manifest["asset_receipt_sha256"],
        "source_receipt_sha256": manifest["source_receipt_sha256"],
        "authored_source_hashes_sha256": identity(hashes),
        "code_commit": manifest.get("code_commit"),
        "source_identity_sha256": source["source_identity_sha256"],
        "source_revision": source["revision"],
        "target_model_id": selected["target"],
        "draft_model_id": selected["drafter"],
        "target_model_identity_sha256": models["target"]["model_identity_sha256"],
        "draft_model_identity_sha256": models["drafter"]["model_identity_sha256"],
    }
    return manifest, config, sources, source, models, frozen


def seed_for(base_seed, benchmark, question_id, turn_id, repeat):
    return int.from_bytes(hashlib.sha256(canonical([base_seed, benchmark, question_id, turn_id, repeat])).digest()[:4], "big")


def seed_runtime(seed, torch, numpy):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def case_fields(method, benchmark, temperature, question_id, turn_id, repeat, mode, base_seed):
    fields = {
        "method": method, "benchmark": benchmark, "temperature": float(temperature),
        "question_id": question_id, "turn_id": turn_id, "repeat": repeat,
        "generation_mode": mode, "seed": seed_for(base_seed, benchmark, question_id, turn_id, repeat),
    }
    fields["case_id"] = identity(fields)
    return fields


def selected_questions(config, sources, method, shard_index, shard_count):
    selected = {}
    for benchmark in config["methods"][method]["benchmarks"]:
        record = sources["benchmarks"][benchmark]
        path = Path(record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError("Reference benchmark hash differs: " + benchmark)
        rows = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
        if len(rows) != record["rows"] or len({canonical(row["question_id"]) for row in rows}) != len(rows):
            raise ValueError("Reference benchmark row count or question identities differ")
        if any(len(row["turns"]) != record["turns_per_row"] for row in rows):
            raise ValueError("Reference benchmark turn count differs")
        limit = config.get("question_limit", len(rows))
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("question_limit must be a positive integer")
        selected[benchmark] = [row for index, row in enumerate(rows[:limit]) if index % shard_count == shard_index]
    if not any(selected.values()):
        raise ValueError("This shard has no selected questions")
    return selected


def planned_cases(config, method, questions):
    for name, minimum in (("timing_repeats", 1), ("warmup_generations_per_mode", 0), ("max_new_tokens", 1), ("max_length", 1), ("total_token", 2), ("top_k", 1)):
        if type(config[name]) is not int or config[name] < minimum:
            raise ValueError("Invalid effective configuration value: " + name)
    if config.get("batch_size", 1) != 1 or config.get("dtype", "float16") != "float16":
        raise ValueError("Official reproduction requires FP16 and batch size one")
    if not config["temperatures"] or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in config["temperatures"]):
        raise ValueError("Effective temperatures must be finite and nonnegative")
    cases = {}
    for benchmark, rows in questions.items():
        for row in rows:
            for temperature in config["temperatures"]:
                for repeat in range(config["timing_repeats"]):
                    for turn_id in range(len(row["turns"])):
                        for mode in MODES:
                            fields = case_fields(method, benchmark, temperature, row["question_id"], turn_id, repeat, mode, config["base_seed"])
                            if fields["case_id"] in cases:
                                raise ValueError("The effective configuration generates duplicate case IDs")
                            cases[fields["case_id"]] = fields
    return cases


def validate_completed(row, frozen, planned, shard_index, shard_count):
    if row.get("status") != "completed" or row.get("warmup") is not False:
        raise ValueError("Resume requires completed measured records")
    if any(row.get(key) != value for key, value in frozen.items()):
        raise ValueError("Resume record identity differs from the frozen run")
    fields = planned.get(row.get("case_id"))
    if fields is None or any(row.get(key) != value for key, value in fields.items()):
        raise ValueError("Resume record is outside the planned cases")
    if row.get("shard_index") != shard_index or row.get("shard_count") != shard_count:
        raise ValueError("Resume record belongs to another shard topology")
    if "input_sha256" not in row or "input_token_ids" not in row or identity(row["input_token_ids"]) != row["input_sha256"]:
        raise ValueError("Resume record input hash is missing or inconsistent")
    if not isinstance(row.get("output_text"), str):
        raise ValueError("Resume record is missing decoded assistant history")
    output, comparison, cycles = row["output_token_ids"], row["comparison_token_ids"], row["cycles"]
    if not isinstance(output, list) or not output or any(type(token) is not int or token < 0 for token in output):
        raise ValueError("Completed output token IDs are invalid")
    if comparison != output[:len(comparison)] or len(comparison) > len(output):
        raise ValueError("Comparison tokens are not a raw-output prefix")
    if row["new_token_count"] != len(output) or row["cycle_count"] != len(cycles) or not cycles:
        raise ValueError("Completed token or cycle counts differ")
    for cycle in cycles:
        for key, minimum in (("accepted_draft_tokens", 0), ("emitted_tokens", 1), ("verifier_nodes", 1), ("proposed_nodes", 0)):
            if type(cycle[key]) is not int or cycle[key] < minimum:
                raise ValueError("Completed cycle contains an invalid count")
        for key in ("drafting_ms", "verification_ms"):
            if type(cycle[key]) not in (int, float) or not math.isfinite(cycle[key]) or cycle[key] < 0:
                raise ValueError("Completed cycle contains an invalid timing")
    if sum(cycle["emitted_tokens"] for cycle in cycles) != len(output):
        raise ValueError("Completed cycle emissions disagree with output tokens")
    if row["generation_mode"] == "speculative" and any(cycle["accepted_draft_tokens"] + 1 != cycle["emitted_tokens"] for cycle in cycles):
        raise ValueError("Completed speculative acceptance accounting differs")
    if not isinstance(row["wall_ms"], (int, float)) or not math.isfinite(row["wall_ms"]) or row["wall_ms"] <= 0:
        raise ValueError("Completed generation wall time is invalid")


def resume_rows(handle, path, frozen, planned, shard_index, shard_count):
    handle.seek(0)
    contents = handle.read()
    boundary = contents.rfind(b"\n") + 1
    rows = {}
    for line in contents[:boundary].split(b"\n")[:-1]:
        if not line.strip():
            raise ValueError("Blank committed raw record")
        row = json.loads(line, parse_constant=reject_nonfinite)
        validate_completed(row, frozen, planned, shard_index, shard_count)
        if row["case_id"] in rows:
            raise ValueError("Duplicate committed case ID")
        rows[row["case_id"]] = row
    recovered = None
    tail = contents[boundary:]
    if tail:
        stamp = str(time.time_ns())
        recovery = path.with_name(path.name + ".recovery-" + stamp + ".bin")
        with recovery.open("xb") as archived:
            archived.write(tail)
            archived.flush()
            os.fsync(archived.fileno())
        handle.truncate(boundary)
        handle.flush()
        os.fsync(handle.fileno())
        recovered = {"path": str(recovery), "sha256": hashlib.sha256(tail).hexdigest(), "bytes": len(tail)}
    return rows, recovered


def encode(tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer([text], add_special_tokens=False, return_tensors="pt")["input_ids"]


def stop_ids(tokenizer):
    eos = tokenizer.eos_token_id
    values = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    values.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    return {value for value in values if isinstance(value, int) and value >= 0}


def decode_completion(tokenizer, tokens, cap):
    stops = stop_ids(tokenizer)
    first = next((index for index, token in enumerate(tokens) if token in stops), None)
    comparison = tokens[:min(cap, first + 1 if first is not None else len(tokens))]
    text = tokenizer.decode(tokens[:first] if first is not None else tokens, spaces_between_special_tokens=False)
    for value in tokenizer.special_tokens_map.values():
        for token in value if isinstance(value, list) else [value]:
            text = text.replace(token, "")
    return text.strip(), comparison


def context_record(backend, config, input_length):
    capacity = backend.model.base_model.config.max_position_embeddings if backend.method == "hass" else config["max_length"]
    nodes = backend.model.ea_layer.total_tokens + 1
    effective = config["max_length"] - nodes + 1 - 10
    remaining = max(0, effective - input_length)
    comparison_cap = min(config["max_new_tokens"], remaining)
    return {
        "prompt_tokens": input_length, "physical_capacity": capacity,
        "verifier_node_budget": nodes, "effective_context_limit": effective,
        "fits_physical_tree": input_length + nodes <= capacity,
        "prefix_within_effective_context": input_length < effective,
        "requested_generation_within_effective_context": input_length + config["max_new_tokens"] <= effective,
        "comparison_cap": comparison_cap,
        "limiting_budget": "effective_context" if remaining < config["max_new_tokens"] else "max_new_tokens" if remaining > config["max_new_tokens"] else "both",
        "automatic_truncation": False,
    }


def check_context(record):
    if not record["fits_physical_tree"]:
        raise ValueError("Encoded prompt plus tree exceeds physical cache capacity; no truncation was applied")
    if not record["prefix_within_effective_context"]:
        raise ValueError("Encoded prompt exceeds the declared effective context; investigate the published context policy before proceeding")


def run_worker(root, config_path, run_dir, manifest_path, method, shard_index, shard_count,
               smoke=False, backend_factory=None, torch_module=None, numpy_module=None):
    require_gpu_slurm()
    root, config_path, run_dir, manifest_path = map(lambda path: Path(path).resolve(), (root, config_path, run_dir, manifest_path))
    if not run_dir.is_relative_to(root / "outputs"):
        raise ValueError("Worker run directories must remain under project outputs/")
    if method not in {"eagle3", "eagle2", "hass"} or shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid worker method or shard topology")
    raw_path = run_dir / "raw" / f"{method}-shard{shard_index}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = run_dir / "workers" / f"{method}-{shard_index}.json"
    error_path = run_dir / "errors" / f"{method}-{shard_index}.jsonl"
    descriptor = os.open(raw_path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "r+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another worker holds this raw-log writer lock") from None
        backend = None
        active_case = None
        receipt = {"status": "running", "method": method, "shard_index": shard_index, "shard_count": shard_count, "slurm_job_id": os.environ["SLURM_JOB_ID"], "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "started_at_utc": datetime.now(timezone.utc).isoformat(), "raw_path": str(raw_path)}
        try:
            manifest, config, sources, source, models, frozen = verify_provenance(root, config_path, manifest_path, method)
            if smoke and frozen["scope"] != "smoke":
                raise ValueError("--smoke requires a smoke-scoped effective manifest")
            questions = selected_questions(config, sources, method, shard_index, shard_count)
            planned = planned_cases(config, method, questions)
            rows, recovered = resume_rows(handle, raw_path, frozen, planned, shard_index, shard_count)
            receipt.update(frozen, resumed_cases=len(rows), planned_cases=len(planned), recovered_tail=recovered)
            atomic_json(receipt_path, receipt)
            torch = torch_module or importlib.import_module("torch")
            numpy = numpy_module or importlib.import_module("numpy")
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("Worker requires exactly one visible CUDA device")
            if hasattr(torch.cuda, "get_device_properties"):
                properties = torch.cuda.get_device_properties(0)
                receipt["gpu"] = {"logical_device": 0, "name": properties.name, "total_memory_bytes": properties.total_memory}
            if backend_factory is None:
                from .backends.official import OfficialBackend
                backend_factory = OfficialBackend
            selected = config["methods"][method]
            backend = backend_factory(method, source["path"], models["target"]["path"], models["drafter"]["path"], device="cuda:0", total_token=config["total_token"], depth=selected["depth"], top_k=config["top_k"])
            tokenizer = backend.tokenizer
            system = {"role": "system", "content": source["system_prompt"]["text"]}
            audit = []
            first_input = None
            first_benchmark = None
            first_question = None
            for benchmark, selected_rows in questions.items():
                for question in selected_rows:
                    encoded = encode(tokenizer, [system, {"role": "user", "content": question["turns"][0]}])
                    context = context_record(backend, config, encoded.shape[1])
                    audit.append({"benchmark": benchmark, "question_id": question["question_id"], **context})
                    if first_input is None:
                        first_input, first_benchmark, first_question = encoded, benchmark, question
            atomic_json(receipt_path.with_name(receipt_path.stem + "-context-audit.json"), {"scope": frozen["scope"], "initial_turns": audit, "subsequent_turns_checked_before_generation": True})
            for context in audit:
                check_context(context)
            warmups = 0
            if len(rows) < len(planned):
                for repeat in range(config["warmup_generations_per_mode"]):
                    for mode in MODES:
                        seed = seed_for(config["base_seed"], first_benchmark, first_question["question_id"], 0, -repeat - 1)
                        seed_runtime(seed, torch, numpy)
                        backend.generate(first_input.clone(), temperature=config["temperatures"][0], max_new_tokens=config["max_new_tokens"], max_length=config["max_length"], generation_mode=mode)
                        warmups += 1
            for benchmark, selected_rows in questions.items():
                for temperature in config["temperatures"]:
                    for repeat in range(config["timing_repeats"]):
                        for question in selected_rows:
                            messages = [dict(system)]
                            for turn_id, user_text in enumerate(question["turns"]):
                                messages.append({"role": "user", "content": user_text})
                                encoded = encode(tokenizer, messages)
                                input_tokens = encoded[0].detach().cpu().tolist()
                                input_sha = identity(input_tokens)
                                context = context_record(backend, config, len(input_tokens))
                                check_context(context)
                                parity = int(identity([benchmark, temperature, question["question_id"], turn_id])[:8], 16)
                                modes = MODES if (parity + repeat) % 2 == 0 else tuple(reversed(MODES))
                                paired = {}
                                for mode in modes:
                                    fields = case_fields(method, benchmark, temperature, question["question_id"], turn_id, repeat, mode, config["base_seed"])
                                    active_case = {**fields, "input_sha256": input_sha}
                                    if fields["case_id"] in rows:
                                        row = rows[fields["case_id"]]
                                        if row["input_sha256"] != input_sha or row["input_token_ids"] != input_tokens:
                                            raise ValueError("Resumed case input prefix differs from reconstructed speculative history")
                                        text, comparison = decode_completion(tokenizer, row["output_token_ids"], context["comparison_cap"])
                                        if row["output_text"] != text or row["comparison_token_ids"] != comparison:
                                            raise ValueError("Resumed decoded history or comparison prefix differs from raw output tokens")
                                        if row.get("comparison_cap") != context["comparison_cap"] or row.get("limiting_budget") != context["limiting_budget"]:
                                            raise ValueError("Resumed comparison budget differs from the shared input context")
                                        if row.get("benchmark_sha256") != sources["benchmarks"][benchmark]["sha256"] or row.get("system_prompt_sha256") != source["system_prompt"]["sha256"]:
                                            raise ValueError("Resumed benchmark or system prompt identity differs")
                                    else:
                                        seed_runtime(fields["seed"], torch, numpy)
                                        result = backend.generate(encoded.clone(), temperature=temperature, max_new_tokens=config["max_new_tokens"], max_length=config["max_length"], generation_mode=mode)
                                        output = result["output_ids"][0, len(input_tokens):].detach().cpu().tolist()
                                        text, comparison = decode_completion(tokenizer, output, context["comparison_cap"])
                                        metadata = {key: value for key, value in result.items() if key not in {"output_ids", "cycles"}}
                                        metadata["verification_includes_posterior"] = mode == "speculative"
                                        row = {
                                            **frozen, **fields,
                                            "status": "completed", "warmup": False,
                                            "shard_index": shard_index, "shard_count": shard_count,
                                            "slurm_job_id": os.environ["SLURM_JOB_ID"], "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                                            "input_sha256": input_sha, "input_token_ids": input_tokens,
                                            "benchmark_sha256": sources["benchmarks"][benchmark]["sha256"],
                                            "system_prompt_sha256": source["system_prompt"]["sha256"],
                                            "cycles": result["cycles"], "new_token_count": result["new_token_count"],
                                            "cycle_count": result["truecycle_count"], "wall_ms": result["wall_total_ms"],
                                            "output_token_ids": output, "comparison_token_ids": comparison,
                                            "comparison_cap": context["comparison_cap"], "limiting_budget": context["limiting_budget"],
                                            "output_text": text, "raw_reported_idx": result["official_last_iteration"],
                                            "backend_metadata": metadata, "context_guard": context,
                                            "unused_final_drafting_ms": result.get("unused_final_drafting_ms", 0.0),
                                            "timing_scope": config["timing_scope"],
                                            "cpu_tokenization_detokenization_and_logging_excluded_from_wall_ms": True,
                                            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                        }
                                        validate_completed(row, frozen, planned, shard_index, shard_count)
                                        append_json(handle, row)
                                        rows[fields["case_id"]] = row
                                    paired[mode] = row
                                messages.append({"role": "assistant", "content": paired["speculative"]["output_text"]})
            if set(rows) != set(planned):
                raise RuntimeError("Worker finished without every planned completed case")
            if verify_provenance(root, config_path, manifest_path, method)[-1] != frozen:
                raise ValueError("Run identities changed during inference")
            backend.close()
            backend = None
            receipt.update(status="complete", completed_cases=len(rows), new_cases=len(rows) - receipt["resumed_cases"], warmup_generations=warmups, completed_at_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(receipt_path, receipt)
            return receipt
        except BaseException as error:
            failure = {"status": "failed", "case": active_case, "method": method, "shard_index": shard_index, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "failed_at_utc": datetime.now(timezone.utc).isoformat()}
            error_path.parent.mkdir(parents=True, exist_ok=True)
            with error_path.open("a+b") as errors:
                append_json(errors, failure)
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass
            receipt.update(status="failed", error_type=type(error).__name__, error=str(error), failed_at_utc=failure["failed_at_utc"])
            atomic_json(receipt_path, receipt)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", choices=("eagle3", "eagle2", "hass"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    receipt = run_worker(args.root, args.config, args.run_dir, args.manifest, args.method, args.shard_index, args.shard_count, args.smoke)
    print(json.dumps({"status": receipt["status"], "completed_cases": receipt["completed_cases"], "raw_path": receipt["raw_path"]}), flush=True)


if __name__ == "__main__":
    main()
