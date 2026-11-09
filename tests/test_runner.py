import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taps_rebuild import runner


class Tensor:
    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self.shape = (len(data), len(data[0])) if data and isinstance(data[0], list) else (len(data),)
        self.ndim = len(self.shape)

    def clone(self):
        return Tensor(self.data)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.data)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return Tensor(self.data[index[0]][index[1]])
        return Tensor(self.data[index])


class Tokenizer:
    eos_token_id = 2
    special_tokens_map = {"eos_token": "<eos>", "additional_special_tokens": ["<eot>"]}

    def convert_tokens_to_ids(self, text):
        return 3

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if tokenize or not add_generation_prompt:
            raise AssertionError("Official template contract changed")
        return json.dumps(messages, sort_keys=True)

    def __call__(self, texts, add_special_tokens, return_tensors):
        if add_special_tokens or return_tensors != "pt":
            raise AssertionError("Official tokenization contract changed")
        return {"input_ids": Tensor([[ord(character) + 1000 for character in texts[0]]])}

    def decode(self, tokens, spaces_between_special_tokens):
        if spaces_between_special_tokens:
            raise AssertionError("Official decoding contract changed")
        return "".join({101: " spec answer<eot> ", 102: " target answer ", 909: "tail", 2: "<eos>", 3: "<eot>"}.get(token, "?") for token in tokens)


class Runtime:
    def __init__(self):
        self.seed = None
        self.numpy_seed = None
        self.cuda_seed = None
        self.torch = types.SimpleNamespace(
            manual_seed=self.manual_seed,
            cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, manual_seed_all=self.manual_seed_all),
        )
        self.numpy = types.SimpleNamespace(random=types.SimpleNamespace(seed=self.seed_numpy))

    def manual_seed(self, seed):
        self.seed = seed

    def seed_numpy(self, seed):
        self.numpy_seed = seed

    def manual_seed_all(self, seed):
        self.cuda_seed = seed


class Backend:
    def __init__(self, owner, method, source_dir, target_path, draft_path, **kwargs):
        self.owner = owner
        self.method = method
        self.tokenizer = Tokenizer()
        self.model = types.SimpleNamespace(
            base_model=types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=8192)),
            ea_layer=types.SimpleNamespace(total_tokens=kwargs["total_token"] - 1),
        )
        self.closed = False
        owner.backends.append(self)

    def generate(self, input_ids, *, temperature, max_new_tokens, max_length, generation_mode):
        mode = generation_mode
        call = {"mode": mode, "input": input_ids.tolist()[0], "seed": self.owner.runtime.seed, "numpy_seed": self.owner.runtime.numpy_seed, "cuda_seed": self.owner.runtime.cuda_seed}
        self.owner.calls.append(call)
        if self.owner.fail_mode == mode:
            raise RuntimeError("injected inference failure")
        output = [101, 2, 909] if mode == "speculative" else [102, 2]
        cycles = [{"accepted_draft_tokens": 2, "emitted_tokens": 3, "verifier_nodes": 60, "proposed_nodes": 59, "observed_depth": 8, "drafting_ms": 0.1, "verification_ms": 0.2}] if mode == "speculative" else [{"accepted_draft_tokens": 0, "emitted_tokens": 1, "verifier_nodes": 1, "proposed_nodes": 0, "observed_depth": 0, "drafting_ms": 0.0, "verification_ms": 0.2} for token in output]
        return {
            "output_ids": Tensor([input_ids.tolist()[0] + output]),
            "new_token_count": len(output), "truecycle_count": len(cycles),
            "cycles": cycles, "wall_total_ms": 1.0,
            "official_last_iteration": len(cycles) - 1,
            "unused_drafter_resident": mode == "target_only",
            "unused_final_drafting_ms": 0.1 if mode == "speculative" else 0.0,
        }

    def close(self):
        self.closed = True


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.saved_random_state = random.getstate()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "project"
        self.run_dir = self.root / "outputs/run"
        self.config_path = self.root / "configs/reproduction.json"
        self.manifest_path = self.run_dir / "manifest.json"
        self.runtime = Runtime()
        self.calls = []
        self.backends = []
        self.fail_mode = None
        self.environment = patch.dict(os.environ, {"SLURM_JOB_ID": "unit-job", "CUDA_VISIBLE_DEVICES": "7"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(random.setstate, self.saved_random_state)
        self.config = {
            "scope": "smoke", "method_order": ["eagle3"],
            "methods": {"eagle3": {"source": "EAGLE", "target": "target", "drafter": "draft", "depth": 7, "benchmarks": ["mt_bench"]}},
            "temperatures": [0], "timing_repeats": 1,
            "warmup_generations_per_mode": 0, "max_new_tokens": 3,
            "max_length": 2048, "total_token": 60, "top_k": 10,
            "base_seed": 20261109, "timing_scope": "generation only",
        }
        self.authored = {}
        for name in ["runner.py", "backends/official.py", "assets.py", "sources.py"]:
            path = self.root / "src/taps_rebuild" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("frozen " + name)
            self.authored[str(path.relative_to(self.root))] = runner.digest(path)
        for name in ["assets", "sources"]:
            path = self.root / f"configs/{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        source_path = self.root / "repos/EAGLE/eagle/model/ea_model.py"
        source_path.parent.mkdir(parents=True)
        source_path.write_text("official source")
        tracked = {"eagle/model/ea_model.py": {"kind": "regular_file", "size_bytes": source_path.stat().st_size, "sha256": runner.digest(source_path)}}
        self.questions = [{"question_id": 7, "turns": ["first question", "second turn"]}, {"question_id": 8, "turns": ["another question", "another second turn"]}]
        benchmark_path = self.root / "data/reference/mt_bench.jsonl"
        benchmark_path.parent.mkdir(parents=True)
        benchmark_path.write_bytes(b"".join(runner.canonical(question) + b"\n" for question in self.questions))
        models = {}
        for name in ["target", "draft"]:
            path = self.root / "models" / name
            path.mkdir(parents=True)
            (path / "config.json").write_text("{}")
            models[name] = {"status": "verified", "path": str(path), "model_identity_sha256": runner.identity(name), "files": {"config.json": {"size_bytes": 2, "sha256": runner.digest(path / "config.json")}}}
        assets = {"status": "complete", "root": str(self.root), "registry": str(self.root / "configs/assets.json"), "registry_sha256": runner.digest(self.root / "configs/assets.json"), "source_sha256": self.authored["src/taps_rebuild/assets.py"], "models": models}
        prompt = "official system"
        sources = {"status": "complete", "root": str(self.root), "registry": str(self.root / "configs/sources.json"), "registry_sha256": runner.digest(self.root / "configs/sources.json"), "source_sha256": self.authored["src/taps_rebuild/sources.py"], "sources": {"EAGLE": {"path": str(source_path.parents[2]), "revision": "a" * 40, "tracked_files": tracked, "source_identity_sha256": runner.identity(tracked), "system_prompt": {"text": prompt, "sha256": hashlib.sha256(prompt.encode()).hexdigest()}}}, "benchmarks": {"mt_bench": {"path": str(benchmark_path), "sha256": runner.digest(benchmark_path), "rows": 2, "turns_per_row": 2}}}
        runner.atomic_json(self.root / "outputs/preparation/assets.json", assets)
        runner.atomic_json(self.root / "outputs/preparation/sources.json", sources)
        self.freeze()

    def freeze(self):
        runner.atomic_json(self.config_path, self.config)
        self.manifest = {
            "config_sha256": runner.digest(self.config_path),
            "scientific_identity_sha256": runner.identity("science"), "run_id": "run",
            "scope": self.config["scope"], "effective_config": self.config,
            "asset_receipt_sha256": runner.digest(self.root / "outputs/preparation/assets.json"),
            "source_receipt_sha256": runner.digest(self.root / "outputs/preparation/sources.json"),
            "authored_source_hashes": self.authored,
        }
        runner.atomic_json(self.manifest_path, self.manifest)

    def run_worker(self, shard_index=0, shard_count=1):
        return runner.run_worker(self.root, self.config_path, self.run_dir, self.manifest_path, "eagle3", shard_index, shard_count, smoke=True, backend_factory=lambda *args, **kwargs: Backend(self, *args, **kwargs), torch_module=self.runtime.torch, numpy_module=self.runtime.numpy)

    def rows(self, shard_index=0):
        path = self.run_dir / "raw" / f"eagle3-shard{shard_index}.jsonl"
        return [json.loads(line) for line in path.read_text().split("\n") if line]

    def rewrite_rows(self, rows):
        path = self.run_dir / "raw/eagle3-shard0.jsonl"
        path.write_bytes(b"".join(runner.canonical(row) + b"\n" for row in rows))

    def test_pairs_share_prefix_seed_and_speculative_assistant_history(self):
        receipt = self.run_worker()
        rows = self.rows()
        self.assertEqual(len(rows), 8)
        self.assertEqual(receipt["completed_cases"], 8)
        self.assertTrue(self.backends[0].closed)
        groups = {}
        for row in rows:
            key = row["question_id"], row["turn_id"]
            groups.setdefault(key, []).append(row)
        for key, pair in groups.items():
            self.assertEqual({row["generation_mode"] for row in pair}, set(runner.MODES))
            self.assertEqual(pair[0]["input_sha256"], pair[1]["input_sha256"])
            self.assertEqual(pair[0]["seed"], pair[1]["seed"])
            messages = json.loads("".join(chr(token - 1000) for token in pair[0]["input_token_ids"]))
            self.assertEqual(messages[0], {"role": "system", "content": "official system"})
            if key[1] == 1:
                self.assertEqual(messages[2], {"role": "assistant", "content": "spec answer"})
        for call in self.calls:
            self.assertEqual(call["seed"], call["numpy_seed"])
            self.assertEqual(call["seed"], call["cuda_seed"])

    def test_inclusive_eos_comparison_and_untrimmed_raw_output(self):
        self.run_worker()
        row = next(row for row in self.rows() if row["generation_mode"] == "speculative")
        self.assertEqual(row["output_token_ids"], [101, 2, 909])
        self.assertEqual(row["comparison_token_ids"], [101, 2])
        self.assertEqual(row["output_text"], "spec answer")
        self.assertEqual(row["new_token_count"], 3)
        self.assertEqual(row["cycle_count"], 1)
        self.assertEqual(row["raw_reported_idx"], 0)
        self.assertEqual(row["unused_final_drafting_ms"], 0.1)
        text, comparison = runner.decode_completion(Tokenizer(), [101, 909, 3, 909], 2)
        self.assertEqual(comparison, [101, 909])
        self.assertEqual(text, "spec answer tail")

    def test_resume_skips_saved_modes_and_uses_saved_speculative_history(self):
        self.run_worker()
        saved = self.rows()
        removed = next(row for row in saved if row["question_id"] == 7 and row["turn_id"] == 1 and row["generation_mode"] == "target_only")
        self.rewrite_rows([row for row in saved if row["case_id"] != removed["case_id"]])
        calls = len(self.calls)
        receipt = self.run_worker()
        self.assertEqual(len(self.calls) - calls, 1)
        self.assertEqual(self.calls[-1]["mode"], "target_only")
        self.assertEqual(receipt["new_cases"], 1)
        self.assertEqual(len({row["case_id"] for row in self.rows()}), 8)
        messages = json.loads("".join(chr(token - 1000) for token in self.calls[-1]["input"]))
        self.assertEqual(messages[2]["content"], "spec answer")
        calls = len(self.calls)
        self.assertEqual(self.run_worker()["new_cases"], 0)
        self.assertEqual(len(self.calls), calls)

    def test_unterminated_tail_is_archived_exactly_before_resume(self):
        self.run_worker()
        path = self.run_dir / "raw/eagle3-shard0.jsonl"
        tail = b'{"interrupted":'
        with path.open("ab") as handle:
            handle.write(tail)
        receipt = self.run_worker()
        recovered = receipt["recovered_tail"]
        self.assertEqual(Path(recovered["path"]).read_bytes(), tail)
        self.assertEqual(recovered["sha256"], hashlib.sha256(tail).hexdigest())
        self.assertTrue(path.read_bytes().endswith(b"\n"))
        self.assertEqual(len(self.rows()), 8)

    def test_missing_or_changed_input_hash_fails_before_model_loading(self):
        self.run_worker()
        rows = self.rows()
        del rows[0]["input_sha256"]
        self.rewrite_rows(rows)
        count = len(self.backends)
        with self.assertRaisesRegex(ValueError, "input hash"):
            self.run_worker()
        self.assertEqual(len(self.backends), count)
        receipt = runner.read_json(self.run_dir / "workers/eagle3-0.json")
        self.assertEqual(receipt["status"], "failed")

    def test_manifest_hash_mismatch_fails_before_loading(self):
        self.config_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "Configuration hash"):
            self.run_worker()
        self.assertEqual(self.backends, [])

    def test_corrupt_saved_history_is_not_used_for_later_turns(self):
        self.run_worker()
        rows = self.rows()
        saved = next(row for row in rows if row["generation_mode"] == "speculative")
        saved["output_text"] = "corrupted assistant text"
        self.rewrite_rows(rows)
        calls = len(self.calls)
        with self.assertRaisesRegex(ValueError, "decoded history"):
            self.run_worker()
        self.assertEqual(len(self.calls), calls)

    def test_zero_repeats_cannot_create_a_false_complete_worker(self):
        self.config["timing_repeats"] = 0
        self.freeze()
        with self.assertRaisesRegex(ValueError, "timing_repeats"):
            self.run_worker()
        self.assertEqual(self.backends, [])

    def test_sharding_uses_original_question_ordinal(self):
        self.run_worker(1, 2)
        self.assertEqual({row["question_id"] for row in self.rows(1)}, {8})
        self.assertEqual(len(self.rows(1)), 4)

    def test_warmups_are_excluded_and_paired_order_alternates_by_repeat(self):
        self.config["warmup_generations_per_mode"] = 1
        self.config["timing_repeats"] = 2
        self.freeze()
        receipt = self.run_worker()
        self.assertEqual(receipt["warmup_generations"], 2)
        self.assertEqual(len(self.rows()), 16)
        self.assertTrue(all(row["warmup"] is False for row in self.rows()))
        ordered = {}
        for row in self.rows():
            if row["question_id"] == 7 and row["turn_id"] == 0:
                ordered.setdefault(row["repeat"], []).append(row["generation_mode"])
        self.assertEqual(ordered[0], list(reversed(ordered[1])))

    def test_failed_generation_has_separate_error_and_no_false_completed_case(self):
        self.fail_mode = "speculative"
        with self.assertRaisesRegex(RuntimeError, "injected inference failure"):
            self.run_worker()
        self.assertTrue(self.backends[0].closed)
        error_path = self.run_dir / "errors/eagle3-0.jsonl"
        failure = json.loads(error_path.read_bytes().split(b"\n")[0])
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["case"]["generation_mode"], "speculative")
        self.assertTrue(all(row["generation_mode"] != "speculative" for row in self.rows()))
        self.assertEqual(runner.read_json(self.run_dir / "workers/eagle3-0.json")["status"], "failed")

    def test_frozen_authored_and_receipt_paths_are_authoritative(self):
        frozen_root = self.run_dir / "inputs/source"
        shutil.copytree(self.root / "src", frozen_root / "src")
        self.manifest["authored_source_root"] = str(frozen_root)
        for key, name in [("asset_receipt_path", "assets"), ("source_receipt_path", "sources")]:
            target = self.run_dir / f"inputs/{name}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.root / f"outputs/preparation/{name}.json", target)
            self.manifest[key] = str(target)
            (self.root / f"outputs/preparation/{name}.json").write_text("invalid live receipt")
        (self.root / "src/taps_rebuild/runner.py").write_text("different live code")
        runner.atomic_json(self.manifest_path, self.manifest)
        self.assertEqual(self.run_worker()["status"], "complete")

    def test_duplicate_worker_cannot_take_the_raw_writer_lock(self):
        path = self.run_dir / "raw/eagle3-shard0.jsonl"
        path.parent.mkdir(parents=True)
        with path.open("w+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "writer lock"):
                self.run_worker()
        self.assertEqual(self.backends, [])
        self.assertFalse((self.run_dir / "workers/eagle3-0.json").exists())

    def test_frozen_registry_paths_preserve_unmodified_preparation_receipts(self):
        originals = {}
        for key, name in [("asset_registry_path", "assets"), ("source_registry_path", "sources")]:
            receipt_path = self.root / f"outputs/preparation/{name}.json"
            originals[name] = receipt_path.read_bytes()
            path = self.run_dir / f"inputs/registries/{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.root / f"configs/{name}.json", path)
            self.manifest[key] = str(path)
            (self.root / f"configs/{name}.json").write_text("changed live registry")
        runner.atomic_json(self.manifest_path, self.manifest)
        self.assertEqual(self.run_worker()["status"], "complete")
        for name, contents in originals.items():
            self.assertEqual((self.root / f"outputs/preparation/{name}.json").read_bytes(), contents)

    def test_context_limited_pairs_share_the_same_comparison_cap_on_resume(self):
        sources_path = self.root / "outputs/preparation/sources.json"
        sources = runner.read_json(sources_path)
        benchmark = sources["benchmarks"]["mt_bench"]
        question = {"question_id": 7, "turns": ["first question"]}
        Path(benchmark["path"]).write_bytes(runner.canonical(question) + b"\n")
        benchmark.update(rows=1, turns_per_row=1, sha256=runner.digest(benchmark["path"]))
        runner.atomic_json(sources_path, sources)
        prefix = runner.encode(Tokenizer(), [{"role": "system", "content": "official system"}, {"role": "user", "content": "first question"}])
        self.config["max_length"] = prefix.shape[1] + self.config["total_token"] - 1 + 10 + 1
        self.freeze()
        self.run_worker()
        rows = self.rows()
        self.assertEqual({row["comparison_cap"] for row in rows}, {1})
        self.assertEqual({row["limiting_budget"] for row in rows}, {"effective_context"})
        self.assertEqual([len(row["comparison_token_ids"]) for row in rows], [1, 1])
        self.assertEqual({row["new_token_count"] for row in rows}, {2, 3})
        calls = len(self.calls)
        self.assertEqual(self.run_worker()["new_cases"], 0)
        self.assertEqual(len(self.calls), calls)


if __name__ == "__main__":
    unittest.main()
