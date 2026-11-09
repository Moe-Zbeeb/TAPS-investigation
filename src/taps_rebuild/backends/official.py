import importlib
import inspect
from pathlib import Path
import sys
import threading
import time


_IMPORT_IDENTITY = None
_RUN_LOCK = threading.Lock()
_MISSING = object()


def _integer(value):
    return int(value.item() if hasattr(value, "item") else value)


def _event(torch):
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


class _Span:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def milliseconds(self):
        return float(self.start.elapsed_time(self.end))


class _Trace:
    def __init__(self, backend, mode):
        self.backend = backend
        self.torch = backend.torch
        self.mode = mode
        self.cycles = []
        self.patches = []
        self.prefill = None
        self.root_proposal = None
        self.pending_draft = None
        self.current = None
        self.initializing = False
        self.update_start = None
        self.previous_target_end = None

    def patch(self, owner, name, replacement):
        previous = vars(owner).get(name, _MISSING)
        self.patches.append((owner, name, previous))
        setattr(owner, name, replacement)

    def restore(self):
        while self.patches:
            owner, name, previous = self.patches.pop()
            if previous is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, previous)

    def timed(self, function, *args, **kwargs):
        start = _event(self.torch)
        result = function(*args, **kwargs)
        end = _event(self.torch)
        return result, _Span(start, end)

    def __enter__(self):
        model = self.backend.model
        try:
            if self.mode == "target_only":
                self.install_target(model)
            else:
                self.install_speculative(model)
        except BaseException:
            self.restore()
            raise
        return self

    def __exit__(self, kind, value, traceback):
        self.restore()

    def install_target(self, model):
        original = model.base_model.forward

        def forward(*args, **kwargs):
            result, span = self.timed(original, *args, **kwargs)
            if self.prefill is None:
                self.prefill = span
            else:
                first = not self.cycles
                self.cycles.append({
                    "cycle_index": len(self.cycles) + 1,
                    "accepted_draft_tokens": 0,
                    "emitted_tokens": 1,
                    "verifier_nodes": 1,
                    "proposed_nodes": 0,
                    "observed_depth": 0,
                    "_tree": span,
                    "_verification": span,
                    "_prefill": self.prefill if first else None,
                    "_root": _Span(self.prefill.end, span.start) if first else None,
                })
            self.previous_target_end = span.end
            return result

        self.patch(model.base_model, "forward", forward)

    def install_speculative(self, model):
        module = self.backend.module
        original_initialize = module.initialize_tree
        original_tree = module.tree_decoding
        original_update = module.update_inference_inputs
        update_signature = inspect.signature(original_update)
        original_forward = model.forward
        original_draft = model.ea_layer.topK_genrate

        def forward(*args, **kwargs):
            if not self.initializing:
                return original_forward(*args, **kwargs)
            if self.prefill is not None:
                raise RuntimeError("Official initialization called target prefill more than once")
            result, self.prefill = self.timed(original_forward, *args, **kwargs)
            return result

        def draft(*args, **kwargs):
            if self.pending_draft is not None:
                raise RuntimeError("An official draft was replaced without verification")
            result, span = self.timed(original_draft, *args, **kwargs)
            self.pending_draft = span
            if self.update_start is not None:
                self.current["_commit"] = _Span(self.update_start, span.start)
            return result

        def initialize_tree(input_ids, model, past_key_values, logits_processor):
            if model is not self.backend.model:
                return original_initialize(input_ids, model, past_key_values, logits_processor)
            self.initializing = True
            try:
                result = original_initialize(input_ids, model, past_key_values, logits_processor)
            finally:
                self.initializing = False
            if self.prefill is None or self.pending_draft is None:
                raise RuntimeError("Official initialization did not expose prefill and drafting")
            self.root_proposal = _Span(self.prefill.end, self.pending_draft.start)
            return result

        def tree_decoding(model, tree_candidates, past_key_values,
                          tree_position_ids, input_ids, retrieve_indices):
            if model is not self.backend.model:
                return original_tree(model, tree_candidates, past_key_values,
                                     tree_position_ids, input_ids, retrieve_indices)
            if self.pending_draft is None:
                raise RuntimeError("Official verification has no preceding draft")
            if self.current is not None and self.current.get("emitted_tokens") is None:
                raise RuntimeError("Official cycle was not committed before the next verification")
            first = not self.cycles
            self.current = {
                "cycle_index": len(self.cycles) + 1,
                "verifier_nodes": int(tree_candidates.shape[-1]),
                "proposed_nodes": int(tree_candidates.shape[-1]) - 1,
                "emitted_tokens": None,
                "_positions": tree_position_ids,
                "_draft": self.pending_draft,
                "_prefill": self.prefill if first else None,
                "_root": self.root_proposal if first else None,
            }
            self.pending_draft = None
            self.cycles.append(self.current)
            result, self.current["_tree"] = self.timed(
                original_tree, model, tree_candidates, past_key_values,
                tree_position_ids, input_ids, retrieve_indices,
            )
            return result

        def update_inference_inputs(*args, **kwargs):
            arguments = update_signature.bind(*args, **kwargs).arguments
            if arguments["model"] is not model:
                return original_update(*args, **kwargs)
            if self.current is None or "_tree" not in self.current:
                raise RuntimeError("Official commit has no preceding verification")
            if self.current["emitted_tokens"] is not None:
                raise RuntimeError("Official cycle was committed more than once")
            start = _event(self.torch)
            self.current["_verification"] = _Span(self.current["_tree"].start, start)
            self.current["_posterior"] = _Span(self.current["_tree"].end, start)
            self.current["_accepted"] = arguments["accept_length"]
            previous_length = arguments["input_ids"].shape[-1]
            self.update_start = start
            try:
                result = original_update(*args, **kwargs)
            finally:
                self.update_start = None
            self.current["emitted_tokens"] = int(result[0].shape[-1] - previous_length)
            return result

        self.patch(model, "forward", forward)
        self.patch(model.ea_layer, "topK_genrate", draft)
        self.patch(module, "initialize_tree", initialize_tree)
        self.patch(module, "tree_decoding", tree_decoding)
        self.patch(module, "update_inference_inputs", update_inference_inputs)

    def resolve(self):
        resolved = []
        timings = {
            "drafting_ms": "_draft",
            "verification_ms": "_verification",
            "target_forward_ms": "_tree",
            "posterior_ms": "_posterior",
            "prefill_ms": "_prefill",
            "root_proposal_ms": "_root",
            "commit_and_next_root_ms": "_commit",
        }
        for cycle in self.cycles:
            item = {key: value for key, value in cycle.items() if not key.startswith("_")}
            if self.mode == "speculative":
                item["accepted_draft_tokens"] = _integer(cycle["_accepted"])
                positions = cycle["_positions"].detach().cpu().tolist()
                item["observed_depth"] = max(positions)
                if item["accepted_draft_tokens"] + 1 != item["emitted_tokens"]:
                    raise RuntimeError("Official accepted length disagrees with committed token count")
            for name, key in timings.items():
                item[name] = cycle[key].milliseconds() if cycle.get(key) is not None else 0.0
            resolved.append(item)
        return resolved


class OfficialBackend:
    def __init__(self, method, source_dir, target_path, draft_path,
                 device="cuda:0", total_token=60, depth=None, top_k=10):
        global _IMPORT_IDENTITY
        if method not in {"eagle3", "eagle2", "hass"}:
            raise ValueError("method must be eagle3, eagle2, or hass")
        if not isinstance(total_token, int) or total_token < 2:
            raise ValueError("total_token must include a root and at least one proposed node")
        depth = (7 if method == "eagle3" else 5) if depth is None else depth
        if not isinstance(depth, int) or depth < 0 or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("depth must be nonnegative and top_k must be positive")
        if total_token - 1 > top_k + depth * top_k * top_k:
            raise ValueError("Tree budget exceeds the nodes generated by depth and top_k")
        self.method = method
        self.source_dir = Path(source_dir).resolve()
        self.target_path = Path(target_path).resolve()
        self.draft_path = Path(draft_path).resolve()
        self.total_token = total_token
        self.depth = depth
        self.top_k = top_k
        self.torch = importlib.import_module("torch")
        self.device = self.torch.device(device)
        if self.device.type != "cuda" or self.device.index not in (None, 0):
            raise ValueError("Expose one GPU and use logical cuda:0")
        self.device = self.torch.device("cuda:0")
        if not self.torch.cuda.is_available() or self.torch.cuda.device_count() != 1:
            raise RuntimeError("OfficialBackend requires exactly one visible CUDA device")
        self.torch.cuda.set_device(self.device)
        identity = (method, str(self.source_dir))
        if _IMPORT_IDENTITY is not None and _IMPORT_IDENTITY != identity:
            raise RuntimeError("Use a separate process for each official method and source checkout")
        name = "model.ea_model" if method == "hass" else "eagle.model.ea_model"
        expected = self.source_dir / Path(*name.split(".")).with_suffix(".py")
        if not expected.is_file():
            raise ValueError(f"Official source entrypoint is missing: {expected}")
        for path in (self.target_path, self.draft_path):
            if not (path / "config.json").is_file():
                raise ValueError(f"A prepared local model directory is required: {path}")
        previous_path = sys.path[:]
        try:
            sys.path.insert(0, str(self.source_dir))
            self.module = importlib.import_module(name)
        finally:
            sys.path[:] = previous_path
        if Path(self.module.__file__).resolve() != expected:
            raise RuntimeError("An official module from another checkout is already imported")
        _IMPORT_IDENTITY = identity
        required = {
            "initialize_tree": ("input_ids", "model", "past_key_values", "logits_processor"),
            "tree_decoding": ("model", "tree_candidates", "past_key_values", "tree_position_ids", "input_ids", "retrieve_indices"),
            "update_inference_inputs": ("input_ids", "candidates", "best_candidate", "accept_length", "retrieve_indices", "logits_processor", "new_token", "past_key_values_data_list", "current_length_data", "model", "hidden_state_new", "sample_p"),
        }
        for function, parameters in required.items():
            if tuple(inspect.signature(getattr(self.module, function)).parameters) != parameters:
                raise RuntimeError(f"The official {function} signature differs from the inspected source")
        arguments = {
            "base_model_path": str(self.target_path),
            "ea_model_path": str(self.draft_path),
            "total_token": total_token,
            "depth": depth,
            "top_k": top_k,
            "torch_dtype": self.torch.float16,
            "low_cpu_mem_usage": True,
            "device_map": {"": 0},
            "local_files_only": True,
        }
        if method != "hass":
            arguments["use_eagle3"] = method == "eagle3"
        self.model = self.module.EaModel.from_pretrained(**arguments)
        self.model.eval()
        self._cache_max_length = None

    @property
    def tokenizer(self):
        if self.model is None:
            raise RuntimeError("The official backend is closed")
        return self.model.get_tokenizer()

    def generate(self, input_ids, *, temperature, max_new_tokens=512,
                 max_length=2048, generation_mode="speculative"):
        if self.model is None:
            raise RuntimeError("The official backend is closed")
        if generation_mode not in {"speculative", "target_only"}:
            raise ValueError("generation_mode must be speculative or target_only")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("Official generation requires one nonempty tokenized prompt")
        if temperature < 0 or max_new_tokens < 1 or max_length <= self.total_token + 10:
            raise ValueError("Invalid generation temperature or length budget")
        if self.method == "hass" and max_length > self.model.base_model.config.max_position_embeddings:
            raise ValueError("The requested length exceeds the pinned HASS target-config cache capacity")
        actual_tree_nodes = self.model.ea_layer.total_tokens + 1
        physical_capacity = self.model.base_model.config.max_position_embeddings if self.method == "hass" else max_length
        if input_ids.shape[1] + actual_tree_nodes > physical_capacity:
            raise ValueError("Prompt plus the official tree exceeds physical KV cache capacity; no truncation was applied")
        if self._cache_max_length is not None and max_length != self._cache_max_length:
            raise ValueError("Create a new backend when changing the official cache length")
        if not _RUN_LOCK.acquire(blocking=False):
            raise RuntimeError("Concurrent official generation in one process is unsupported")
        try:
            with self.torch.cuda.device(self.device), self.torch.inference_mode():
                prepared = input_ids.clone().to(self.device)
                self.torch.cuda.synchronize(self.device)
                trace = _Trace(self, generation_mode)
                started = time.perf_counter()
                function = self.model.eagenerate if generation_mode == "speculative" else self.model.naivegenerate
                self._cache_max_length = max_length
                with trace:
                    result = function(
                        prepared, temperature=temperature, max_new_tokens=max_new_tokens,
                        max_length=max_length, log=True, is_llama3=True,
                    )
                self.torch.cuda.synchronize(self.device)
                wall_total_ms = (time.perf_counter() - started) * 1000
                if not isinstance(result, tuple) or len(result) < 3:
                    raise RuntimeError("The official generator returned an unexpected structure")
                output_ids, official_count, last_iteration = result[:3]
                if output_ids.ndim != 2 or output_ids.shape[0] != 1:
                    raise RuntimeError("The official generator returned a non-single-example output")
                actual_count = int(output_ids.shape[1] - prepared.shape[1])
                cycles = trace.resolve()
                if sum(cycle["emitted_tokens"] for cycle in cycles) != actual_count:
                    raise RuntimeError("Instrumented cycles disagree with the actual output token count")
                last_iteration = _integer(last_iteration)
                if last_iteration + 1 != len(cycles):
                    raise RuntimeError("Instrumented cycles disagree with the official last iteration")
                if self.method == "hass" and generation_mode == "speculative" and len(result) > 3:
                    if [_integer(value) for value in result[3]] != [cycle["accepted_draft_tokens"] for cycle in cycles]:
                        raise RuntimeError("Instrumented acceptance disagrees with the HASS acceptance log")
                completion = output_ids[0, prepared.shape[1]:].detach().cpu().tolist()
                stop_ids = {self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")}
                first_stop = next((index for index, token in enumerate(completion) if token in stop_ids), None)
                official_count = _integer(official_count)
                effective_limit = max_length - self.model.ea_layer.total_tokens - 10
                return {
                    "output_ids": output_ids,
                    "new_token_count": actual_count,
                    "official_new_token_count": official_count,
                    "official_minus_actual_token_count": official_count - actual_count,
                    "truecycle_count": len(cycles),
                    "official_last_iteration": last_iteration,
                    "cycles": cycles,
                    "wall_total_ms": wall_total_ms,
                    "generation_mode": generation_mode,
                    "method": self.method,
                    "unused_drafter_resident": generation_mode == "target_only",
                    "target_only_memory_baseline": False,
                    "unused_final_drafting_ms": trace.pending_draft.milliseconds() if trace.pending_draft is not None else 0.0,
                    "first_stop_token_index": first_stop,
                    "post_stop_token_count": actual_count - first_stop - 1 if first_stop is not None else 0,
                    "cap_overshoot_tokens": max(0, actual_count - max_new_tokens),
                    "effective_context_limit": effective_limit,
                    "effective_context_overshoot_tokens": max(0, output_ids.shape[1] - effective_limit),
                    "cuda_events_resolved_after_generation_sync": True,
                    "root_proposal_includes_feature_setup": True,
                    "verification_includes_posterior": generation_mode == "speculative",
                    "context_guard": {
                        "physical_capacity": physical_capacity,
                        "declared_max_length": max_length,
                        "effective_context_limit": effective_limit,
                        "prompt_tokens": int(prepared.shape[1]),
                        "verifier_node_budget": actual_tree_nodes,
                        "initial_physical_headroom_tokens": physical_capacity - prepared.shape[1] - actual_tree_nodes,
                        "initial_exceeds_effective_context_limit": prepared.shape[1] > effective_limit,
                    },
                    "seed_control": "caller",
                }
        except BaseException:
            try:
                self.torch.cuda.synchronize(self.device)
            except Exception:
                pass
            raise
        finally:
            _RUN_LOCK.release()

    def close(self):
        if not _RUN_LOCK.acquire(blocking=False):
            raise RuntimeError("Cannot close an official backend during generation")
        try:
            self.model = None
            with self.torch.cuda.device(self.device):
                self.torch.cuda.empty_cache()
        finally:
            _RUN_LOCK.release()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
