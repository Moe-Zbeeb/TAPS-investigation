import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import threading

from .official import OfficialBackend
from ..composition import Calibrator, pooled_budget_tree, validate_tree


CNETS_SHA256 = "b0e2512db4529c9689ffd768e9a2812059a5715d786686a131a751c56fbf3292"
_CAPTURE_LOCK = threading.Lock()
_MISSING = object()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_native_scores(torch, layer, function, *args, **kwargs):
    if not _CAPTURE_LOCK.acquire(blocking=False):
        raise RuntimeError("Concurrent native score capture is unsupported")
    original = torch.topk
    previous = vars(torch).get("topk", _MISSING)
    owner = threading.get_ident()
    expected_pool = layer.top_k + layer.depth * layer.top_k * layer.top_k
    captured = []

    def topk(values, *arguments, **options):
        result = original(values, *arguments, **options)
        count = options.get("k", arguments[0] if arguments else None)
        dimension = options.get("dim", arguments[1] if len(arguments) > 1 else -1)
        if threading.get_ident() == owner and values.ndim == 1 and values.shape[0] == expected_pool and count == layer.total_tokens and dimension in (-1, 0):
            captured.append((values, result.indices))
        return result

    try:
        torch.topk = topk
        result = function(*args, **kwargs)
    finally:
        if previous is _MISSING:
            delattr(torch, "topk")
        else:
            torch.topk = previous
        _CAPTURE_LOCK.release()
    if len(captured) != 1:
        raise RuntimeError("The inspected native final top-k selection was not observed exactly once")
    scores, indices = captured[0]
    values = scores.detach().cpu().tolist()
    selected = sorted(indices.detach().cpu().tolist())
    return result, [float(values[index]) for index in selected]


def native_candidate_tree(result, joint_logs, *, expert_id, root_token, root_source):
    if not isinstance(result, tuple) or len(result) != 4:
        raise ValueError("Native EAGLE-3 drafting must return its four official buffers")
    tokens, retrieve, mask, positions = result
    count = len(joint_logs) + 1
    if tuple(tokens.shape) != (1, count) or tuple(mask.shape) != (1, 1, count, count) or tuple(positions.shape) != (count,) or retrieve.ndim != 2:
        raise ValueError("Native EAGLE-3 draft buffer shapes differ from the selected node budget")
    tokens = tokens[0].detach().cpu().tolist()
    positions = positions.detach().cpu().tolist()
    rows = mask[0, 0].detach().cpu().tolist()
    if tokens[0] != root_token or positions[0] != 0:
        raise ValueError("A native expert changed the shared target-selected root")
    parents = [-1]
    paths = [[]]
    for index in range(1, count):
        ancestors = [parent for parent, allowed in enumerate(rows[index]) if bool(allowed)]
        ordered = sorted(ancestors, key=lambda parent: positions[parent])
        if positions[index] != len(ordered) - 1 or not ordered or ordered[0] != 0 or ordered[-1] != index or [positions[parent] for parent in ordered] != list(range(len(ordered))):
            raise ValueError("Native ancestry or relative position IDs are not a valid tree")
        parent = ordered[-2]
        if parent >= index:
            raise ValueError("Native flattened candidates are not in ancestral order")
        parents.append(parent)
        paths.append([tokens[node] for node in ordered[1:]])
    scores = [1.0] + [math.exp(value) for value in joint_logs]
    validate_tree({"tokens": tokens, "parent_indices": parents, "depth": positions,
                   "paths_after_root": paths, "scores": scores, "provenance": [[] for _ in tokens],
                   "tree_attention_mask": [[bool(value) for value in row] for row in rows],
                   "retrieve_paths": retrieve.detach().cpu().tolist(), "retrieve_padding_value": -1,
                   "metadata": {"budget": count}})
    return {"expert_id": expert_id, "root_token": root_token, "root_source": root_source,
            "nodes": [{"path": path, "log_probability": value} for path, value in zip(paths[1:], joint_logs)]}


class PooledEagle3Backend:
    def __init__(self, source_dir, target_path, drafts, *, target_identity, draft_identities,
                 expert_target_identities, calibrators=None, engineering_uncalibrated=False,
                 device="cuda:0", total_token=60, depth=7, top_k=10):
        self.source_dir = Path(source_dir).resolve()
        self.target_path = Path(target_path).resolve()
        if _sha(self.source_dir / "eagle/model/cnets.py") != CNETS_SHA256:
            raise ValueError("Native score capture requires the exact inspected official EAGLE-3 source")
        self.experts = sorted(drafts)
        if not 1 <= len(self.experts) <= 7 or any(not isinstance(name, str) or not name for name in self.experts):
            raise ValueError("One to seven uniquely identified EAGLE-3 experts are required")
        if not re.fullmatch("[0-9a-f]{64}", target_identity) or set(self.experts) != set(draft_identities) or set(self.experts) != set(expert_target_identities):
            raise ValueError("Every expert requires pinned draft and target identities")
        if any(expert_target_identities[name] != target_identity or not re.fullmatch("[0-9a-f]{64}", draft_identities[name]) for name in self.experts):
            raise ValueError("Pooled experts must share one verified target identity and backbone")
        target_config = json.loads((self.target_path / "config.json").read_text())
        for name in self.experts:
            configuration = json.loads((Path(drafts[name]) / "config.json").read_text())
            if any(configuration.get(key) != target_config.get(key) for key in ("hidden_size", "vocab_size")) or not 0 < configuration.get("draft_vocab_size", 0) <= target_config["vocab_size"]:
                raise ValueError("Every expert must be an EAGLE-3 draft for the shared target architecture")
        self.calibrators = calibrators or {}
        if self.calibrators:
            if set(self.calibrators) != set(self.experts) or any(not isinstance(self.calibrators[name], Calibrator) or self.calibrators[name].expert_id != name for name in self.experts):
                raise ValueError("Development calibration must identify every pooled expert")
            self.score_mode = "estimated"
        elif engineering_uncalibrated is True:
            self.score_mode = "uncalibrated_diagnostic"
        else:
            raise ValueError("Uncalibrated pooling requires an explicit engineering-only flag")
        self.torch = importlib.import_module("torch")
        if not self.torch.cuda.is_available() or self.torch.cuda.device_count() != 1:
            raise RuntimeError("Pooled engineering workers require exactly one visible GPU")
        self.device = self.torch.device(device)
        if self.device.type != "cuda" or self.device.index not in (None, 0):
            raise ValueError("Expose one GPU and use logical cuda:0 for pooled drafting")
        self.device = self.torch.device("cuda:0")
        self.total_token = total_token
        self.primary = None
        self.layers = []
        self.native_methods = []
        self.reset_methods = []
        self.proposals = []
        self._lock = threading.Lock()
        self._patches = []
        self.torch.cuda.reset_peak_memory_stats(self.device)
        try:
            for index, name in enumerate(self.experts):
                backend = OfficialBackend("eagle3", self.source_dir, self.target_path, drafts[name], device=device, total_token=total_token, depth=depth, top_k=top_k)
                if index == 0:
                    self.primary = backend
                layer = backend.model.ea_layer
                self.layers.append(layer)
                self.native_methods.append(layer.topK_genrate)
                self.reset_methods.append(layer.reset_kv)
                if index:
                    backend.close()
            self.torch.cuda.synchronize(self.device)
            initialization_peak = self.torch.cuda.max_memory_allocated(self.device)
            steady = self.torch.cuda.memory_allocated(self.device)
            self.metadata = {
                "method": "pooled_native_eagle3", "scope": "engineering", "scientific_result": False,
                "requires_reproduction_gate": True, "target_identity": target_identity,
                "draft_identities": dict(draft_identities), "expert_target_identities": dict(expert_target_identities),
                "expert_ids": self.experts, "score_mode": self.score_mode, "node_budget_including_root": total_token,
                "source_cnets_sha256": CNETS_SHA256, "verifier_count": 1,
                "initialization_peak_allocated_bytes": initialization_peak,
                "steady_allocated_bytes_after_extra_targets_released": steady,
                "initialization_uses_transient_extra_target": len(self.experts) > 1,
                "host_candidate_pooling_and_score_transfers_included_in_generation_time": True,
                "target_probabilities_used_for_proposal_selection": False,
                "calibrators": {name: value.metadata() for name, value in self.calibrators.items()},
            }
            primary_layer = self.primary.model.ea_layer
            for attribute, replacement in (("topK_genrate", self._draft), ("reset_kv", self._reset)):
                self._patches.append((primary_layer, attribute, vars(primary_layer).get(attribute, _MISSING)))
                setattr(primary_layer, attribute, replacement)
        except BaseException:
            self._restore()
            self.layers.clear()
            self.native_methods.clear()
            self.reset_methods.clear()
            if self.primary is not None:
                self.primary.close()
            raise

    @property
    def tokenizer(self):
        return self.primary.tokenizer

    def _reset(self):
        for reset in self.reset_methods:
            reset()

    def _draft(self, hidden_states, input_ids, head, logits_processor):
        root_token = int(input_ids[0, -1].item())
        root_source = "target_greedy" if logits_processor is None else "target_sampled"
        candidates = []
        native_mask_dtype = None
        for name, layer, function in zip(self.experts, self.layers, self.native_methods):
            native, logs = capture_native_scores(self.torch, layer, function, hidden_states.clone(), input_ids.clone(), head, logits_processor)
            native_mask_dtype = native[2].dtype
            candidates.append(native_candidate_tree(native, logs, expert_id=name, root_token=root_token, root_source=root_source))
        tree = pooled_budget_tree(candidates, root_token=root_token, root_source=root_source, budget=self.total_token, calibrators=self.calibrators, score_mode=self.score_mode)
        self.proposals.append({"paths_after_root": tree["paths_after_root"], "scores": tree["scores"], "provenance": tree["provenance"], "metadata": tree["metadata"]})
        torch = self.torch
        return (
            torch.tensor(tree["tokens"], dtype=torch.long, device=input_ids.device).unsqueeze(0),
            torch.tensor(tree["retrieve_paths"], dtype=torch.long, device=hidden_states.device),
            torch.tensor(tree["tree_attention_mask"], dtype=native_mask_dtype, device=hidden_states.device).unsqueeze(0).unsqueeze(0),
            torch.tensor(tree["depth"], dtype=torch.long, device=hidden_states.device),
        )

    def generate(self, input_ids, **kwargs):
        if self.primary is None:
            raise RuntimeError("The pooled backend is closed")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent pooled generation is unsupported")
        try:
            self.proposals = []
            self.torch.cuda.reset_peak_memory_stats(self.device)
            result = self.primary.generate(input_ids, **kwargs)
            if kwargs.get("generation_mode", "speculative") == "speculative":
                if len(self.proposals) != result["truecycle_count"] + 1:
                    raise RuntimeError("Pooled proposal count differs from official consumed and final-unused drafting")
                for cycle, proposal in zip(result["cycles"], self.proposals[:-1]):
                    cycle["pooled_tree"] = proposal
            result.update(method="pooled_native_eagle3", composition={**self.metadata,
                          "generation_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device),
                          "resident_drafter_count": len(self.layers),
                          "unused_final_proposal": self.proposals[-1] if self.proposals else None})
            return result
        finally:
            self._lock.release()

    def _restore(self):
        while self._patches:
            owner, name, previous = self._patches.pop()
            if previous is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, previous)

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot close during pooled generation")
        try:
            self._restore()
            self.layers.clear()
            self.native_methods.clear()
            self.reset_methods.clear()
            if self.primary is not None:
                self.primary.close()
                self.primary = None
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
