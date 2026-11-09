import copy
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taps_rebuild.backends import pooled


class Tensor:
    def __init__(self, data, dtype="float32", device="cuda:0"):
        self.data = copy.deepcopy(data)
        self.dtype = dtype
        self.device = device
        value = self.data
        dimensions = []
        while isinstance(value, list):
            dimensions.append(len(value))
            value = value[0] if value else None
        self.shape = tuple(dimensions)
        self.ndim = len(self.shape)

    def clone(self):
        return Tensor(self.data, self.dtype, self.device)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.data)

    def item(self):
        return self.data

    def unsqueeze(self, axis):
        if axis != 0:
            raise AssertionError("Unexpected test tensor dimension")
        return Tensor([self.data], self.dtype, self.device)

    def __getitem__(self, indices):
        value = self.data
        for index in indices if isinstance(indices, tuple) else (indices,):
            value = value[index]
        return Tensor(value, self.dtype, self.device)


class CUDA:
    def __init__(self):
        self.active = 0
        self.peak = 0

    def is_available(self):
        return True

    def device_count(self):
        return 1

    def reset_peak_memory_stats(self, device):
        self.peak = self.active

    def max_memory_allocated(self, device):
        return self.peak

    def memory_allocated(self, device):
        return self.active

    def synchronize(self, device):
        pass


class Torch:
    long = "int64"

    def __init__(self):
        self.cuda = CUDA()

    def device(self, value):
        return types.SimpleNamespace(type=value.split(":")[0], index=int(value.split(":")[1]) if ":" in value else None)

    def tensor(self, values, dtype, device):
        return Tensor(values, dtype, device)

    def topk(self, values, k, dim=-1):
        selected = sorted(range(len(values.data)), key=lambda index: (-values.data[index], index))[:k]
        return types.SimpleNamespace(indices=Tensor(selected), values=Tensor([values.data[index] for index in selected]))


class Layer:
    def __init__(self, torch, total_token, depth, top_k):
        self.torch = torch
        self.total_tokens = total_token - 1
        self.depth = depth
        self.top_k = top_k
        self.calls = 0
        self.resets = 0
        self.omit_capture = False

    def reset_kv(self):
        self.resets += 1

    def topK_genrate(self, hidden_states, input_ids, head, logits_processor):
        self.calls += 1
        if not self.omit_capture:
            self.torch.topk(Tensor([math.log(value) for value in [0.6, 0.3, 0.48, 0.01, 0.01, 0.01]]), self.total_tokens, dim=-1)
        root = input_ids[0, -1].item()
        return (
            Tensor([[root, 1, 2, 3]], "int64"),
            Tensor([[0, 2, -1], [0, 1, 3]], "int64"),
            Tensor([[[[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 1]]]]),
            Tensor([0, 1, 1, 2], "int64"),
        )


class Backend:
    def __init__(self, owner, method, source_dir, target_path, draft_path, **kwargs):
        if method != "eagle3":
            raise AssertionError("Pooled implementation mixed official methods")
        self.owner = owner
        self.layer = Layer(owner.torch, kwargs["total_token"], kwargs["depth"], kwargs["top_k"])
        self.model = types.SimpleNamespace(ea_layer=self.layer)
        self.tokenizer = "shared-tokenizer"
        self.closed = False
        owner.instances.append(self)
        owner.torch.cuda.active += 17
        owner.torch.cuda.peak = max(owner.torch.cuda.peak, owner.torch.cuda.active)

    def generate(self, input_ids, generation_mode="speculative", **kwargs):
        self.model.ea_layer.reset_kv()
        if generation_mode == "speculative":
            native_input = Tensor([input_ids.tolist()[0] + [9]])
            first = self.model.ea_layer.topK_genrate(Tensor([[[1.0]]]), native_input, None, None)
            self.owner.last_buffers = first
            self.model.ea_layer.topK_genrate(Tensor([[[1.0]]]), native_input, None, None)
            cycles = [{"accepted_draft_tokens": 2, "emitted_tokens": 3, "verifier_nodes": 4, "proposed_nodes": 3}]
        else:
            cycles = [{"accepted_draft_tokens": 0, "emitted_tokens": 1} for _ in range(3)]
        return {"output_ids": Tensor([input_ids.tolist()[0] + [9, 1, 3]], "int64"),
                "truecycle_count": len(cycles), "cycles": cycles,
                "generation_mode": generation_mode, "new_token_count": 3,
                "unused_final_drafting_ms": 0.1 if generation_mode == "speculative" else 0}

    def close(self):
        if not self.closed:
            self.closed = True
            self.model = None
            self.owner.torch.cuda.active -= 16


class PooledTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.target = self.root / "target"
        self.draft = self.root / "draft"
        file = self.source / "eagle/model/cnets.py"
        file.parent.mkdir(parents=True)
        file.write_text("isolated native fixture")
        for directory, contents in [(self.target, '{"hidden_size":4,"vocab_size":8}'), (self.draft, '{"hidden_size":4,"vocab_size":8,"draft_vocab_size":4}')]:
            directory.mkdir()
            (directory / "config.json").write_text(contents)
        self.torch = Torch()
        self.instances = []
        self.last_buffers = None
        self.addCleanup(patch.stopall)
        patch.object(pooled, "CNETS_SHA256", hashlib.sha256(file.read_bytes()).hexdigest()).start()
        patch.object(pooled.importlib, "import_module", return_value=self.torch).start()
        patch.object(pooled, "OfficialBackend", side_effect=lambda *args, **kwargs: Backend(self, *args, **kwargs)).start()

    def make_backend(self, **overrides):
        arguments = dict(target_identity="a" * 64, draft_identities={"a": "b" * 64, "b": "b" * 64},
                         expert_target_identities={"a": "a" * 64, "b": "a" * 64},
                         engineering_uncalibrated=True, total_token=4, depth=1, top_k=2)
        arguments.update(overrides)
        return pooled.PooledEagle3Backend(self.source, self.target, {"a": self.draft, "b": self.draft}, **arguments)

    def test_selected_joint_logs_follow_native_sorted_node_indices(self):
        layer = Layer(self.torch, 4, 1, 2)
        original = self.torch.topk
        result, scores = pooled.capture_native_scores(self.torch, layer, layer.topK_genrate, Tensor([[[1.0]]]), Tensor([[50, 9]]), None, None)
        self.assertEqual(self.torch.topk, original)
        candidate = pooled.native_candidate_tree(result, scores, expert_id="a", root_token=9, root_source="target_greedy")
        self.assertEqual([node["path"] for node in candidate["nodes"]], [[1], [2], [1, 3]])
        for actual, expected in zip(scores, [0.6, 0.3, 0.48]):
            self.assertAlmostEqual(math.exp(actual), expected)

    def test_identical_released_drafts_pool_once_and_reuse_one_verifier(self):
        backend = self.make_backend()
        primary = self.instances[0]
        self.assertTrue(self.instances[1].closed)
        self.assertIsNotNone(primary.model)
        result = backend.generate(Tensor([[50]]), temperature=0)
        self.assertEqual(result["method"], "pooled_native_eagle3")
        proposal = result["cycles"][0]["pooled_tree"]
        self.assertEqual(proposal["metadata"]["unique_pool_nodes"], 4)
        self.assertEqual(len(proposal["paths_after_root"]), 4)
        self.assertEqual({item["expert_id"] for item in proposal["provenance"][1]}, {"a", "b"})
        self.assertEqual(tuple(self.last_buffers[2].shape), (1, 1, 4, 4))
        self.assertEqual([layer.calls for layer in backend.layers], [2, 2])
        self.assertEqual([layer.resets for layer in backend.layers], [1, 1])
        self.assertEqual(result["composition"]["verifier_count"], 1)
        self.assertGreater(result["composition"]["initialization_peak_allocated_bytes"], result["composition"]["steady_allocated_bytes_after_extra_targets_released"])
        self.assertFalse(result["composition"]["scientific_result"])
        self.assertIsNotNone(result["composition"]["unused_final_proposal"])
        reference = backend.generate(Tensor([[50]]), temperature=0, generation_mode="target_only")
        self.assertEqual(reference["output_ids"].tolist(), result["output_ids"].tolist())
        self.assertEqual([layer.resets for layer in backend.layers], [2, 2])
        backend.close()
        self.assertTrue(primary.closed)
        self.assertNotIn("topK_genrate", vars(primary.layer))
        self.assertNotIn("reset_kv", vars(primary.layer))

    def test_capture_failure_restores_global_hook_and_can_retry(self):
        backend = self.make_backend()
        original = self.torch.topk
        backend.layers[0].omit_capture = True
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            backend.generate(Tensor([[50]]), temperature=0)
        self.assertEqual(self.torch.topk, original)
        backend.layers[0].omit_capture = False
        self.assertEqual(backend.generate(Tensor([[50]]), temperature=0)["truecycle_count"], 1)
        backend.close()

    def test_shared_target_and_calibration_guards_precede_model_load(self):
        with self.assertRaisesRegex(ValueError, "share one verified target"):
            self.make_backend(expert_target_identities={"a": "a" * 64, "b": "c" * 64})
        with self.assertRaisesRegex(ValueError, "engineering-only"):
            self.make_backend(engineering_uncalibrated=False)
        self.assertEqual(self.instances, [])

    def test_uninspected_source_is_rejected(self):
        (self.source / "eagle/model/cnets.py").write_text("modified native source")
        with self.assertRaisesRegex(ValueError, "exact inspected"):
            self.make_backend()
        self.assertEqual(self.instances, [])


if __name__ == "__main__":
    unittest.main()
