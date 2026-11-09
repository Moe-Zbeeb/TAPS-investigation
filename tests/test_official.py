import contextlib
import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taps_rebuild.backends import official


class Tensor:
    def __init__(self, data, device="cpu"):
        self.data = copy.deepcopy(data)
        self.device = device
        self.shape = (len(data), len(data[0])) if data and isinstance(data[0], list) else (len(data),)
        self.ndim = len(self.shape)

    def clone(self):
        return Tensor(self.data, self.device)

    def to(self, device):
        self.device = device
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.data)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return Tensor(self.data[index[0]][index[1]], self.device)
        return Tensor(self.data[index], self.device)


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class CUDA:
    def __init__(self):
        self.clock = 0
        self.synchronized = False
        self.synchronizations = 0
        self.resolutions = 0
        self.visible_devices = 1

    def Event(self, enable_timing):
        cuda = self

        class Event:
            def record(self):
                cuda.clock += 1
                cuda.synchronized = False
                self.tick = cuda.clock

            def elapsed_time(self, other):
                if not cuda.synchronized:
                    raise AssertionError("An event was resolved before generation synchronization")
                cuda.resolutions += 1
                return other.tick - self.tick

        return Event()

    def is_available(self):
        return True

    def device_count(self):
        return self.visible_devices

    def set_device(self, device):
        self.selected_device = device

    def device(self, device):
        return contextlib.nullcontext()

    def synchronize(self, device):
        self.synchronized = True
        self.synchronizations += 1

    def empty_cache(self):
        pass


class Tokenizer:
    eos_token_id = 2

    def convert_tokens_to_ids(self, token):
        return 3


class Layer:
    def __init__(self, total_token, depth):
        self.total_tokens = total_token - 1
        self.depth = depth
        self.calls = 0
        self.resets = 0

    def reset_kv(self):
        self.resets += 1

    def topK_genrate(self, *args, **kwargs):
        self.calls += 1
        tokens = Tensor([[100] + list(range(self.total_tokens))])
        positions = Tensor([0] + [(index % (self.depth + 1)) + 1 for index in range(self.total_tokens)])
        return tokens, None, None, positions


class Base:
    config = types.SimpleNamespace(max_position_embeddings=8192)

    def forward(self, *args, **kwargs):
        return types.SimpleNamespace(logits=None)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Model:
    def __init__(self, module, arguments, method):
        self.module = module
        self.arguments = arguments
        self.method = method
        self.ea_layer = Layer(arguments["total_token"], arguments["depth"])
        self.base_model = Base()
        self.plan = [[11, 12, 13], [2]]
        self.target_plan = [11, 12, 2]
        self.counter_bias = 0
        self.index_bias = 0
        self.fail = False

    def eval(self):
        return self

    def get_tokenizer(self):
        return Tokenizer()

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def eagenerate(self, input_ids, temperature, max_new_tokens, max_length, log, is_llama3):
        self.ea_layer.reset_kv()
        output = input_ids.clone()
        tree = self.module.initialize_tree(output, self, None, None)
        count = 0
        accepted = []
        for index, tokens in enumerate(self.plan):
            self.module.tree_decoding(self, tree[0], None, tree[3], output, None)
            if self.fail:
                raise RuntimeError("injected official failure")
            length = len(tokens) - 1
            result = self.module.update_inference_inputs(
                output, Tensor([tokens]), 0, Scalar(length), None, None,
                count, None, None, self, None, None,
            )
            output = result[0]
            tree = result[1:5]
            count = result[5]
            accepted.append(length)
            if 2 in tokens or 3 in tokens or count > max_new_tokens:
                break
        result = (output, Scalar(count + self.counter_bias), index + self.index_bias)
        return result + (accepted,) if self.method == "hass" else result

    def naivegenerate(self, input_ids, temperature, max_new_tokens, max_length, log, is_llama3):
        self.ea_layer.reset_kv()
        output = input_ids.clone()
        self.base_model(output)
        for index, token in enumerate(self.target_plan):
            self.base_model(Tensor([[token]]))
            output = Tensor([output.tolist()[0] + [token]])
            if token in (2, 3) or index + 1 > max_new_tokens:
                break
        return output, index + 1, index


class OfficialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.target = self.root / "target"
        self.draft = self.root / "draft"
        for directory in (self.target, self.draft):
            directory.mkdir()
            (directory / "config.json").write_text("{}")
        self.cuda = CUDA()
        self.torch = types.SimpleNamespace(
            cuda=self.cuda, float16="float16", inference_mode=contextlib.nullcontext,
            device=lambda value: types.SimpleNamespace(type=value.split(":")[0], index=int(value.split(":")[1]) if ":" in value else None),
        )
        self.identity_patch = patch.object(official, "_IMPORT_IDENTITY", None)
        self.identity_patch.start()
        self.import_patch = patch.object(official.importlib, "import_module", side_effect=self.import_module)
        self.import_patch.start()

    def tearDown(self):
        self.import_patch.stop()
        self.identity_patch.stop()
        self.temporary.cleanup()

    def import_module(self, name):
        return self.torch if name == "torch" else self.module

    def make_backend(self, method="eagle3"):
        name = "model.ea_model" if method == "hass" else "eagle.model.ea_model"
        entrypoint = self.source / Path(*name.split(".")).with_suffix(".py")
        entrypoint.parent.mkdir(parents=True, exist_ok=True)
        entrypoint.write_text("")
        self.module = types.ModuleType(name)
        self.module.__file__ = str(entrypoint)

        def initialize_tree(input_ids, model, past_key_values, logits_processor):
            model.forward(input_ids)
            return model.ea_layer.topK_genrate(None, input_ids, None, None) + (None, None, None)

        def tree_decoding(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices):
            model.forward(tree_candidates)
            return None, None, None

        def update_inference_inputs(input_ids, candidates, best_candidate, accept_length,
                                    retrieve_indices, logits_processor, new_token,
                                    past_key_values_data_list, current_length_data, model,
                                    hidden_state_new, sample_p):
            tokens = candidates.tolist()[0]
            output = Tensor([input_ids.tolist()[0] + tokens])
            tree = model.ea_layer.topK_genrate(None, output, None, None)
            return (output,) + tree + (new_token + len(tokens), None, None)

        self.module.initialize_tree = initialize_tree
        self.module.tree_decoding = tree_decoding
        self.module.update_inference_inputs = update_inference_inputs
        self.originals = (initialize_tree, tree_decoding, update_inference_inputs)
        self.module.EaModel = types.SimpleNamespace(from_pretrained=lambda **arguments: Model(self.module, arguments, method))
        return official.OfficialBackend(method, self.source, self.target, self.draft)

    def assert_restored(self, backend):
        self.assertEqual(self.originals, (self.module.initialize_tree, self.module.tree_decoding, self.module.update_inference_inputs))
        self.assertNotIn("forward", vars(backend.model))
        self.assertNotIn("forward", vars(backend.model.base_model))
        self.assertNotIn("topK_genrate", vars(backend.model.ea_layer))

    def test_speculative_cycles_count_consumed_and_unused_drafts_once(self):
        backend = self.make_backend()
        prompt = Tensor([[50, 51]])
        result = backend.generate(prompt, temperature=0, max_new_tokens=3)
        self.assertEqual(prompt.tolist(), [[50, 51]])
        self.assertEqual(result["output_ids"].tolist(), [[50, 51, 11, 12, 13, 2]])
        self.assertEqual(result["new_token_count"], 4)
        self.assertEqual(result["truecycle_count"], 2)
        self.assertEqual([c["accepted_draft_tokens"] for c in result["cycles"]], [2, 0])
        self.assertEqual([c["emitted_tokens"] for c in result["cycles"]], [3, 1])
        self.assertEqual([c["verifier_nodes"] for c in result["cycles"]], [60, 60])
        self.assertEqual([c["proposed_nodes"] for c in result["cycles"]], [59, 59])
        self.assertEqual([c["observed_depth"] for c in result["cycles"]], [8, 8])
        self.assertEqual([c["prefill_ms"] for c in result["cycles"]], [1, 0])
        self.assertEqual([c["drafting_ms"] for c in result["cycles"]], [1, 1])
        self.assertEqual([c["root_proposal_ms"] for c in result["cycles"]], [1, 0])
        for cycle in result["cycles"]:
            self.assertEqual(cycle["verification_ms"], cycle["target_forward_ms"] + cycle["posterior_ms"])
        self.assertEqual(result["unused_final_drafting_ms"], 1)
        self.assertTrue(result["verification_includes_posterior"])
        self.assertEqual(backend.model.ea_layer.calls, 3)
        self.assertEqual(result["cap_overshoot_tokens"], 1)
        self.assertEqual(result["first_stop_token_index"], 3)
        self.assertEqual(self.cuda.synchronizations, 2)
        self.assertGreater(self.cuda.resolutions, 0)
        self.assert_restored(backend)

    def test_target_only_uses_official_naive_path_and_marks_resident_draft(self):
        backend = self.make_backend()
        result = backend.generate(Tensor([[50, 51]]), temperature=0, generation_mode="target_only")
        self.assertEqual(result["truecycle_count"], 3)
        self.assertEqual(result["new_token_count"], 3)
        self.assertTrue(result["unused_drafter_resident"])
        self.assertFalse(result["target_only_memory_baseline"])
        self.assertFalse(result["verification_includes_posterior"])
        self.assertEqual(backend.model.ea_layer.calls, 0)
        self.assertEqual(backend.model.ea_layer.resets, 1)
        self.assertEqual([c["accepted_draft_tokens"] for c in result["cycles"]], [0, 0, 0])
        self.assertEqual([c["prefill_ms"] for c in result["cycles"]], [1, 0, 0])
        self.assertEqual([c["verification_ms"] for c in result["cycles"]], [1, 1, 1])
        self.assertEqual(result["unused_final_drafting_ms"], 0)
        self.assert_restored(backend)

    def test_raw_eos_overshoot_and_counter_disagreement_are_preserved(self):
        backend = self.make_backend()
        backend.model.plan = [[11, 2, 13]]
        backend.model.counter_bias = 2
        result = backend.generate(Tensor([[50]]), temperature=1)
        self.assertEqual(result["output_ids"].tolist(), [[50, 11, 2, 13]])
        self.assertEqual(result["new_token_count"], 3)
        self.assertEqual(result["official_new_token_count"], 5)
        self.assertEqual(result["official_minus_actual_token_count"], 2)
        self.assertEqual(result["post_stop_token_count"], 1)
        self.assertEqual(result["truecycle_count"], 1)

    def test_exception_restores_hooks_and_releases_generation_lock(self):
        backend = self.make_backend()
        backend.model.fail = True
        with self.assertRaisesRegex(RuntimeError, "injected official failure"):
            backend.generate(Tensor([[50]]), temperature=0)
        self.assert_restored(backend)
        backend.model.fail = False
        result = backend.generate(Tensor([[50]]), temperature=0)
        self.assertEqual(result["truecycle_count"], 2)

    def test_official_index_is_checked_against_instrumented_cycles(self):
        backend = self.make_backend()
        backend.model.index_bias = 1
        with self.assertRaisesRegex(RuntimeError, "official last iteration"):
            backend.generate(Tensor([[50]]), temperature=0)
        self.assert_restored(backend)

    def test_hass_defaults_and_official_acceptance_log(self):
        backend = self.make_backend("hass")
        self.assertEqual(backend.depth, 5)
        self.assertNotIn("use_eagle3", backend.model.arguments)
        result = backend.generate(Tensor([[50]]), temperature=1)
        self.assertEqual([c["observed_depth"] for c in result["cycles"]], [6, 6])
        self.assertEqual(backend.model.arguments["torch_dtype"], "float16")
        self.assertTrue(backend.model.arguments["local_files_only"])

    def test_eagle2_selects_original_drafter_and_separate_method_process(self):
        backend = self.make_backend("eagle2")
        self.assertFalse(backend.model.arguments["use_eagle3"])
        self.assertEqual(backend.depth, 5)
        with self.assertRaisesRegex(RuntimeError, "separate process"):
            official.OfficialBackend("eagle3", self.source, self.target, self.draft)

    def test_one_gpu_and_fixed_cache_length_are_enforced(self):
        backend = self.make_backend()
        backend.generate(Tensor([[50]]), temperature=0)
        with self.assertRaisesRegex(ValueError, "cache length"):
            backend.generate(Tensor([[50]]), temperature=0, max_length=1024)
        self.cuda.visible_devices = 2
        with self.assertRaisesRegex(RuntimeError, "exactly one visible"):
            official.OfficialBackend("eagle3", self.source, self.target, self.draft)

    def test_context_wrappers_do_not_instrument_another_model(self):
        backend = self.make_backend()
        other = Model(self.module, backend.model.arguments, "eagle3")
        with official._Trace(backend, "speculative") as trace:
            self.module.initialize_tree(Tensor([[50]]), model=other, past_key_values=None, logits_processor=None)
            self.assertIsNone(trace.prefill)
            self.assertIsNone(trace.pending_draft)
            self.assertEqual(trace.cycles, [])
        self.assert_restored(backend)

    def test_repeated_modes_reset_trace_and_preserve_official_kv_reset(self):
        backend = self.make_backend()
        for mode, expected in [("speculative", 2), ("target_only", 3), ("speculative", 2)]:
            result = backend.generate(Tensor([[50]]), temperature=0, generation_mode=mode)
            self.assertEqual(result["truecycle_count"], expected)
            self.assert_restored(backend)
        self.assertEqual(backend.model.ea_layer.resets, 3)

    def test_physical_context_overflow_is_rejected_without_truncating(self):
        backend = self.make_backend()
        prompt = Tensor([list(range(2000))])
        with self.assertRaisesRegex(ValueError, "physical KV cache capacity"):
            backend.generate(prompt, temperature=0)
        self.assertEqual(prompt.shape[1], 2000)
        self.assertEqual(backend.model.ea_layer.resets, 0)


if __name__ == "__main__":
    unittest.main()
