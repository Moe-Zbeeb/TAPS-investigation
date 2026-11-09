from bisect import bisect_right
from dataclasses import dataclass, replace
import copy
import hashlib
import importlib
import json
import math
import random
import re


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _log_probability(value):
    value = float(value)
    if math.isnan(value) or value > 0:
        raise ValueError("Path log probabilities must be nonpositive and not NaN")
    return value


def _labels(examples, split, forbidden=()):
    if split not in {"dev", "test"} or not examples:
        raise ValueError("Verification examples require an explicit dev or test split")
    rows = []
    seen = set()
    for example in examples:
        key = example["example_id"]
        if not isinstance(key, str) or not key or key in seen or key in forbidden:
            raise ValueError("Verification example IDs must be unique and disjoint from forbidden examples")
        if example.get("split") != split or example.get("label_source") != "target_verification":
            raise ValueError("Calibration labels must come from the declared verification split")
        if example["accepted"] not in (0, 1):
            raise ValueError("Verification labels must be binary")
        seen.add(key)
        rows.append((key, _log_probability(example["log_probability"]), int(example["accepted"])))
    return rows


def calibration_diagnostics(probabilities, accepted, bins=10):
    if type(bins) is not int or bins < 1 or len(probabilities) != len(accepted) or not probabilities:
        raise ValueError("Calibration diagnostics require paired nonempty probabilities and labels")
    buckets = [[] for _ in range(bins)]
    squared = 0.0
    log_loss = 0.0
    for probability, label in zip(probabilities, accepted):
        if not math.isfinite(probability) or not 0 <= probability <= 1 or label not in (0, 1):
            raise ValueError("Calibration probabilities or labels are invalid")
        buckets[min(bins - 1, int(probability * bins))].append((probability, label))
        squared += (probability - label) ** 2
        clipped = min(1 - 1e-12, max(1e-12, probability))
        log_loss -= label * math.log(clipped) + (1 - label) * math.log1p(-clipped)
    details = []
    ece = 0.0
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        confidence = math.fsum(value for value, label in bucket) / len(bucket)
        rate = math.fsum(label for value, label in bucket) / len(bucket)
        ece += len(bucket) * abs(confidence - rate) / len(probabilities)
        details.append({"bin": index, "count": len(bucket), "mean_probability": confidence, "acceptance_rate": rate})
    return {"count": len(probabilities), "brier_score": squared / len(probabilities), "ece": ece, "log_loss": log_loss / len(probabilities), "bins": details}


@dataclass(frozen=True)
class Calibrator:
    expert_id: str
    method: str
    dataset_identity: str
    example_ids: tuple
    forbidden_ids_identity: str
    fitted_split: str = "dev"
    temperature: float = 1.0
    knots: tuple = ()
    values: tuple = ()
    fit_metrics: tuple = ()

    def __post_init__(self):
        if self.fitted_split != "dev" or not isinstance(self.expert_id, str) or not self.expert_id or not self.example_ids or len(set(self.example_ids)) != len(self.example_ids):
            raise ValueError("A calibrator must be fitted on identified development examples")
        if not re.fullmatch("[0-9a-f]{64}", self.dataset_identity):
            raise ValueError("Calibration data requires an immutable content identity")
        if self.method not in {"temperature", "isotonic"} or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid calibration method or temperature")
        if self.method == "isotonic" and (not self.knots or len(self.knots) != len(self.values)):
            raise ValueError("Isotonic calibration requires matched knots and values")
        if self.method == "isotonic" and (list(self.knots) != sorted(set(self.knots)) or any(not math.isfinite(value) or not 0 <= value <= 1 for value in self.values) or list(self.values) != sorted(self.values)):
            raise ValueError("Isotonic knots and probabilities must be ordered and valid")

    def predict(self, log_probability):
        value = _log_probability(log_probability)
        if self.method == "temperature":
            return math.exp(value / self.temperature)
        return self.values[max(0, bisect_right(self.knots, value) - 1)]

    def metadata(self):
        value = {
            "expert_id": self.expert_id, "method": self.method,
            "dataset_identity": self.dataset_identity, "fitted_split": self.fitted_split,
            "example_ids": list(self.example_ids), "forbidden_ids_identity": self.forbidden_ids_identity,
            "label_source": "target_verification", "temperature": self.temperature,
            "knots": [None if point == -math.inf else point for point in self.knots],
            "values": list(self.values), "fit_metrics_brier_ece_log_loss": list(self.fit_metrics),
        }
        return {**value, "calibrator_identity": _identity(value)}

    @classmethod
    def from_metadata(cls, value):
        identity = value["calibrator_identity"]
        if _identity({key: item for key, item in value.items() if key != "calibrator_identity"}) != identity or value.get("label_source") != "target_verification":
            raise ValueError("Calibrator metadata identity or label origin differs")
        return cls(expert_id=value["expert_id"], method=value["method"], dataset_identity=value["dataset_identity"],
                   example_ids=tuple(value["example_ids"]), forbidden_ids_identity=value["forbidden_ids_identity"],
                   fitted_split=value["fitted_split"], temperature=value["temperature"],
                   knots=tuple(-math.inf if point is None else point for point in value["knots"]),
                   values=tuple(value["values"]), fit_metrics=tuple(value["fit_metrics_brier_ece_log_loss"]))


def fit_calibrator(expert_id, examples, *, method="temperature", split="dev", dataset_identity, forbidden_ids=()):
    if split != "dev":
        raise ValueError("Calibrators may only be fitted on development verification examples")
    forbidden_ids = tuple(forbidden_ids)
    rows = _labels(examples, split, set(forbidden_ids))
    arguments = dict(expert_id=expert_id, method=method, dataset_identity=dataset_identity,
                     example_ids=tuple(sorted(row[0] for row in rows)), forbidden_ids_identity=_identity(sorted(forbidden_ids)))
    if method == "temperature":
        def objective(log_temperature):
            temperature = math.exp(log_temperature)
            probabilities = [math.exp(logp / temperature) for _, logp, _ in rows]
            return calibration_diagnostics(probabilities, [label for _, _, label in rows])["log_loss"]
        lower, upper = math.log(0.05), math.log(20.0)
        ratio = (math.sqrt(5) - 1) / 2
        left, right = upper - ratio * (upper - lower), lower + ratio * (upper - lower)
        for _ in range(80):
            if objective(left) <= objective(right):
                upper, right = right, left
                left = upper - ratio * (upper - lower)
            else:
                lower, left = left, right
                right = lower + ratio * (upper - lower)
        candidates = [math.log(0.05), math.log(20.0), (lower + upper) / 2]
        best = min(candidates, key=lambda value: (objective(value), value))
        calibrator = Calibrator(**arguments, temperature=math.exp(best))
    elif method == "isotonic":
        grouped = {}
        for _, logp, label in rows:
            count, total = grouped.get(logp, (0, 0))
            grouped[logp] = count + 1, total + label
        knots = sorted(grouped)
        blocks = []
        for index, point in enumerate(knots):
            count, total = grouped[point]
            blocks.append([index, index + 1, count, total])
            while len(blocks) > 1 and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]:
                second, first = blocks.pop(), blocks.pop()
                blocks.append([first[0], second[1], first[2] + second[2], first[3] + second[3]])
        values = [0.0] * len(knots)
        for start, end, count, total in blocks:
            values[start:end] = [total / count] * (end - start)
        calibrator = Calibrator(**arguments, knots=tuple(knots), values=tuple(values))
    else:
        raise ValueError("Unknown calibration method")
    diagnostics = calibration_diagnostics([calibrator.predict(logp) for _, logp, _ in rows], [label for _, _, label in rows])
    return replace(calibrator, fit_metrics=tuple(diagnostics[key] for key in ("brier_score", "ece", "log_loss")))


def evaluate_calibrator(calibrator, examples, *, split, dataset_identity):
    rows = _labels(examples, split)
    if not re.fullmatch("[0-9a-f]{64}", dataset_identity):
        raise ValueError("Evaluation data requires an immutable content identity")
    overlap = bool(set(row[0] for row in rows) & set(calibrator.example_ids))
    if split == "test" and overlap:
        raise ValueError("Held-out calibration diagnostics overlap fitting examples")
    diagnostics = calibration_diagnostics([calibrator.predict(logp) for _, logp, _ in rows], [label for _, _, label in rows])
    return {**diagnostics, "split": split, "dataset_identity": dataset_identity,
            "overlaps_fitting_examples": overlap,
            "calibrator_identity": calibrator.metadata()["calibrator_identity"]}


def pooled_budget_tree(drafts, *, root_token, root_source, budget, calibrators=None, score_mode="estimated"):
    if type(root_token) is not int or root_token < 0 or root_source not in {"target_sampled", "target_greedy"}:
        raise ValueError("The common root must be supplied by target sampling or greedy decoding")
    if type(budget) is not int or budget < 1 or score_mode not in {"estimated", "oracle", "uncalibrated_diagnostic"}:
        raise ValueError("Invalid fixed node budget or score mode")
    if not drafts or len({draft["expert_id"] for draft in drafts}) != len(drafts):
        raise ValueError("Draft expert IDs must be nonempty and unique")
    calibrators = calibrators or {}
    pool = {(): {"score": 1.0, "provenance": []}}
    projections = 0
    calibration = {}
    for draft in drafts:
        expert = draft["expert_id"]
        if not isinstance(expert, str) or not expert or draft["root_token"] != root_token or draft["root_source"] != root_source:
            raise ValueError("Every expert must use the same target-selected root assumption")
        if score_mode == "estimated":
            calibrator = calibrators.get(expert)
            if not isinstance(calibrator, Calibrator) or calibrator.expert_id != expert:
                raise ValueError("Estimated scores require a development-fitted calibrator per expert")
            calibration[expert] = calibrator.metadata()
        elif score_mode == "oracle" and calibrators:
            raise ValueError("Oracle properties must use exact probabilities without fitted calibrators")
        nodes = {}
        for node in draft["nodes"]:
            path = tuple(node["path"])
            if not path or any(type(token) is not int or token < 0 for token in path) or path in nodes:
                raise ValueError("Expert node paths must be nonempty, valid and unique")
            nodes[path] = _log_probability(node["log_probability"])
        if any(path[:-1] and path[:-1] not in nodes for path in nodes):
            raise ValueError("Each expert candidate tree must be prefix closed")
        scores = {(): 1.0}
        for path in sorted(nodes, key=lambda value: (len(value), value)):
            raw = nodes[path]
            probability = calibrators[expert].predict(raw) if score_mode == "estimated" else math.exp(raw)
            parent_score = scores[path[:-1]]
            if score_mode == "oracle" and probability > parent_score + 1e-12:
                raise ValueError("Exact target path probabilities must be monotone along ancestry")
            projected = min(parent_score, probability)
            projections += int(projected < probability)
            scores[path] = projected
            provenance = {"expert_id": expert, "raw_log_probability": None if raw == -math.inf else raw,
                          "calibrated_probability": probability, "projected_probability": projected}
            if path in pool:
                if score_mode == "oracle" and not math.isclose(pool[path]["score"], probability, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("Experts disagree on an allegedly exact target path probability")
                pool[path]["score"] = max(pool[path]["score"], projected)
                pool[path]["provenance"].append(provenance)
            else:
                pool[path] = {"score": projected, "provenance": [provenance]}
    if len(pool) < budget:
        raise ValueError("The unique candidate pool cannot fill the fixed node budget")
    selected = sorted(pool, key=lambda path: (-pool[path]["score"], len(path), path))[:budget]
    if () not in selected or any(path and path[:-1] not in selected for path in selected):
        raise RuntimeError("Monotone score selection produced a prefix hole")
    selected.sort(key=lambda path: (len(path), path))
    indices = {path: index for index, path in enumerate(selected)}
    parents = [-1 if not path else indices[path[:-1]] for path in selected]
    mask = []
    for index in range(len(selected)):
        row = [False] * len(selected)
        ancestor = index
        while ancestor >= 0:
            row[ancestor] = True
            ancestor = parents[ancestor]
        mask.append(row)
    leaves = [index for index in range(len(selected)) if index not in parents]
    retrieves = []
    for leaf in leaves:
        path = []
        while leaf >= 0:
            path.append(leaf)
            leaf = parents[leaf]
        retrieves.append(list(reversed(path)))
    width = max(map(len, retrieves))
    tree = {
        "tokens": [root_token if not path else path[-1] for path in selected],
        "parent_indices": parents, "depth": [len(path) for path in selected],
        "paths_after_root": [list(path) for path in selected],
        "tree_attention_mask": mask, "retrieve_paths": [path + [-1] * (width - len(path)) for path in retrieves],
        "retrieve_padding_value": -1, "scores": [pool[path]["score"] for path in selected],
        "provenance": [pool[path]["provenance"] for path in selected],
        "metadata": {"budget": budget, "budget_includes_root": True, "root_source": root_source,
                     "score_mode": score_mode, "score_aggregation": "maximum_across_calibrated_expert_estimates",
                     "oracle_not_available_at_inference": score_mode == "oracle", "estimated_optimality_claimed": False,
                     "monotonic_projections": projections, "calibrators": calibration,
                     "unique_pool_nodes": len(pool), "selected_score_sum_excluding_root": math.fsum(pool[path]["score"] for path in selected if path),
                     "scientific_result": False},
    }
    validate_tree(tree)
    return tree


def validate_tree(tree):
    tokens, parents, depths, paths = (tree[key] for key in ("tokens", "parent_indices", "depth", "paths_after_root"))
    count = len(tokens)
    if count < 1 or count != tree["metadata"]["budget"] or any(len(values) != count for values in (parents, depths, paths, tree["scores"], tree["provenance"], tree["tree_attention_mask"])):
        raise ValueError("Tree arrays must fill the same fixed budget")
    if parents[0] != -1 or depths[0] != 0 or paths[0] != [] or len({tuple(path) for path in paths}) != count:
        raise ValueError("Tree must contain one root and unique complete token prefixes")
    for index in range(count):
        if type(tokens[index]) is not int or tokens[index] < 0:
            raise ValueError("Tree token IDs must be nonnegative integers")
        if index and (not 0 <= parents[index] < index or depths[index] != depths[parents[index]] + 1 or paths[index] != paths[parents[index]] + [tokens[index]]):
            raise ValueError("Tree ancestry contains a prefix hole or invalid parent")
        if not math.isfinite(tree["scores"][index]) or not 0 <= tree["scores"][index] <= 1 or (index and tree["scores"][index] > tree["scores"][parents[index]] + 1e-12):
            raise ValueError("Tree scores must be valid monotone path probabilities")
        expected = [False] * count
        cursor = index
        while cursor >= 0:
            expected[cursor] = True
            cursor = parents[cursor]
        if tree["tree_attention_mask"][index] != expected:
            raise ValueError("Tree attention must contain exactly each node's ancestors")
    leaves = {index for index in range(count) if index not in parents}
    if tree.get("retrieve_padding_value") != -1 or len({len(row) for row in tree["retrieve_paths"]}) != 1:
        raise ValueError("Retrieval paths must be rectangular with explicit -1 padding")
    observed = set()
    for retrieve in tree["retrieve_paths"]:
        if any(type(index) is not int or not -1 <= index < count for index in retrieve):
            raise ValueError("Retrieval indices are outside the flat tree")
        valid = [index for index in retrieve if index != -1]
        if retrieve != valid + [-1] * (len(retrieve) - len(valid)) or not valid or valid[0] != 0 or valid[-1] in observed or valid[-1] not in leaves:
            raise ValueError("Invalid padded retrieval path")
        if any(parents[child] != parent for parent, child in zip(valid, valid[1:])):
            raise ValueError("Retrieval path contains a non-ancestral edge")
        observed.add(valid[-1])
    if observed != leaves:
        raise ValueError("Retrieval paths must cover every selected leaf exactly once")
    return True


def _normalise_python(probabilities):
    values = list(map(float, probabilities))
    total = math.fsum(values)
    if not values or any(not math.isfinite(value) or value < 0 for value in values) or not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Target probabilities must form a finite normalized distribution")
    return [value / total for value in values]


def _children(children, vocabulary):
    result = []
    for token in children:
        if type(token) is not int or not 0 <= token < vocabulary:
            raise ValueError("Candidate child is outside the target vocabulary")
        if token not in result:
            result.append(token)
    return result


def exact_sequential_distribution(probabilities, children):
    remaining = _normalise_python(probabilities)
    children = _children(children, len(remaining))
    outcome = [0.0] * len(remaining)
    reach = 1.0
    for token in children:
        mass = math.fsum(remaining)
        if mass <= 0:
            break
        acceptance = remaining[token] / mass
        outcome[token] += reach * acceptance
        reach *= 1 - acceptance
        remaining[token] = 0.0
    mass = math.fsum(remaining)
    if mass > 0:
        for token, probability in enumerate(remaining):
            outcome[token] += reach * probability / mass
    return outcome


def sample_target_children(probabilities, children, *, greedy=False, rng=None, generator=None):
    is_tensor = hasattr(probabilities, "detach") and hasattr(probabilities, "device")
    if is_tensor:
        torch = importlib.import_module("torch")
        if probabilities.ndim != 1 or not probabilities.is_floating_point():
            raise ValueError("Target probability tensors must be one-dimensional floating point")
        values = probabilities.detach().to(dtype=torch.float64)
        mass = values.sum()
        if not bool(torch.isfinite(values).all().item()) or bool((values < 0).any().item()) or not math.isclose(float(mass.item()), 1.0, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("Target tensor probabilities must form a normalized distribution")
        values = values / mass
        children = _children(children, int(values.shape[0]))
        if greedy:
            token = int(values.argmax().item())
            return {"token": token, "accepted": token in children, "tried_children": children, "uses_draft_probabilities": False}
        remaining = values.clone()
        tried = []
        for token in children:
            tried.append(token)
            mass = remaining.sum()
            if bool((torch.rand((), device=values.device, generator=generator) < remaining[token] / mass).item()):
                return {"token": token, "accepted": True, "tried_children": tried, "uses_draft_probabilities": False}
            remaining[token] = 0
        token = int(torch.multinomial(remaining / remaining.sum(), 1, generator=generator).item())
    else:
        values = _normalise_python(probabilities)
        children = _children(children, len(values))
        if greedy:
            token = max(range(len(values)), key=lambda index: (values[index], -index))
            return {"token": token, "accepted": token in children, "tried_children": children, "uses_draft_probabilities": False}
        rng = rng or random
        remaining = values[:]
        tried = []
        for token in children:
            tried.append(token)
            if rng.random() < remaining[token] / math.fsum(remaining):
                return {"token": token, "accepted": True, "tried_children": tried, "uses_draft_probabilities": False}
            remaining[token] = 0
        mass = math.fsum(remaining)
        threshold = rng.random() * mass
        cumulative = 0.0
        positive = [index for index, value in enumerate(remaining) if value > 0]
        token = positive[-1]
        for index in positive:
            cumulative += remaining[index]
            if threshold < cumulative:
                token = index
                break
    return {"token": token, "accepted": False, "tried_children": tried, "uses_draft_probabilities": False}


def verify_tree(tree, target_probabilities, *, greedy=False, rng=None, generator=None):
    validate_tree(tree)
    if len(target_probabilities) != len(tree["tokens"]):
        raise ValueError("Each verified node requires its actual target next-token distribution")
    accepted = []
    current = 0
    while True:
        children = [index for index, parent in enumerate(tree["parent_indices"]) if parent == current]
        tokens = [tree["tokens"][index] for index in children]
        result = sample_target_children(target_probabilities[current], tokens, greedy=greedy, rng=rng, generator=generator)
        if not result["accepted"]:
            return {"accepted_draft_tokens": len(accepted), "accepted_node_indices": accepted,
                    "emitted_tokens": [tree["tokens"][0]] + [tree["tokens"][index] for index in accepted],
                    "next_root_token": result["token"], "next_root_uncommitted": True,
                    "root_already_target_selected": True, "uses_draft_probabilities": False,
                    "implementation_scope": "fixed_candidate_correctness_reference", "scientific_result": False}
        current = children[tokens.index(result["token"])]
        accepted.append(current)


def _python_shape(value):
    if isinstance(value, list):
        if not value:
            raise ValueError("Empty Python parameter lists need an explicit tensor dtype")
        children = [_python_shape(item) for item in value]
        if any(child[0] != children[0][0] or child[2] != children[0][2] for child in children):
            raise ValueError("Python state parameters must have rectangular shapes and one dtype")
        return (len(value),) + children[0][0], sum((child[1] for child in children), []), children[0][2]
    if type(value) not in (int, float, bool) or not math.isfinite(value):
        raise ValueError("State parameters must be finite numeric values")
    return (), [value], type(value).__name__


def _reshape_python(values, shape):
    if not shape:
        return values[0]
    stride = math.prod(shape[1:])
    return [_reshape_python(values[index * stride:(index + 1) * stride], shape[1:]) for index in range(shape[0])]


def merge_weights(ancestor, experts, *, ancestor_identity, expert_ancestors, method="linear", weights=None,
                  scale=1.0, density=0.2, seed=0, diagnostic_independent=False):
    if method not in {"linear", "task_arithmetic", "ties", "dare"} or not experts or not ancestor:
        raise ValueError("A supported merge and nonempty ancestor/expert states are required")
    names = sorted(experts)
    if set(names) != set(expert_ancestors) or not re.fullmatch("[0-9a-f]{64}", ancestor_identity):
        raise ValueError("Every expert requires its shared ancestor content identity")
    shared = all(expert_ancestors[name] == ancestor_identity for name in names)
    if not shared and not diagnostic_independent:
        raise ValueError("Independent initializations require an explicit diagnostic flag")
    if any(not re.fullmatch("[0-9a-f]{64}", expert_ancestors[name]) for name in names):
        raise ValueError("Expert ancestor identities must be immutable content hashes")
    if not math.isfinite(scale) or not 0 < density <= 1 or type(seed) is not int or seed < 0 or type(diagnostic_independent) is not bool:
        raise ValueError("Invalid merge scale, retained density or seed")
    weights = {name: 1.0 for name in names} if weights is None else weights
    if set(weights) != set(names) or any(not math.isfinite(weights[name]) for name in names):
        raise ValueError("Merge weights must match expert IDs and be finite")
    coefficients = [float(weights[name]) for name in names]
    if method == "linear":
        if any(value < 0 for value in coefficients) or math.fsum(coefficients) <= 0:
            raise ValueError("Linear averaging requires nonnegative weights with positive total")
        total = math.fsum(coefficients)
        coefficients = [value / total for value in coefficients]
    keys = sorted(ancestor)
    if any(set(experts[name]) != set(keys) for name in names):
        raise ValueError("All merged state dictionaries must have identical keys")
    tensor_mode = hasattr(ancestor[keys[0]], "detach")
    schema = {}
    floating = []
    output = {}
    torch = importlib.import_module("torch") if tensor_mode else None
    for key in keys:
        base = ancestor[key]
        values = [base] + [experts[name][key] for name in names]
        if tensor_mode:
            if any(not hasattr(value, "detach") or value.shape != base.shape or value.dtype != base.dtype for value in values):
                raise ValueError("Tensor keys must match shapes and dtypes")
            is_float = base.is_floating_point()
            schema[key] = tuple(base.shape), base.numel(), str(base.dtype)
            if not is_float and any(not torch.equal(value.to(base.device), base) for value in values[1:]):
                raise ValueError("Nonfloating buffers must be identical across experts")
            output[key] = base.detach().clone()
        else:
            parsed = [_python_shape(value) for value in values]
            if any(value[0] != parsed[0][0] or value[2] != parsed[0][2] for value in parsed):
                raise ValueError("Python state keys must match shapes and dtypes")
            shape, flat, dtype = parsed[0]
            schema[key] = shape, len(flat), dtype
            is_float = dtype == "float"
            if not is_float and any(value != base for value in values[1:]):
                raise ValueError("Nonfloating buffers must be identical across experts")
            output[key] = copy.deepcopy(base)
        if is_float:
            floating.append(key)
    if not floating:
        raise ValueError("A weight merge requires at least one floating-point parameter")
    if tensor_mode:
        device = ancestor[floating[0]].device
        dtype = torch.float64 if any(ancestor[key].dtype == torch.float64 for key in floating) else torch.float32
        def flatten(state):
            return torch.cat([state[key].detach().reshape(-1).to(device=device, dtype=dtype) for key in floating])
        base = flatten(ancestor)
        stack = torch.stack([flatten(experts[name]) for name in names])
        if not bool(torch.isfinite(stack).all().item()) or not bool(torch.isfinite(base).all().item()):
            raise ValueError("Merge tensors must contain finite values")
        coefficients_tensor = torch.tensor(coefficients, device=device, dtype=dtype)[:, None]
        delta = stack - base
        if method == "linear":
            merged = (stack * coefficients_tensor).sum(dim=0)
        else:
            if method == "dare":
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                delta = delta * (torch.rand(delta.shape, device=device, generator=generator) < density) / density
            if method == "ties":
                keep = max(1, math.ceil(delta.shape[1] * density))
                indices = torch.argsort(delta.abs(), dim=1, descending=True, stable=True)[:, :keep]
                mask = torch.zeros_like(delta).scatter_(1, indices, 1)
                delta = delta * mask
            delta = delta * coefficients_tensor
            if method == "ties":
                signs = delta.sum(dim=0).sign()
                signs = torch.where(signs == 0, signs.sum().sign(), signs)
                selected = torch.where(signs[None] > 0, delta > 0, delta < 0)
                update = (delta * selected).sum(dim=0) / selected.sum(dim=0).clamp(min=1)
            else:
                update = delta.sum(dim=0)
            merged = base + scale * update
        offset = 0
        for key in floating:
            shape, size, dtype_name = schema[key]
            output[key] = merged[offset:offset + size].reshape(shape).to(device=ancestor[key].device, dtype=ancestor[key].dtype)
            offset += size
        computation = str(dtype)
    else:
        def flatten(state):
            return sum((_python_shape(state[key])[1] for key in floating), [])
        base = flatten(ancestor)
        stack = [flatten(experts[name]) for name in names]
        delta = [[value - initial for value, initial in zip(row, base)] for row in stack]
        if method == "linear":
            merged = [math.fsum(weight * row[index] for weight, row in zip(coefficients, stack)) for index in range(len(base))]
        else:
            if method == "dare":
                generator = random.Random(seed)
                delta = [[value / density if generator.random() < density else 0.0 for value in row] for row in delta]
            if method == "ties":
                keep = max(1, math.ceil(len(base) * density))
                trimmed = []
                for row in delta:
                    chosen = set(sorted(range(len(row)), key=lambda coordinate: (-abs(row[coordinate]), coordinate))[:keep])
                    trimmed.append([value if index in chosen else 0.0 for index, value in enumerate(row)])
                delta = trimmed
            delta = [[value * weight for value in row] for row, weight in zip(delta, coefficients)]
            if method == "ties":
                sign = lambda value: (value > 0) - (value < 0)
                signs = [sign(math.fsum(row[index] for row in delta)) for index in range(len(base))]
                majority = sign(sum(signs))
                signs = [value or majority for value in signs]
                update = []
                for index, chosen in enumerate(signs):
                    selected = [row[index] for row in delta if row[index] > 0] if chosen > 0 else [row[index] for row in delta if row[index] < 0]
                    update.append(math.fsum(selected) / len(selected) if selected else 0.0)
            else:
                update = [math.fsum(row[index] for row in delta) for index in range(len(base))]
            merged = [value + scale * change for value, change in zip(base, update)]
        offset = 0
        for key in floating:
            shape, size, dtype_name = schema[key]
            output[key] = _reshape_python(merged[offset:offset + size], shape)
            offset += size
        computation = "python_float64"
    return {"state_dict": output, "metadata": {"method": method, "ancestor_identity": ancestor_identity,
            "expert_ancestors": dict(expert_ancestors), "shared_ancestor": shared,
            "diagnostic_independent_initializations": not shared, "expert_order": names,
            "weights": dict(zip(names, coefficients)), "scale": scale, "retained_density": density, "seed": seed,
            "parameter_order": floating, "parameter_schema": schema, "computation_dtype": computation,
            "ties_trim": "global_top_ceil_density_stable_coordinate_ties", "ties_sign": "mass_global_majority_then_negative_if_tied",
            "dare_aggregation": "task_arithmetic", "requires_reproduction_gate": True, "scientific_result": False}}
