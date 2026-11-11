def output_budget(prompt_tokens, requested_tokens, cache_tokens, reserved_tree_tokens):
    values = (prompt_tokens, requested_tokens, cache_tokens, reserved_tree_tokens)
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError('Budget dimensions must be positive integers')
    budget = min(requested_tokens, cache_tokens - reserved_tree_tokens - 10 - prompt_tokens)
    if budget < 1:
        raise ValueError('No generation budget remains after the registered context reservation')
    return budget


def comparison_tokens(raw, budget, stop_ids):
    if type(budget) is not int or budget < 1:
        raise ValueError('Comparison budget must be a positive integer')
    if not isinstance(raw, list) or any(type(token) is not int or token < 0 for token in raw):
        raise ValueError('Raw completion must contain nonnegative integer token IDs')
    stop = next((index + 1 for index, token in enumerate(raw) if token in stop_ids), None)
    if len(raw) < budget and stop is None:
        raise ValueError('Generation ended before the predetermined budget without a terminal token')
    return raw[:min(budget, stop if stop is not None else len(raw))]
