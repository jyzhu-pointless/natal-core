    if PLACEHOLDER_NJIT_GUARD_CONDITION:
        _r = PLACEHOLDER_NJIT_FN_NAME(state, config, deme_id)
        if _r != 0:
            return _r
