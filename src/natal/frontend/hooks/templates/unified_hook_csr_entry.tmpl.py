    # CSR hook_idx=PLACEHOLDER_SCHEDULE_IDX
    _r = execute_single_csr_hook(
        hook_idx=PLACEHOLDER_SCHEDULE_IDX, n_hooks=_HP_N_HOOKS,
        op_offsets=_HP_OP_OFFSETS,
        op_types_data=_HP_OP_TYPES_DATA,
        zidx_offsets_data=_HP_ZIDX_OFFSETS_DATA,
        zidx_data=_HP_ZIDX_DATA,
        age_offsets_data=_HP_AGE_OFFSETS_DATA,
        age_data=_HP_AGE_DATA,
        sex_masks_data=_HP_SEX_MASKS_DATA,
        params_data=_HP_PARAMS_DATA,
        condition_offsets_data=_HP_CONDITION_OFFSETS_DATA,
        condition_types_data=_HP_CONDITION_TYPES_DATA,
        condition_params_data=_HP_CONDITION_PARAMS_DATA,
        deme_selector_types=_HP_DEME_SELECTOR_TYPES,
        deme_selector_offsets=_HP_DEME_SELECTOR_OFFSETS,
        deme_selector_data=_HP_DEME_SELECTOR_DATA,
        individual_count=ind_count,
        sperm_storage=sperm_store,
        has_sperm_storage=PLACEHOLDER_HAS_SPERM,
        tick=tick, stochastic=stochastic,
        continuous_sampling=continuous_sampling, deme_id=deme_id,
    )
    if _r != 0:
        return _r
