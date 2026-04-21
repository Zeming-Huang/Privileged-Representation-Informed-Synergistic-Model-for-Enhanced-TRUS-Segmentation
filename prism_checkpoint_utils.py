def remap_legacy_prism_state_dict_keys(state_dict):
    """
    Normalize older PRISM checkpoint keys to the current public inference names.

    Older checkpoints stored adaptive-fusion weights under:
      cross_modal_extractor.adaptive_fusion.*

    Current public inference code expects:
      cross_modal_extractor.fusion.*
    """

    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("cross_modal_extractor.adaptive_fusion."):
            new_key = key.replace(
                "cross_modal_extractor.adaptive_fusion.",
                "cross_modal_extractor.fusion.",
                1,
            )
        remapped[new_key] = value
    return remapped
