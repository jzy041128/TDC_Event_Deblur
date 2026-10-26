import copy


def checkpoint_metadata(model, model_config):
    return {
        "model_config": copy.deepcopy(model_config),
        "resolved_model_config": copy.deepcopy(model.checkpoint_config),
        "fusion_mode": model.fusion_mode,
        "fusion_options": copy.deepcopy(model.fusion_options),
        "event_in": model.event_in,
        "base_dim": model.base_dim,
        "tdc_kernel_size": model.tdc_kernel_size,
        "precision": "float32",
    }


def validate_pretrain_options(paths):
    flag = paths.get("pretrain_backbone_only", False)
    if type(flag) is not bool:
        raise ValueError("path.pretrain_backbone_only must be boolean")
    configured = lambda value: value not in (None, "", "~")
    if flag and (not configured(paths.get("pretrain_model")) or configured(paths.get("resume_state"))):
        raise ValueError("pretrain_backbone_only requires pretrain_model and cannot be used with resume_state")
    return flag


def load_pretrained_weights(model, checkpoint, backbone_only=False):
    model.validate_event3d_checkpoint(checkpoint, allow_fusion_transfer=backbone_only)
    state = checkpoint["model_state_dict"]
    if not backbone_only:
        model.load_state_dict(state, strict=True)
        return "Loaded complete model weights (strict=True)."
    # Partial loading is allowed only for an explicit initialization transfer, never resume.
    retained = {key: value for key, value in state.items() if not key.startswith("fusions.")}
    expected = {key: value for key, value in model.state_dict().items() if not key.startswith("fusions.")}
    missing, unexpected = set(expected) - set(retained), set(retained) - set(expected)
    mismatched = [key for key in set(expected) & set(retained) if retained[key].shape != expected[key].shape]
    if missing or unexpected or mismatched:
        raise ValueError(f"Backbone initialization mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}, shapes={sorted(mismatched)}")
    result = model.load_state_dict(retained, strict=False)
    if result.unexpected_keys or any(not key.startswith("fusions.") for key in result.missing_keys):
        raise ValueError("Only fusions.* may be missing during backbone-only initialization")
    return f"Loaded {len(retained)} backbone tensors; all fusions.* excluded. New optimizer/epoch 0."


def fusion_description(model):
    if model.fusion_options is None:
        return None
    stages = [type(fusion.temporal_fusion).__name__ for fusion in model.fusions]
    opt = model.fusion_options
    dims = [model.base_dim * 2 ** i for i in range(3)]
    temporal_widths = [max(c // opt["temporal_reduction"], opt["temporal_min_dim"]) for c in dims]
    local_widths = [max(c // opt["local_st_reduction"], opt["local_st_min_dim"]) for c in dims]
    local_stages = [i for i, stage in enumerate(model.fusions)
                    if any(type(module).__name__ == "LocalSTRead" for module in stage.temporal_fusion.modules())]
    return (
        f"Temporal fusion stages: {stages} | E3 input: full 5D, no temporal mean | "
        f"temporal Q/K/V widths by stage={temporal_widths} | "
        f"local Q/K/V total widths={local_widths}, heads={opt['local_st_heads']}, "
        f"per-head widths={[width // opt['local_st_heads'] for width in local_widths]}, "
        f"actual local stages={local_stages}, chunk={opt['local_st_query_chunk']}, "
        f"local checkpoint={opt['local_st_checkpoint_chunks'] if local_stages else 'not used'} | "
        "shared initialization reference: RGB-gated 4CA (untrained) | precision: float32"
    )
