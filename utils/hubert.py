"""HuBERT loading helpers with PyTorch weight-norm compatibility."""

from __future__ import annotations

import torch


DEFAULT_HUBERT_MODEL = "facebook/hubert-large-ls960-ft"
HUBERT_CACHE_VERSION = 3


def _repair_legacy_weight_norm(model, model_name, loading_info):
    expected_missing = {
        "hubert.encoder.pos_conv_embed.conv.parametrizations.weight.original0",
        "hubert.encoder.pos_conv_embed.conv.parametrizations.weight.original1",
    }
    missing = set(loading_info.get("missing_keys", []))
    unexpected_missing = missing - expected_missing
    if unexpected_missing:
        raise RuntimeError(f"HuBERT checkpoint has unexpected missing weights: {sorted(unexpected_missing)}")
    if not (missing & expected_missing):
        return False

    # Older Hugging Face checkpoints store weight_g/weight_v, while newer
    # PyTorch exposes the same weight_norm tensors as parametrization originals.
    from transformers.modeling_utils import load_state_dict
    from transformers.utils import WEIGHTS_NAME
    from transformers.utils.hub import cached_file

    archive_path = cached_file(model_name, WEIGHTS_NAME)
    checkpoint = load_state_dict(archive_path)
    magnitude = checkpoint["hubert.encoder.pos_conv_embed.conv.weight_g"]
    direction = checkpoint["hubert.encoder.pos_conv_embed.conv.weight_v"]
    convolution = model.encoder.pos_conv_embed.conv
    with torch.no_grad():
        convolution.parametrizations.weight.original0.copy_(magnitude)
        convolution.parametrizations.weight.original1.copy_(direction)
    del checkpoint
    return True


def load_hubert_components(model_name=DEFAULT_HUBERT_MODEL, device=None):
    """Load the processor and a fully initialized HuBERT base model."""
    from transformers import HubertModel, Wav2Vec2Processor
    from transformers.utils import logging as transformers_logging

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    previous_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        model, loading_info = HubertModel.from_pretrained(
            model_name, output_loading_info=True
        )
    finally:
        transformers_logging.set_verbosity(previous_verbosity)

    repaired = _repair_legacy_weight_norm(model, model_name, loading_info)
    if repaired:
        print("Repaired HuBERT positional-convolution weight_norm keys")
    if device is not None:
        model = model.to(device)
    model.eval()
    return processor, model
