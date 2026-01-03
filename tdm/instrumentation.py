"""
Activation capture and model instrumentation for TDM.

Provides hooks to capture residual stream activations from transformer models.
"""

from typing import Dict, List, Optional, Tuple, Literal, Union
from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from tdm.utils import logger, ensure_numpy


# Global cache for loaded models
_model_cache: Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]] = {}


def load_model_and_tokenizer(
    model_name: str = "gpt2",
    device: str = "cuda",
    cache: bool = True
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a HuggingFace causal language model and tokenizer.
    
    Args:
        model_name: Name or path of the model
        device: Device to load model on ("cuda" or "cpu")
        cache: Whether to cache the loaded model
    
    Returns:
        Tuple of (model, tokenizer)
    """
    cache_key = f"{model_name}_{device}"
    
    if cache and cache_key in _model_cache:
        logger.debug(f"Using cached model: {model_name}")
        return _model_cache[cache_key]
    
    logger.info(f"Loading model: {model_name} on {device}")
    
    # Handle device availability
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    model = model.to(device)
    
    model.eval()
    
    if cache:
        _model_cache[cache_key] = (model, tokenizer)
    
    logger.info(f"Model loaded: {model.config.n_layer} layers, {model.config.n_embd} dim")
    
    return model, tokenizer


def get_layer_modules(model: PreTrainedModel) -> List[nn.Module]:
    """
    Get the list of transformer layer modules.
    
    Handles different model architectures (GPT-2, LLaMA, etc.)
    """
    # Try common attribute names
    if hasattr(model, 'transformer'):
        # GPT-2 style
        if hasattr(model.transformer, 'h'):
            return list(model.transformer.h)
    elif hasattr(model, 'model'):
        # LLaMA style
        if hasattr(model.model, 'layers'):
            return list(model.model.layers)
    elif hasattr(model, 'gpt_neox'):
        # GPT-NeoX style
        if hasattr(model.gpt_neox, 'layers'):
            return list(model.gpt_neox.layers)
    
    raise ValueError(f"Unknown model architecture: {type(model)}")


def normalize_layer_indices(layers: List[int], n_layers: int) -> List[int]:
    """Convert negative indices to positive and validate."""
    normalized = []
    for layer in layers:
        if layer < 0:
            layer = n_layers + layer
        if 0 <= layer < n_layers:
            normalized.append(layer)
        else:
            logger.warning(f"Layer index {layer} out of range [0, {n_layers})")
    return normalized


class ActivationCapture:
    """Context manager for capturing activations during forward pass."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        layers: List[int],
        capture_input: bool = False
    ):
        """
        Initialize activation capture.
        
        Args:
            model: The transformer model
            layers: Layer indices to capture (supports negative indexing)
            capture_input: If True, capture layer input; else capture output
        """
        self.model = model
        self.layer_modules = get_layer_modules(model)
        self.n_layers = len(self.layer_modules)
        self.layers = normalize_layer_indices(layers, self.n_layers)
        self.capture_input = capture_input
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
    
    def _hook_fn(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            if self.capture_input:
                # Input is a tuple, first element is the hidden states
                self.activations[layer_idx] = input[0].detach()
            else:
                # Output is typically hidden_states or a tuple
                if isinstance(output, tuple):
                    self.activations[layer_idx] = output[0].detach()
                else:
                    self.activations[layer_idx] = output.detach()
        return hook
    
    def __enter__(self):
        """Register hooks."""
        for layer_idx in self.layers:
            hook = self.layer_modules[layer_idx].register_forward_hook(
                self._hook_fn(layer_idx)
            )
            self.hooks.append(hook)
        return self
    
    def __exit__(self, *args):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get captured activation for a layer."""
        if layer_idx < 0:
            layer_idx = self.n_layers + layer_idx
        return self.activations.get(layer_idx)


def get_residual_activations(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    layers: List[int] = [-1],
    positions: Optional[Union[List[int], str]] = "last",
    pooling: Literal["last_token", "mean_last_k", "all"] = "last_token",
    pool_k: int = 4,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Extract residual stream activations from specified layers and positions.
    
    Args:
        model: The transformer model
        input_ids: Input token IDs [batch, seq_len]
        layers: Layer indices to capture (supports negative indexing)
        positions: Token positions to capture ("last", "all", or list of indices)
        pooling: Pooling strategy for positions
        pool_k: Number of tokens for mean_last_k pooling
        attention_mask: Optional attention mask
    
    Returns:
        Feature tensor [batch, concat_dim] where concat_dim = len(layers) * hidden_dim
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    batch_size, seq_len = input_ids.shape
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    with ActivationCapture(model, layers) as capture:
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
    
    # Collect and pool activations
    features_list = []
    
    for layer_idx in layers:
        # Get activation [batch, seq, hidden]
        act = capture.get(layer_idx)
        if act is None:
            raise ValueError(f"Failed to capture activation for layer {layer_idx}")
        
        # Pool over positions
        if pooling == "last_token":
            # Get last non-padded token for each sample
            if attention_mask is not None:
                # Find last valid position
                lengths = attention_mask.sum(dim=1).long() - 1
                pooled = act[torch.arange(batch_size), lengths]
            else:
                pooled = act[:, -1, :]  # [batch, hidden]
        
        elif pooling == "mean_last_k":
            k = min(pool_k, seq_len)
            if attention_mask is not None:
                # Mask and mean
                mask = attention_mask[:, -k:].unsqueeze(-1).float()
                pooled = (act[:, -k:, :] * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = act[:, -k:, :].mean(dim=1)  # [batch, hidden]
        
        elif pooling == "all":
            # Flatten all positions
            pooled = act.view(batch_size, -1)  # [batch, seq * hidden]
        
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        features_list.append(pooled)
    
    # Concatenate features from all layers
    features = torch.cat(features_list, dim=-1)  # [batch, len(layers) * hidden]
    
    return features


def get_token_activations(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    layer: int = -1,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Get activations for all tokens at a specific layer.
    
    Args:
        model: The transformer model
        input_ids: Input token IDs [batch, seq_len] or [seq_len]
        layer: Layer index
        attention_mask: Optional attention mask
    
    Returns:
        Activation tensor [batch, seq_len, hidden_dim]
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    with ActivationCapture(model, [layer]) as capture:
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
    
    act = capture.get(layer)
    if act is None:
        n_layers = len(get_layer_modules(model))
        if layer < 0:
            layer = n_layers + layer
        raise ValueError(f"Failed to capture activation for layer {layer}")
    
    return act


@contextmanager
def activation_patching_hook(
    model: PreTrainedModel,
    layer: int,
    patch_fn,
    positions: Optional[List[int]] = None
):
    """
    Context manager for patching activations during forward pass.
    
    Args:
        model: The transformer model
        layer: Layer index to patch
        patch_fn: Function (activation) -> patched_activation
        positions: Optional list of positions to patch (None = all)
    """
    layer_modules = get_layer_modules(model)
    n_layers = len(layer_modules)
    
    if layer < 0:
        layer = n_layers + layer
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        if positions is not None:
            # Only patch specific positions
            patched = hidden_states.clone()
            for pos in positions:
                patched[:, pos, :] = patch_fn(hidden_states[:, pos, :])
            hidden_states = patched
        else:
            hidden_states = patch_fn(hidden_states)
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states
    
    hook = layer_modules[layer].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        hook.remove()


def clear_model_cache():
    """Clear the model cache to free memory."""
    global _model_cache
    _model_cache.clear()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
