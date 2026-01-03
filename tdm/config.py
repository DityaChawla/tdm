"""
Configuration management for TDM.

Provides centralized configuration with sensible defaults.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal


@dataclass
class TDMConfig:
    """Central configuration for the Triangulated Defection Monitor."""
    
    # Model settings
    model_name: str = "gpt2"
    device: str = "cuda"  # Will fallback to CPU if CUDA unavailable
    
    # Activation capture settings
    target_layers: List[int] = field(default_factory=lambda: [-1])  # Last layer by default
    pooling: Literal["last_token", "mean_last_k"] = "last_token"
    pool_k: int = 4  # For mean_last_k pooling
    
    # Probe settings
    probe_type: Literal["linear", "mahalanobis"] = "linear"
    probe_hidden_dim: Optional[int] = None  # For MLP probes (not used for linear)
    
    # Drift detector settings
    drift_embedder: str = "distilbert-base-uncased"
    drift_buffer_size: int = 100  # Rolling baseline buffer
    
    # Canary settings
    n_canaries: int = 4
    canary_transforms: List[str] = field(default_factory=lambda: [
        "suffix", "math", "whitespace", "reorder"
    ])
    
    # Calibration settings
    alpha: float = 0.05  # False positive budget
    calibration_method: Literal["quantile", "conformal"] = "quantile"
    
    # Fusion settings
    fusion_weights: dict = field(default_factory=lambda: {
        "probe": 0.5,
        "drift": 0.25,
        "canary": 0.25
    })
    fusion_bias: float = 0.0
    use_stacking: bool = False
    
    # Policy settings
    block_threshold: float = 0.8
    reroute_threshold: float = 0.6
    reroute_model: str = "gpt2"  # Smaller/safer model for rerouting
    enable_patching: bool = True
    patching_layer: Optional[int] = None  # Auto-select if None
    
    # TriggerSpan settings
    top_k_spans: int = 3
    max_span_length: int = 10
    
    # Server settings
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    
    # Paths
    artifacts_dir: str = "./artifacts"
    logs_dir: str = "./logs"
    log_file: str = "tdm.jsonl"
    
    # Mode
    mode: Literal["whitebox", "blackbox"] = "whitebox"
    
    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        import torch
        
        # Auto-detect device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        
        # Create directories
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Set patching layer to last target layer if not specified
        if self.patching_layer is None and self.target_layers:
            self.patching_layer = self.target_layers[-1]
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TDMConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @property
    def probe_path(self) -> str:
        return os.path.join(self.artifacts_dir, "probe.pt")
    
    @property
    def probe_unsup_path(self) -> str:
        return os.path.join(self.artifacts_dir, "probe_unsup.pkl")
    
    @property
    def probe_metadata_path(self) -> str:
        return os.path.join(self.artifacts_dir, "probe_metadata.json")
    
    @property
    def thresholds_path(self) -> str:
        return os.path.join(self.artifacts_dir, "thresholds.json")
    
    @property
    def calibration_stats_path(self) -> str:
        return os.path.join(self.artifacts_dir, "calibration_stats.json")
    
    @property
    def stacking_model_path(self) -> str:
        return os.path.join(self.artifacts_dir, "stacking_model.pt")
    
    @property
    def log_path(self) -> str:
        return os.path.join(self.logs_dir, self.log_file)


# Global default config instance
_default_config: Optional[TDMConfig] = None


def get_config() -> TDMConfig:
    """Get the global default configuration."""
    global _default_config
    if _default_config is None:
        _default_config = TDMConfig()
    return _default_config


def set_config(config: TDMConfig):
    """Set the global default configuration."""
    global _default_config
    _default_config = config
