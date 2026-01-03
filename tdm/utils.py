"""
Utility functions for TDM.

Provides common helpers for logging, file I/O, and data processing.
"""

import os
import json
import logging
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import torch
import numpy as np


# Configure logging
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup and return the TDM logger."""
    logger = logging.getLogger("tdm")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logging()


def log_request(
    log_path: str,
    prompt: str,
    risk: float,
    decision: str,
    signals: Dict[str, float],
    thresholds: Dict[str, float],
    triggerspans: Optional[List[Dict]] = None,
    causal_validation: Optional[Dict] = None,
    mode: str = "whitebox",
    latency_ms: Optional[float] = None,
    metadata: Optional[Dict] = None
) -> None:
    """
    Log a scored request to JSONL file.
    
    Args:
        log_path: Path to the JSONL log file
        prompt: The input prompt
        risk: Final risk score
        decision: Decision made (ALLOW/BLOCK/REROUTE/PATCHED_RESPONSE)
        signals: Individual signal scores
        thresholds: Thresholds used
        triggerspans: Localized trigger spans
        causal_validation: Causal validation results
        mode: Operating mode (whitebox/blackbox)
        latency_ms: Request latency in milliseconds
        metadata: Additional metadata
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "mode": mode,
        "risk": float(risk),
        "decision": decision,
        "signals": {k: float(v) for k, v in signals.items()},
        "thresholds": {k: float(v) for k, v in thresholds.items()},
    }
    
    if triggerspans:
        log_entry["triggerspans"] = triggerspans
    
    if causal_validation:
        log_entry["causal_validation"] = causal_validation
    
    if latency_ms is not None:
        log_entry["latency_ms"] = float(latency_ms)
    
    if metadata:
        log_entry["metadata"] = metadata
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
    
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + "\n")


def sigmoid(x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """Numerically stable sigmoid function."""
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    elif isinstance(x, np.ndarray):
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)), 
                       np.exp(x) / (1 + np.exp(x)))
    else:
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    return 1.0 - cosine_similarity(a, b)


def ensure_numpy(x: Union[torch.Tensor, np.ndarray, List]) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    return x


def ensure_tensor(x: Union[torch.Tensor, np.ndarray, List], device: str = "cpu") -> torch.Tensor:
    """Convert input to torch tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    elif isinstance(x, list):
        return torch.tensor(x, device=device)
    return x.to(device)


def load_json(path: str) -> Dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict, path: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def batch_iterator(data: List, batch_size: int):
    """Yield batches from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def tpr_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, target_fpr: float) -> float:
    """
    Compute TPR at a specific FPR threshold.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        target_fpr: Target false positive rate
    
    Returns:
        TPR at the given FPR (or closest achievable)
    """
    from sklearn.metrics import roc_curve
    
    y_true = ensure_numpy(y_true).astype(int)
    y_scores = ensure_numpy(y_scores)
    
    if len(np.unique(y_true)) < 2:
        return 0.0
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Find index where FPR first exceeds target
    idx = np.searchsorted(fpr, target_fpr, side='right')
    if idx == 0:
        return 0.0
    # Use the TPR at the threshold just before exceeding target FPR
    return float(tpr[min(idx, len(tpr) - 1)])


def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute classification metrics including low-FPR metrics for research.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        threshold: Classification threshold (auto-computed if None)
    
    Returns:
        Dictionary with metrics including TPR at FPR=1e-3, 1e-4
    """
    from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
    
    y_true = ensure_numpy(y_true).astype(int)
    y_scores = ensure_numpy(y_scores)
    
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return {
            "auroc": 0.5, "tpr_at_fpr_0001": 0.0, "tpr_at_fpr_001": 0.0,
            "tpr_at_fpr_01": 0.0, "tpr_at_fpr_05": 0.0, "tpr_at_fpr_10": 0.0,
            "precision": 0.0, "recall": 0.0, "threshold": 0.5
        }
    
    auroc = roc_auc_score(y_true, y_scores)
    
    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    
    # Find TPR at specific FPR levels (research-grade: 1e-4, 1e-3)
    def safe_tpr_at_fpr(target_fpr):
        idx = np.searchsorted(fpr, target_fpr, side='right')
        if idx == 0:
            return 0.0
        return float(tpr[min(idx, len(tpr) - 1)])
    
    tpr_at_fpr_0001 = safe_tpr_at_fpr(0.0001)  # FPR = 1e-4
    tpr_at_fpr_001 = safe_tpr_at_fpr(0.001)    # FPR = 1e-3
    tpr_at_fpr_01 = safe_tpr_at_fpr(0.01)      # FPR = 1%
    tpr_at_fpr_05 = safe_tpr_at_fpr(0.05)      # FPR = 5%
    tpr_at_fpr_10 = safe_tpr_at_fpr(0.10)      # FPR = 10%
    
    # Precision-recall
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    
    if threshold is None:
        # Use Youden's J statistic for optimal threshold
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        threshold = thresholds_roc[optimal_idx] if optimal_idx < len(thresholds_roc) else 0.5
    
    # Compute metrics at threshold
    y_pred = (y_scores >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr_at_thresh = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        "auroc": float(auroc),
        "tpr_at_fpr_0001": float(tpr_at_fpr_0001),
        "tpr_at_fpr_001": float(tpr_at_fpr_001),
        "tpr_at_fpr_01": float(tpr_at_fpr_01),
        "tpr_at_fpr_05": float(tpr_at_fpr_05),
        "tpr_at_fpr_10": float(tpr_at_fpr_10),
        "precision": float(precision_val),
        "recall": float(recall_val),
        "fpr_at_threshold": float(fpr_at_thresh),
        "threshold": float(threshold)
    }


class RollingBuffer:
    """A fixed-size rolling buffer for storing recent items."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.buffer: List[Any] = []
    
    def add(self, item: Any) -> None:
        """Add item to buffer, removing oldest if full."""
        self.buffer.append(item)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def get_all(self) -> List[Any]:
        """Get all items in buffer."""
        return self.buffer.copy()
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = []
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def save(self, path: str) -> None:
        """Save buffer to file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable = []
        for item in self.buffer:
            if isinstance(item, np.ndarray):
                serializable.append(item.tolist())
            else:
                serializable.append(item)
        save_json({"buffer": serializable, "max_size": self.max_size}, path)
    
    @classmethod
    def load(cls, path: str) -> "RollingBuffer":
        """Load buffer from file."""
        data = load_json(path)
        buffer = cls(max_size=data.get("max_size", 100))
        buffer.buffer = [np.array(x) if isinstance(x, list) else x for x in data["buffer"]]
        return buffer
