"""
Signal fusion for TDM.

Combines multiple detection signals into a unified risk score.
Fixed normalization and proper handling of canary scores.
"""

import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from tdm.utils import logger, sigmoid, save_json, load_json


class SignalFusion:
    """
    Fuses multiple detection signals into a single risk score.
    
    Supports rule-based fusion and optional learned stacking.
    Properly handles z-score normalization and ensures all signals are included.
    """
    
    def __init__(
        self,
        calibrator=None,
        weights: Optional[Dict[str, float]] = None,
        bias: float = 0.0,
        use_stacking: bool = False,
        stacking_model_path: Optional[str] = None,
        normalize_before_fusion: bool = True
    ):
        """
        Initialize signal fusion.
        
        Args:
            calibrator: Calibrator instance for z-score normalization
            weights: Dict of signal weights (default: equal weights)
            bias: Bias term for weighted combination
            use_stacking: Whether to use learned stacking model
            stacking_model_path: Path to saved stacking model
            normalize_before_fusion: If True, z-score normalize before fusion
        """
        self.calibrator = calibrator
        
        # Default weights - ensure canary is always included
        self.weights = weights or {
            "probe": 0.5,
            "drift": 0.25,
            "canary": 0.25
        }
        self.bias = bias
        self.normalize_before_fusion = normalize_before_fusion
        
        self.use_stacking = use_stacking
        self.stacking_model: Optional[nn.Module] = None
        self._stacking_feature_names: List[str] = []
        
        if use_stacking and stacking_model_path and os.path.exists(stacking_model_path):
            self.load_stacking_model(stacking_model_path)
    
    def normalize_scores(self, signals: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize raw scores to z-scores using calibration statistics.
        
        Args:
            signals: Dict of raw signal scores
        
        Returns:
            Dict of z-scored signals
        """
        normalized = {}
        
        for name, value in signals.items():
            if self.calibrator is not None and hasattr(self.calibrator, 'normalize_score'):
                z = self.calibrator.normalize_score(name, value)
            else:
                z = value  # No normalization
            normalized[name] = z
        
        return normalized
    
    def rule_based_flag(
        self,
        signals: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Apply rule-based flagging logic.
        
        Rule: flag = (probe > tau AND (drift > tau OR canary > tau)) 
                     OR (drift > tau AND canary > tau)
        
        Args:
            signals: Raw signal scores
            thresholds: Signal thresholds (uses calibrator if None)
        
        Returns:
            True if flagged
        """
        if thresholds is None and self.calibrator is not None:
            thresholds = self.calibrator.thresholds
        
        if thresholds is None:
            return False
        
        probe_flag = signals.get("probe", 0) > thresholds.get("probe", float('inf'))
        drift_flag = signals.get("drift", 0) > thresholds.get("drift", float('inf'))
        canary_flag = signals.get("canary", 0) > thresholds.get("canary", float('inf'))
        
        flag = (probe_flag and (drift_flag or canary_flag)) or (drift_flag and canary_flag)
        
        return flag
    
    def weighted_risk(
        self,
        signals: Dict[str, float],
        use_raw_scores: bool = True
    ) -> float:
        """
        Compute weighted risk score.
        
        FIXED: Two modes of operation:
        1. use_raw_scores=True: Use raw probability scores directly (assumes signals are in [0,1])
        2. use_raw_scores=False: Z-score normalize, then apply sigmoid to combined score
        
        Args:
            signals: Raw signal scores (should be in [0, 1] for probe scores)
            use_raw_scores: If True, combine raw scores; if False, z-score first
        
        Returns:
            Risk score in [0, 1]
        """
        if use_raw_scores:
            # Mode 1: Raw score fusion (preferred for calibrated probability scores)
            # Just weighted average of raw scores
            weighted_sum = 0.0
            total_weight = 0.0
            
            for name, value in signals.items():
                weight = self.weights.get(name, 0)
                if weight > 0:
                    # Ensure value is in reasonable range
                    value = max(0.0, min(1.0, value))
                    weighted_sum += weight * value
                    total_weight += weight
            
            if total_weight == 0:
                return 0.0
            
            # Normalize by total weight to get weighted average
            risk = weighted_sum / total_weight
            return float(risk)
        
        else:
            # Mode 2: Z-score fusion with sigmoid
            z_scores = self.normalize_scores(signals)
            
            weighted_sum = self.bias
            for name, z in z_scores.items():
                weight = self.weights.get(name, 0)
                weighted_sum += weight * z
            
            # Apply sigmoid to get probability
            risk = sigmoid(weighted_sum)
            return float(risk)
    
    def fuse(
        self,
        signals: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None,
        use_raw_scores: bool = True
    ) -> Dict[str, Union[float, bool]]:
        """
        Fuse signals into final risk assessment.
        
        Args:
            signals: Dict of signal values
            thresholds: Signal thresholds
            use_raw_scores: If True, combine raw scores; if False, z-score first
        
        Returns:
            Dict with risk score, flag, z_scores
        """
        # Ensure all expected signals are present (with defaults)
        complete_signals = {
            "probe": signals.get("probe", 0.0),
            "drift": signals.get("drift", 0.0),
            "canary": signals.get("canary", 0.0)
        }
        # Include any additional signals
        complete_signals.update(signals)
        
        # Normalize for z-scores (for diagnostics)
        z_scores = self.normalize_scores(complete_signals)
        
        # Use stacking if available
        if self.use_stacking and self.stacking_model is not None:
            risk = self._stacking_predict(z_scores)
        else:
            risk = self.weighted_risk(complete_signals, use_raw_scores=use_raw_scores)
        
        # Rule-based flag
        flag = self.rule_based_flag(complete_signals, thresholds)
        
        return {
            "risk": float(risk),
            "flag": flag,
            "z_scores": z_scores,
            "raw_signals": complete_signals
        }
    
    def _stacking_predict(self, z_scores: Dict[str, float]) -> float:
        """Predict using stacking model."""
        # Create feature vector in consistent order
        feature_names = self._stacking_feature_names or sorted(z_scores.keys())
        features = torch.tensor([z_scores.get(n, 0) for n in feature_names], dtype=torch.float32)
        features = features.unsqueeze(0)  # [1, n_features]
        
        with torch.no_grad():
            logit = self.stacking_model(features)
            risk = torch.sigmoid(logit).item()
        
        return risk
    
    def compute_fused_threshold(
        self,
        benign_signals_list: List[Dict[str, float]],
        alpha: float = 0.001,
        use_raw_scores: bool = True
    ) -> float:
        """
        Compute fused risk threshold via conformal calibration.
        
        Args:
            benign_signals_list: List of signal dicts from benign samples
            alpha: Target FPR
            use_raw_scores: Scoring mode
        
        Returns:
            Fused threshold at 1-alpha quantile
        """
        fused_scores = []
        for signals in benign_signals_list:
            risk = self.weighted_risk(signals, use_raw_scores=use_raw_scores)
            fused_scores.append(risk)
        
        fused_scores = np.array(fused_scores)
        threshold = np.quantile(fused_scores, 1 - alpha)
        
        logger.info(f"Fused threshold at alpha={alpha}: {threshold:.6f}")
        return float(threshold)
    
    def train_stacking(
        self,
        signals_list: List[Dict[str, float]],
        labels: np.ndarray,
        device: str = "cpu",
        epochs: int = 100,
        learning_rate: float = 1e-2
    ) -> Dict:
        """
        Train a stacking model on signal features.
        
        Args:
            signals_list: List of signal dicts
            labels: Binary labels
            device: Training device
            epochs: Training epochs
            learning_rate: Learning rate
        
        Returns:
            Training statistics
        """
        # Extract features
        feature_names = sorted(signals_list[0].keys())
        n_features = len(feature_names)
        
        X = []
        for signals in signals_list:
            z_scores = self.normalize_scores(signals)
            features = [z_scores.get(n, 0) for n in feature_names]
            X.append(features)
        
        X = torch.tensor(X, dtype=torch.float32, device=device)
        y = torch.tensor(labels, dtype=torch.float32, device=device)
        
        # Simple linear model
        self.stacking_model = nn.Linear(n_features, 1).to(device)
        
        optimizer = torch.optim.Adam(self.stacking_model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self.stacking_model(X).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        with torch.no_grad():
            preds = (torch.sigmoid(self.stacking_model(X).squeeze(-1)) > 0.5).float()
            accuracy = (preds == y).float().mean().item()
        
        self.use_stacking = True
        self._stacking_feature_names = feature_names
        
        logger.info(f"Stacking model trained: accuracy={accuracy:.4f}")
        
        return {
            "accuracy": accuracy,
            "n_features": n_features,
            "feature_names": feature_names
        }
    
    def save_stacking_model(self, path: str):
        """Save stacking model."""
        if self.stacking_model is None:
            raise ValueError("No stacking model to save")
        
        torch.save({
            "state_dict": self.stacking_model.state_dict(),
            "feature_names": self._stacking_feature_names,
            "n_features": len(self._stacking_feature_names)
        }, path)
        logger.info(f"Stacking model saved to {path}")
    
    def load_stacking_model(self, path: str, device: str = "cpu"):
        """Load stacking model."""
        checkpoint = torch.load(path, map_location=device)
        
        n_features = checkpoint["n_features"]
        self.stacking_model = nn.Linear(n_features, 1).to(device)
        self.stacking_model.load_state_dict(checkpoint["state_dict"])
        self.stacking_model.eval()
        
        self._stacking_feature_names = checkpoint["feature_names"]
        self.use_stacking = True
        
        logger.info(f"Stacking model loaded from {path}")
    
    @classmethod
    def from_config(cls, config) -> "SignalFusion":
        """Create fusion from config."""
        from tdm.calibration import Calibrator
        
        calibrator = None
        thresholds_path = getattr(config, 'thresholds_path', None)
        if thresholds_path and os.path.exists(thresholds_path):
            calibrator = Calibrator.load(thresholds_path)
        
        stacking_path = getattr(config, 'stacking_model_path', None)
        use_stacking = getattr(config, 'use_stacking', False)
        
        return cls(
            calibrator=calibrator,
            weights=getattr(config, 'fusion_weights', None),
            bias=getattr(config, 'fusion_bias', 0.0),
            use_stacking=use_stacking,
            stacking_model_path=stacking_path if use_stacking else None
        )


def combine_scores_simple(
    probe_score: float,
    drift_score: float,
    canary_score: float,
    weights: tuple = (0.5, 0.25, 0.25)
) -> float:
    """
    Simple weighted combination of scores.
    
    Convenience function for quick fusion without full setup.
    Uses weighted average (not sigmoid) for calibrated scores.
    """
    w_probe, w_drift, w_canary = weights
    total_weight = w_probe + w_drift + w_canary
    
    combined = (w_probe * probe_score + w_drift * drift_score + w_canary * canary_score)
    return combined / total_weight if total_weight > 0 else 0.0


def check_fusion_quality(
    probe_auroc: float,
    fused_auroc: float,
    threshold: float = 0.01
) -> Dict[str, bool]:
    """
    Check if fusion is improving over probe-only baseline.
    
    Args:
        probe_auroc: AUROC using only probe scores
        fused_auroc: AUROC using fused scores
        threshold: Minimum improvement needed
    
    Returns:
        Dict with quality checks
    """
    improvement = fused_auroc - probe_auroc
    
    result = {
        "fusion_helps": improvement > threshold,
        "fusion_neutral": abs(improvement) <= threshold,
        "fusion_hurts": improvement < -threshold,
        "improvement": improvement
    }
    
    if result["fusion_hurts"]:
        logger.warning(
            f"FUSION WARNING: Fused AUROC ({fused_auroc:.4f}) is worse than "
            f"probe-only ({probe_auroc:.4f}). Fusion may be misconfigured or "
            f"drift/canary signals are noisy."
        )
    
    return result
