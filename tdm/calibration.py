"""
Calibration for TDM detection thresholds.

Provides conformal quantile thresholding for false positive control.
Includes strict enforcement for research-grade low-FPR calibration.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
from scipy import stats

from tdm.utils import logger, save_json, load_json


# ============================================================================
# Constants for Research-Grade Calibration
# ============================================================================

MIN_SAMPLES_MULTIPLIER = 10  # Require n >= 10/alpha for reliable calibration
STRICT_ALPHA_THRESHOLD = 0.01  # Alpha below this requires strict sample checks


class CalibrationError(Exception):
    """Raised when calibration requirements are not met."""
    pass


class InsufficientSamplesError(CalibrationError):
    """Raised when sample size is too small for the requested alpha."""
    pass


# ============================================================================
# Main Calibrator Class
# ============================================================================

class Calibrator:
    """
    Calibrates detection thresholds using conformal quantile methods.
    
    Given a set of benign (known-safe) samples, computes score thresholds
    that control the false positive rate at a specified alpha level.
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        method: str = "quantile"
    ):
        """
        Initialize calibrator.
        
        Args:
            alpha: Desired false positive rate (e.g., 0.05 = 5% FPR)
            method: Calibration method ("quantile" or "conformal")
        """
        self.alpha = alpha
        self.method = method
        
        self.thresholds: Dict[str, float] = {}
        self.stats: Dict[str, Dict[str, float]] = {}
        self.n_calibration_samples: int = 0
        self.timestamp: Optional[str] = None
        self.dataset_hash: Optional[str] = None
    
    def get_minimum_samples(self) -> int:
        """Get minimum required samples for this alpha."""
        return int(np.ceil(MIN_SAMPLES_MULTIPLIER / self.alpha))
    
    def check_sample_sufficiency(
        self,
        n_samples: int,
        score_name: str = "",
        strict: bool = False
    ) -> bool:
        """
        Check if sample size is sufficient for reliable calibration.
        
        Args:
            n_samples: Number of calibration samples
            score_name: Name of score being calibrated
            strict: If True, raise error instead of warning
        
        Returns:
            True if sufficient
        
        Raises:
            InsufficientSamplesError if strict=True and insufficient
        """
        min_required = self.get_minimum_samples()
        
        if n_samples < min_required:
            msg = (
                f"CALIBRATION {'ERROR' if strict else 'WARNING'}: "
                f"'{score_name}' has {n_samples} samples but alpha={self.alpha} "
                f"requires >= {min_required} samples (10/alpha) for reliable low-FPR thresholds. "
            )
            
            if self.alpha <= 0.001:
                msg += (
                    f"For alpha=1e-3 or lower, use at least 10,000 samples. "
                    f"Results with current sample size will be UNSTABLE and NOT research-grade."
                )
            
            if strict:
                raise InsufficientSamplesError(msg)
            else:
                logger.warning(msg)
                return False
        
        return True
    
    def calibrate(
        self,
        scores: Dict[str, np.ndarray],
        score_names: Optional[List[str]] = None,
        strict: bool = False
    ) -> Dict[str, float]:
        """
        Calibrate thresholds on benign calibration scores.
        
        Args:
            scores: Dict mapping score names to arrays of benign scores
            score_names: Optional subset of score names to calibrate
            strict: If True, error (not warn) on insufficient samples
        
        Returns:
            Dict of calibrated thresholds
        
        Raises:
            InsufficientSamplesError if strict=True and samples insufficient
        """
        if score_names is None:
            score_names = list(scores.keys())
        
        # Global sample check
        min_required = self.get_minimum_samples()
        
        for name in score_names:
            if name not in scores:
                logger.warning(f"Score '{name}' not found in calibration data")
                continue
            
            score_values = np.asarray(scores[name])
            n_samples = len(score_values)
            
            if n_samples == 0:
                logger.warning(f"Empty calibration data for '{name}'")
                continue
            
            # Check sample sufficiency
            self.check_sample_sufficiency(n_samples, name, strict=strict)
            
            # Compute threshold
            if self.method == "quantile":
                # Simple quantile threshold
                threshold = np.quantile(score_values, 1 - self.alpha)
            
            elif self.method == "conformal":
                # Conformal p-value with finite-sample correction
                n = len(score_values)
                # Adjust quantile for finite sample: guarantees FPR <= alpha
                adjusted_quantile = (np.ceil((1 - self.alpha) * (n + 1))) / n
                adjusted_quantile = min(adjusted_quantile, 1.0)
                threshold = np.quantile(score_values, adjusted_quantile)
            
            else:
                raise ValueError(f"Unknown calibration method: {self.method}")
            
            self.thresholds[name] = float(threshold)
            
            # Store comprehensive statistics for normalization and analysis
            self.stats[name] = {
                "mean": float(np.mean(score_values)),
                "std": float(np.std(score_values) + 1e-8),
                "min": float(np.min(score_values)),
                "max": float(np.max(score_values)),
                "median": float(np.median(score_values)),
                "q25": float(np.quantile(score_values, 0.25)),
                "q75": float(np.quantile(score_values, 0.75)),
                "q90": float(np.quantile(score_values, 0.90)),
                "q95": float(np.quantile(score_values, 0.95)),
                "q99": float(np.quantile(score_values, 0.99)),
                "q999": float(np.quantile(score_values, 0.999)) if n_samples >= 1000 else float(np.max(score_values)),
                "q9999": float(np.quantile(score_values, 0.9999)) if n_samples >= 10000 else float(np.max(score_values)),
                "n_samples": n_samples,
                "min_required": min_required,
                "sufficient": n_samples >= min_required
            }
        
        self.n_calibration_samples = min(len(v) for v in scores.values()) if scores else 0
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Compute dataset hash for reproducibility
        import hashlib
        hash_input = str(sorted([(k, len(v), float(np.mean(v))) for k, v in scores.items()]))
        self.dataset_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        
        logger.info(f"Calibrated {len(self.thresholds)} thresholds at alpha={self.alpha}")
        for name, thresh in self.thresholds.items():
            sufficient = self.stats[name]["sufficient"] if name in self.stats else False
            status = "✓" if sufficient else "⚠"
            logger.info(f"  {status} {name}: threshold={thresh:.6f} (n={self.stats[name]['n_samples']})")
        
        return self.thresholds.copy()
    
    def validate_on_holdout(
        self,
        holdout_scores: Dict[str, np.ndarray],
        holdout_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Dict]:
        """
        Validate calibration on held-out data.
        
        Args:
            holdout_scores: Dict of score arrays on held-out benign data
            holdout_labels: Optional labels (if provided, 0=benign for FPR calculation)
        
        Returns:
            Dict with observed FPR for each score type
        """
        results = {}
        
        for name, scores in holdout_scores.items():
            if name not in self.thresholds:
                continue
            
            threshold = self.thresholds[name]
            scores = np.asarray(scores)
            
            # If labels provided, only use benign (label=0) for FPR
            if holdout_labels is not None:
                benign_mask = np.asarray(holdout_labels) == 0
                benign_scores = scores[benign_mask]
            else:
                benign_scores = scores
            
            # Compute observed FPR (false positives on benign data)
            n_flagged = np.sum(benign_scores > threshold)
            observed_fpr = n_flagged / len(benign_scores) if len(benign_scores) > 0 else 0.0
            
            results[name] = {
                "threshold": float(threshold),
                "n_samples": len(benign_scores),
                "n_flagged": int(n_flagged),
                "observed_fpr": float(observed_fpr),
                "target_alpha": self.alpha,
                "fpr_ratio": observed_fpr / self.alpha if self.alpha > 0 else 0,
                "fpr_within_tolerance": abs(observed_fpr - self.alpha) < 2 * self.alpha
            }
            
            if observed_fpr > 2 * self.alpha:
                logger.warning(
                    f"CALIBRATION VALIDATION: '{name}' observed FPR={observed_fpr:.6f} "
                    f"exceeds 2x target alpha={self.alpha}"
                )
            elif observed_fpr < 0.5 * self.alpha:
                logger.info(
                    f"CALIBRATION NOTE: '{name}' observed FPR={observed_fpr:.6f} is conservative "
                    f"(below 0.5x target alpha={self.alpha})"
                )
        
        return results
    
    def get_threshold(self, score_name: str) -> Optional[float]:
        """Get calibrated threshold for a score type."""
        return self.thresholds.get(score_name)
    
    def get_threshold_at_alpha(self, score_name: str, target_alpha: float) -> Optional[float]:
        """
        Get threshold for a different alpha using stored statistics.
        
        Useful for computing thresholds at multiple operating points.
        
        Args:
            score_name: Score type
            target_alpha: Target FPR
        
        Returns:
            Threshold at target_alpha (approximate)
        """
        if score_name not in self.stats:
            return None
        
        stat = self.stats[score_name]
        
        # Use stored quantiles if target matches
        quantile = 1 - target_alpha
        if quantile == 0.99 and "q99" in stat:
            return stat["q99"]
        if quantile == 0.999 and "q999" in stat:
            return stat["q999"]
        if quantile == 0.9999 and "q9999" in stat:
            return stat["q9999"]
        
        # Otherwise use normal approximation
        z = stats.norm.ppf(quantile)
        return stat["mean"] + z * stat["std"]
    
    def compute_pvalue(
        self,
        score_name: str,
        score_value: float
    ) -> Optional[float]:
        """
        Compute p-value for a score.
        
        p-value = P(seeing this score or higher on benign data)
        
        Args:
            score_name: Score type
            score_value: Observed score
        
        Returns:
            p-value or None if not calibrated
        """
        if score_name not in self.stats:
            return None
        
        stat = self.stats[score_name]
        
        # Use normal approximation
        z = (score_value - stat["mean"]) / stat["std"]
        p_value = 1 - stats.norm.cdf(z)
        
        return float(p_value)
    
    def normalize_score(
        self,
        score_name: str,
        score_value: float
    ) -> float:
        """
        Normalize score to z-score using calibration statistics.
        
        Args:
            score_name: Score type
            score_value: Raw score
        
        Returns:
            Z-score
        """
        if score_name not in self.stats:
            return score_value
        
        stat = self.stats[score_name]
        return (score_value - stat["mean"]) / stat["std"]
    
    def is_flagged(
        self,
        score_name: str,
        score_value: float
    ) -> bool:
        """Check if a score exceeds its threshold."""
        threshold = self.thresholds.get(score_name)
        if threshold is None:
            return False
        return score_value > threshold
    
    def save(self, path: str):
        """Save calibration data to JSON file."""
        data = {
            "alpha": self.alpha,
            "method": self.method,
            "thresholds": self.thresholds,
            "stats": self.stats,
            "n_calibration_samples": self.n_calibration_samples,
            "timestamp": self.timestamp,
            "dataset_hash": self.dataset_hash,
            "min_samples_required": self.get_minimum_samples(),
            "version": "2.0"  # Research-grade version
        }
        save_json(data, path)
        logger.info(f"Calibration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "Calibrator":
        """Load calibration from file."""
        data = load_json(path)
        
        calibrator = cls(
            alpha=data["alpha"],
            method=data.get("method", "quantile")
        )
        calibrator.thresholds = data["thresholds"]
        calibrator.stats = data.get("stats", {})
        calibrator.n_calibration_samples = data.get("n_calibration_samples", 0)
        calibrator.timestamp = data.get("timestamp")
        calibrator.dataset_hash = data.get("dataset_hash")
        
        logger.info(f"Calibration loaded from {path}")
        return calibrator


def load_calibration(path: str) -> Calibrator:
    """Load calibration from file (convenience function)."""
    return Calibrator.load(path)


# ============================================================================
# Multi-Alpha Calibrator for Research Evaluation
# ============================================================================

class MultiAlphaCalibrator:
    """
    Calibrator that computes thresholds at multiple alpha levels.
    
    Useful for generating ROC-style curves at specific operating points.
    """
    
    def __init__(
        self,
        alphas: Optional[List[float]] = None,
        method: str = "quantile"
    ):
        """
        Initialize multi-alpha calibrator.
        
        Args:
            alphas: List of alpha levels (default: research-relevant levels)
            method: Calibration method
        """
        self.alphas = alphas or [0.1, 0.05, 0.01, 0.001, 0.0001]
        self.method = method
        self.calibrators: Dict[float, Calibrator] = {}
    
    def calibrate(
        self,
        scores: Dict[str, np.ndarray],
        strict: bool = False
    ) -> Dict[float, Dict[str, float]]:
        """
        Calibrate at all alpha levels.
        
        Args:
            scores: Dict of score arrays
            strict: If True, error on insufficient samples for any alpha
        
        Returns:
            Dict mapping alpha -> thresholds
        """
        results = {}
        
        for alpha in self.alphas:
            cal = Calibrator(alpha=alpha, method=self.method)
            try:
                thresholds = cal.calibrate(scores, strict=strict)
                self.calibrators[alpha] = cal
                results[alpha] = thresholds
            except InsufficientSamplesError as e:
                logger.warning(f"Skipping alpha={alpha}: {e}")
        
        return results
    
    def get_thresholds_for_score(self, score_name: str) -> Dict[float, float]:
        """Get thresholds at all alpha levels for a given score."""
        return {
            alpha: cal.thresholds.get(score_name, float('inf'))
            for alpha, cal in self.calibrators.items()
        }
    
    def save(self, path: str):
        """Save all calibrations."""
        data = {
            "alphas": self.alphas,
            "method": self.method,
            "calibrations": {
                str(alpha): {
                    "thresholds": cal.thresholds,
                    "stats": cal.stats
                }
                for alpha, cal in self.calibrators.items()
            }
        }
        save_json(data, path)


# ============================================================================
# Adaptive Calibrator (Online Recalibration)
# ============================================================================

class AdaptiveCalibrator(Calibrator):
    """
    Calibrator that supports online recalibration.
    
    Maintains a sliding window of recent scores for adaptive thresholds.
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        method: str = "quantile",
        window_size: int = 1000,
        min_samples: int = 100
    ):
        """
        Initialize adaptive calibrator.
        
        Args:
            alpha: Desired false positive rate
            method: Calibration method
            window_size: Maximum samples to store per score type
            min_samples: Minimum samples before recalibration
        """
        super().__init__(alpha=alpha, method=method)
        
        self.window_size = window_size
        self.min_samples = min_samples
        self.score_buffers: Dict[str, List[float]] = {}
    
    def add_score(
        self,
        score_name: str,
        score_value: float,
        is_benign: bool = True
    ):
        """
        Add a score to the buffer.
        
        Args:
            score_name: Score type
            score_value: Score value
            is_benign: Whether this is known to be benign
        """
        if not is_benign:
            return  # Only track benign scores
        
        if score_name not in self.score_buffers:
            self.score_buffers[score_name] = []
        
        self.score_buffers[score_name].append(score_value)
        
        # Trim to window size
        if len(self.score_buffers[score_name]) > self.window_size:
            self.score_buffers[score_name] = self.score_buffers[score_name][-self.window_size:]
    
    def recalibrate(self) -> bool:
        """
        Recalibrate thresholds using buffered scores.
        
        Returns:
            True if recalibration was performed
        """
        # Check if we have enough samples
        if not self.score_buffers:
            return False
        
        min_buffer_size = min(len(v) for v in self.score_buffers.values())
        if min_buffer_size < self.min_samples:
            return False
        
        # Build score dict from buffers
        scores = {
            name: np.array(values)
            for name, values in self.score_buffers.items()
        }
        
        self.calibrate(scores, strict=False)
        return True
    
    def save(self, path: str):
        """Save calibration with buffers."""
        data = {
            "alpha": self.alpha,
            "method": self.method,
            "thresholds": self.thresholds,
            "stats": self.stats,
            "n_calibration_samples": self.n_calibration_samples,
            "timestamp": self.timestamp,
            "window_size": self.window_size,
            "min_samples": self.min_samples,
            "score_buffers": self.score_buffers,
            "version": "2.0"
        }
        save_json(data, path)
    
    @classmethod
    def load(cls, path: str) -> "AdaptiveCalibrator":
        """Load adaptive calibrator from file."""
        data = load_json(path)
        
        calibrator = cls(
            alpha=data["alpha"],
            method=data.get("method", "quantile"),
            window_size=data.get("window_size", 1000),
            min_samples=data.get("min_samples", 100)
        )
        calibrator.thresholds = data["thresholds"]
        calibrator.stats = data.get("stats", {})
        calibrator.n_calibration_samples = data.get("n_calibration_samples", 0)
        calibrator.timestamp = data.get("timestamp")
        calibrator.score_buffers = data.get("score_buffers", {})
        
        return calibrator
