"""
Triangulated Defection Monitor (TDM)

A monitoring sidecar for detecting sleeper-agent/backdoor behavior in LLMs.
Provides calibrated risk scores, decisions, and forensic evidence.
"""

__version__ = "0.1.0"
__author__ = "TDM Research Team"

from tdm.config import TDMConfig
from tdm.instrumentation import load_model_and_tokenizer, get_residual_activations
from tdm.probe import LinearProbe, train_probe, load_probe
from tdm.drift import DriftDetector
from tdm.canary import CanaryChecker
from tdm.calibration import Calibrator, load_calibration
from tdm.fusion import SignalFusion
from tdm.patching import ActivationPatcher
from tdm.triggerspan import TriggerSpanLocalizer
from tdm.policy import PolicyEngine

__all__ = [
    "TDMConfig",
    "load_model_and_tokenizer",
    "get_residual_activations",
    "LinearProbe",
    "train_probe",
    "load_probe",
    "DriftDetector",
    "CanaryChecker",
    "Calibrator",
    "load_calibration",
    "SignalFusion",
    "ActivationPatcher",
    "TriggerSpanLocalizer",
    "PolicyEngine",
]
