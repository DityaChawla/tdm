"""
Policy engine for TDM.

Implements decision logic: ALLOW, BLOCK, REROUTE, PATCHED_RESPONSE.
"""

from typing import Dict, Optional, Union
from enum import Enum
from dataclasses import dataclass

from tdm.utils import logger


class Decision(Enum):
    """Possible decisions from the policy engine."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REROUTE = "REROUTE"
    PATCHED_RESPONSE = "PATCHED_RESPONSE"


@dataclass
class PolicyResult:
    """Result from policy evaluation."""
    decision: Decision
    risk: float
    signals: Dict[str, float]
    output: Optional[str] = None
    patched_output: Optional[str] = None
    rerouted_output: Optional[str] = None
    causal_validation: Optional[Dict] = None
    triggerspans: Optional[list] = None
    explanation: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "decision": self.decision.value,
            "risk": self.risk,
            "signals": self.signals,
            "output": self.output,
            "patched_output": self.patched_output,
            "rerouted_output": self.rerouted_output,
            "causal_validation": self.causal_validation,
            "triggerspans": self.triggerspans,
            "explanation": self.explanation
        }


class PolicyEngine:
    """
    Evaluates risk and determines response policy.
    
    Policy logic:
    - If risk < block_threshold: ALLOW
    - If risk >= block_threshold and causal patching succeeds: PATCHED_RESPONSE
    - If risk >= block_threshold and rerouting enabled: REROUTE
    - Otherwise: BLOCK
    """
    
    def __init__(
        self,
        block_threshold: float = 0.8,
        reroute_threshold: float = 0.6,
        enable_patching: bool = True,
        enable_rerouting: bool = True,
        reroute_model_name: Optional[str] = None,
        mode: str = "whitebox"
    ):
        """
        Initialize policy engine.
        
        Args:
            block_threshold: Risk threshold for blocking
            reroute_threshold: Risk threshold for rerouting
            enable_patching: Enable causal patching attempts
            enable_rerouting: Enable rerouting to safe model
            reroute_model_name: Model to reroute to
            mode: Operating mode ("whitebox" or "blackbox")
        """
        self.block_threshold = block_threshold
        self.reroute_threshold = reroute_threshold
        self.enable_patching = enable_patching
        self.enable_rerouting = enable_rerouting
        self.reroute_model_name = reroute_model_name
        self.mode = mode
        
        self._reroute_model = None
        self._reroute_tokenizer = None
    
    def _load_reroute_model(self):
        """Lazy load the reroute model."""
        if self._reroute_model is None and self.reroute_model_name:
            from tdm.instrumentation import load_model_and_tokenizer
            
            logger.info(f"Loading reroute model: {self.reroute_model_name}")
            self._reroute_model, self._reroute_tokenizer = load_model_and_tokenizer(
                self.reroute_model_name,
                device="cpu"  # Use CPU for safety/isolation
            )
    
    def _generate_rerouted(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate response using reroute model."""
        import torch
        
        self._load_reroute_model()
        
        if self._reroute_model is None:
            return ""
        
        inputs = self._reroute_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self._reroute_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._reroute_tokenizer.pad_token_id or self._reroute_tokenizer.eos_token_id
            )
        
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        return self._reroute_tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    def evaluate(
        self,
        prompt: str,
        risk: float,
        signals: Dict[str, float],
        output: Optional[str] = None,
        patcher=None,
        risk_fn=None,
        max_new_tokens: int = 50
    ) -> PolicyResult:
        """
        Evaluate policy and determine response.
        
        Args:
            prompt: Input prompt
            risk: Computed risk score
            signals: Individual signal scores
            output: Generated output (if any)
            patcher: ActivationPatcher for causal validation
            risk_fn: Risk function for patched output
            max_new_tokens: Max tokens for generation
        
        Returns:
            PolicyResult with decision and details
        """
        result = PolicyResult(
            decision=Decision.ALLOW,
            risk=risk,
            signals=signals,
            output=output,
            explanation=""
        )
        
        # Low risk: ALLOW
        if risk < self.reroute_threshold:
            result.decision = Decision.ALLOW
            result.explanation = f"Risk {risk:.3f} below reroute threshold {self.reroute_threshold}"
            return result
        
        # Medium risk: consider rerouting
        if risk < self.block_threshold:
            if self.enable_rerouting and self.reroute_model_name:
                result.decision = Decision.REROUTE
                result.rerouted_output = self._generate_rerouted(prompt, max_new_tokens)
                result.explanation = f"Risk {risk:.3f} in reroute range, using safe model"
                return result
            else:
                # No rerouting available, allow with warning
                result.decision = Decision.ALLOW
                result.explanation = f"Risk {risk:.3f} elevated but no rerouting configured"
                return result
        
        # High risk: try patching (whitebox only)
        if self.mode == "whitebox" and self.enable_patching and patcher is not None and risk_fn is not None:
            causal_result = patcher.causal_validation(
                prompt,
                risk_fn,
                max_new_tokens=max_new_tokens
            )
            result.causal_validation = causal_result
            
            if causal_result["causal_confirmed"]:
                # Patching reduced risk, use patched output
                result.decision = Decision.PATCHED_RESPONSE
                result.patched_output = causal_result["output_patched"]
                result.explanation = (
                    f"Risk {risk:.3f} high, but causal patching reduced risk "
                    f"from {causal_result['risk_before']:.3f} to {causal_result['risk_after']:.3f}"
                )
                return result
        
        # High risk with no successful patching: try rerouting
        if self.enable_rerouting and self.reroute_model_name:
            result.decision = Decision.REROUTE
            result.rerouted_output = self._generate_rerouted(prompt, max_new_tokens)
            result.explanation = f"Risk {risk:.3f} high, patching unavailable/failed, using safe model"
            return result
        
        # No options left: BLOCK
        result.decision = Decision.BLOCK
        result.explanation = f"Risk {risk:.3f} exceeds threshold {self.block_threshold}, blocking request"
        return result
    
    def get_final_output(self, result: PolicyResult) -> Optional[str]:
        """
        Get the appropriate output based on decision.
        
        Args:
            result: PolicyResult from evaluate()
        
        Returns:
            Output string or None if blocked
        """
        if result.decision == Decision.ALLOW:
            return result.output
        elif result.decision == Decision.PATCHED_RESPONSE:
            return result.patched_output
        elif result.decision == Decision.REROUTE:
            return result.rerouted_output
        else:  # BLOCK
            return None
    
    @classmethod
    def from_config(cls, config) -> "PolicyEngine":
        """Create policy engine from config."""
        return cls(
            block_threshold=config.block_threshold,
            reroute_threshold=config.reroute_threshold,
            enable_patching=config.enable_patching,
            enable_rerouting=config.reroute_model is not None,
            reroute_model_name=config.reroute_model,
            mode=config.mode
        )


def make_decision(
    risk: float,
    block_threshold: float = 0.8,
    reroute_threshold: float = 0.6
) -> str:
    """
    Simple decision function based on thresholds.
    
    Args:
        risk: Risk score
        block_threshold: Threshold for blocking
        reroute_threshold: Threshold for rerouting
    
    Returns:
        Decision string
    """
    if risk < reroute_threshold:
        return "ALLOW"
    elif risk < block_threshold:
        return "REROUTE"
    else:
        return "BLOCK"
