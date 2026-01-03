"""
Trigger span localization for TDM.

Identifies the input token spans most responsible for risk.

FIXED: Now includes gradient-based localization for last-token pooling,
which makes occlusion-based methods ineffective.
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from tdm.utils import logger


@dataclass
class TriggerSpan:
    """A localized trigger span with attribution."""
    start: int
    end: int
    text: str
    token_indices: List[int]
    tokens: List[str]
    importance: float
    delta: float  # Risk drop when masked


class TriggerSpanLocalizer:
    """
    Localizes trigger spans in input using multiple attribution methods.
    
    Supports:
    1. Occlusion-based: removes tokens and measures risk drop (works with mean pooling)
    2. Gradient-based: computes gradient of probe output w.r.t. embeddings (works with last-token pooling)
    3. Attention-based: uses attention weights to identify important tokens
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        risk_fn: Callable[[str], float],
        device: str = "cpu",
        min_risk_threshold: float = 0.5,
        use_removal: bool = True,
        min_importance_threshold: float = 0.01,
        probe=None,
        use_gradient: bool = True
    ):
        """
        Initialize trigger span localizer.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            risk_fn: Function (prompt) -> risk_score in [0, 1]
            device: Device for computation
            min_risk_threshold: Minimum baseline risk to run localization
            use_removal: If True, remove tokens; if False, mask them
            min_importance_threshold: Minimum delta to consider span meaningful
            probe: Optional LinearProbe for gradient-based localization
            use_gradient: If True, use gradient-based localization when occlusion fails
        """
        self.model = model
        self.tokenizer = tokenizer
        self.risk_fn = risk_fn
        self.device = device
        self.min_risk_threshold = min_risk_threshold
        self.use_removal = use_removal
        self.min_importance_threshold = min_importance_threshold
        self.probe = probe
        self.use_gradient = use_gradient
        
        if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None:
            self.mask_token = tokenizer.mask_token
        else:
            self.mask_token = None
    
    def compute_gradient_importance(
        self,
        prompt: str,
        baseline_risk: Optional[float] = None
    ) -> Tuple[np.ndarray, List[str], float]:
        """
        Compute token importance using gradient-based attribution.
        
        Uses integrated gradients / input x gradient for attribution.
        Works with last-token pooling by computing gradient of probe output
        w.r.t. input embeddings.
        
        Args:
            prompt: Input prompt
            baseline_risk: Pre-computed baseline risk
        
        Returns:
            Tuple of (importance array, token list, baseline_risk)
        """
        if self.model is None or self.probe is None:
            logger.warning("Gradient-based localization requires model and probe")
            return np.array([]), [], 0.0
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        input_ids = inputs["input_ids"].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        if len(tokens) == 0:
            return np.array([]), [], 0.0
        
        # Compute baseline risk if not provided
        if baseline_risk is None:
            baseline_risk = self.risk_fn(prompt)
        
        if baseline_risk < self.min_risk_threshold:
            return np.array([]), [], baseline_risk
        
        # Get embeddings with gradient tracking
        self.model.eval()
        
        try:
            # Get input embeddings
            if hasattr(self.model, 'transformer'):
                # GPT-2 style
                embeddings = self.model.transformer.wte(input_ids)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                # LLaMA style
                embeddings = self.model.model.embed_tokens(input_ids)
            else:
                # Generic
                embeddings = self.model.get_input_embeddings()(input_ids)
            
            embeddings = embeddings.detach().requires_grad_(True)
            
            # Forward pass with embeddings
            if hasattr(self.model, 'transformer'):
                # GPT-2
                position_ids = torch.arange(embeddings.shape[1], device=self.device).unsqueeze(0)
                position_embeds = self.model.transformer.wpe(position_ids)
                hidden = embeddings + position_embeds
                
                for block in self.model.transformer.h:
                    outputs = block(hidden)
                    hidden = outputs[0]
                
                hidden = self.model.transformer.ln_f(hidden)
            else:
                # Fallback: use model with inputs_embeds if supported
                outputs = self.model(inputs_embeds=embeddings, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
            
            # Get last token features (same as probe uses)
            features = hidden[:, -1, :]  # [1, hidden_dim]
            
            # Compute probe output
            logits = self.probe.linear(features)  # [1, 1]
            prob = torch.sigmoid(logits)
            
            # Backward pass
            prob.backward()
            
            # Get gradient w.r.t. embeddings
            grad = embeddings.grad  # [1, seq_len, hidden_dim]
            
            # Compute importance: input x gradient (L2 norm over hidden dim)
            importance = (embeddings * grad).norm(dim=-1).squeeze(0).detach().cpu().numpy()
            
            # Normalize to [0, 1]
            if importance.max() > 0:
                importance = importance / importance.max()
            
            return importance, tokens, baseline_risk
            
        except Exception as e:
            logger.warning(f"Gradient-based localization failed: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), [], baseline_risk
    
    def compute_attention_importance(
        self,
        prompt: str,
        baseline_risk: Optional[float] = None
    ) -> Tuple[np.ndarray, List[str], float]:
        """
        Compute token importance using attention weights.
        
        Uses the attention from the last token to all previous tokens
        as a proxy for importance.
        """
        if self.model is None:
            return np.array([]), [], 0.0
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        input_ids = inputs["input_ids"].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        if len(tokens) == 0:
            return np.array([]), [], 0.0
        
        if baseline_risk is None:
            baseline_risk = self.risk_fn(prompt)
        
        if baseline_risk < self.min_risk_threshold:
            return np.array([]), [], baseline_risk
        
        self.model.eval()
        
        try:
            with torch.no_grad():
                outputs = self.model(input_ids, output_attentions=True)
            
            # Get attention from last layer, last token
            # Shape: [batch, heads, seq, seq]
            attentions = outputs.attentions[-1]  # Last layer
            
            # Average over heads, get attention FROM last token TO all tokens
            last_token_attn = attentions[0, :, -1, :].mean(dim=0)  # [seq]
            importance = last_token_attn.cpu().numpy()
            
            # Normalize
            if importance.max() > 0:
                importance = importance / importance.max()
            
            return importance, tokens, baseline_risk
            
        except Exception as e:
            logger.warning(f"Attention-based localization failed: {e}")
            return np.array([]), [], baseline_risk
    
    def compute_token_importance(
        self,
        prompt: str,
        baseline_risk: Optional[float] = None
    ) -> Tuple[np.ndarray, List[str], float]:
        """
        Compute importance score for each token.
        
        Tries multiple methods:
        1. Gradient-based (if probe available)
        2. Attention-based (fallback)
        3. Occlusion-based (original method)
        
        Args:
            prompt: Input prompt
            baseline_risk: Pre-computed baseline risk
        
        Returns:
            Tuple of (importance array, token list, baseline_risk)
        """
        # Compute baseline risk
        if baseline_risk is None:
            baseline_risk = self.risk_fn(prompt)
        
        if baseline_risk < self.min_risk_threshold:
            logger.info(
                f"TRIGGERSPAN: Skipping - baseline risk {baseline_risk:.4f} "
                f"< threshold {self.min_risk_threshold}"
            )
            return np.array([]), [], baseline_risk
        
        # Try gradient-based first (best for last-token pooling)
        if self.use_gradient and self.probe is not None and hasattr(self.probe, 'linear'):
            importance, tokens, risk = self.compute_gradient_importance(prompt, baseline_risk)
            if len(importance) > 0 and importance.max() > 0.1:
                logger.info(f"TRIGGERSPAN: Using gradient-based attribution (max={importance.max():.4f})")
                return importance, tokens, risk
        
        # Try attention-based (works for any transformer)
        importance, tokens, risk = self.compute_attention_importance(prompt, baseline_risk)
        if len(importance) > 0 and importance.max() > 0.1:
            logger.info(f"TRIGGERSPAN: Using attention-based attribution (max={importance.max():.4f})")
            return importance, tokens, risk
        
        # Fall back to occlusion-based
        return self._compute_occlusion_importance(prompt, baseline_risk)
    
    def _compute_occlusion_importance(
        self,
        prompt: str,
        baseline_risk: float
    ) -> Tuple[np.ndarray, List[str], float]:
        """Original occlusion-based importance computation."""
        tokens = self.tokenizer.tokenize(prompt)
        
        if len(tokens) == 0:
            return np.array([]), [], baseline_risk
        
        importance = np.zeros(len(tokens))
        
        for i in range(len(tokens)):
            if self.use_removal or self.mask_token is None:
                removed_tokens = tokens[:i] + tokens[i+1:]
                if len(removed_tokens) == 0:
                    masked_prompt = ""
                else:
                    masked_prompt = self._reconstruct_text(removed_tokens)
            else:
                masked_tokens = tokens.copy()
                masked_tokens[i] = self.mask_token
                masked_prompt = self._reconstruct_text(masked_tokens)
            
            if not masked_prompt.strip():
                masked_risk = 0.0
            else:
                try:
                    masked_risk = self.risk_fn(masked_prompt)
                except Exception as e:
                    logger.warning(f"Error computing risk for masked prompt: {e}")
                    masked_risk = baseline_risk
            
            importance[i] = baseline_risk - masked_risk
        
        max_importance = np.max(np.abs(importance)) if len(importance) > 0 else 0.0
        if baseline_risk > 0.5 and max_importance < self.min_importance_threshold:
            logger.warning(
                f"TRIGGERSPAN WARNING: High baseline risk ({baseline_risk:.4f}) but max delta "
                f"is only {max_importance:.6f}. Occlusion method ineffective with last-token pooling."
            )
        
        return importance, tokens, baseline_risk
    
    def _reconstruct_text(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens."""
        if hasattr(self.tokenizer, 'convert_tokens_to_string'):
            return self.tokenizer.convert_tokens_to_string(tokens)
        else:
            text = ""
            for token in tokens:
                if token.startswith("Ġ"):
                    text += " " + token[1:]
                elif token.startswith("##"):
                    text += token[2:]
                else:
                    text += token
            return text.strip()
    
    def expand_span(
        self,
        seed_idx: int,
        tokens: List[str],
        importance: np.ndarray,
        max_length: int = 10
    ) -> Tuple[int, int, float]:
        """Expand a seed token into a contiguous span."""
        n_tokens = len(tokens)
        best_start = seed_idx
        best_end = seed_idx + 1
        best_importance = importance[seed_idx]
        
        start = seed_idx
        end = seed_idx + 1
        
        while end - start < max_length:
            left_importance = 0.0
            if start > 0:
                left_importance = importance[start - 1]
            
            right_importance = 0.0
            if end < n_tokens:
                right_importance = importance[end]
            
            if left_importance > right_importance and start > 0:
                start -= 1
                new_importance = importance[start:end].sum()
            elif right_importance > 0 and end < n_tokens:
                end += 1
                new_importance = importance[start:end].sum()
            else:
                break
            
            if new_importance > best_importance:
                best_start = start
                best_end = end
                best_importance = new_importance
        
        return best_start, best_end, best_importance
    
    def localize(
        self,
        prompt: str,
        top_k: int = 3,
        max_span_length: int = 10,
        baseline_risk: Optional[float] = None
    ) -> List[TriggerSpan]:
        """
        Localize top-K trigger spans in the input.
        
        Args:
            prompt: Input prompt
            top_k: Number of spans to return
            max_span_length: Maximum span length
            baseline_risk: Pre-computed baseline risk
        
        Returns:
            List of TriggerSpan objects
        """
        importance, tokens, baseline_risk = self.compute_token_importance(
            prompt, baseline_risk
        )
        
        if len(tokens) == 0 or len(importance) == 0:
            return []
        
        # Ensure importance matches tokens length
        if len(importance) != len(tokens):
            # Padding mismatch - use min length
            min_len = min(len(importance), len(tokens))
            importance = importance[:min_len]
            tokens = tokens[:min_len]
        
        seed_indices = np.argsort(importance)[::-1]
        
        spans: List[TriggerSpan] = []
        used_indices = set()
        
        for seed_idx in seed_indices:
            if seed_idx in used_indices:
                continue
            
            if len(spans) >= top_k:
                break
            
            # For gradient/attention methods, use normalized threshold
            threshold = 0.1 if self.use_gradient else self.min_importance_threshold
            if importance[seed_idx] < threshold:
                continue
            
            start, end, span_importance = self.expand_span(
                seed_idx, tokens, importance, max_span_length
            )
            
            span_indices = set(range(start, end))
            if span_indices & used_indices:
                continue
            
            used_indices.update(span_indices)
            
            # Get span text
            span_tokens = tokens[start:end]
            span_text = self._reconstruct_text(span_tokens)
            
            spans.append(TriggerSpan(
                start=start,
                end=end,
                text=span_text,
                token_indices=list(range(start, end)),
                tokens=span_tokens,
                importance=float(span_importance),
                delta=float(importance[start:end].sum())
            ))
        
        # Sort by importance
        spans.sort(key=lambda s: s.delta, reverse=True)
        
        return spans
    
    def localize_to_dict(
        self,
        prompt: str,
        top_k: int = 3,
        max_span_length: int = 10
    ) -> List[Dict]:
        """
        Localize spans and return as list of dicts (for JSON serialization).
        """
        spans = self.localize(prompt, top_k, max_span_length)
        
        return [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "tokens": s.tokens,
                "importance": s.importance,
                "delta": s.delta
            }
            for s in spans
        ]


def evaluate_localization_accuracy(
    localizer: TriggerSpanLocalizer,
    dataset,
    trigger_texts: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate how well localization identifies known triggers.
    
    Args:
        localizer: TriggerSpanLocalizer instance
        dataset: Dataset with prompts, labels, and trigger_texts
        trigger_texts: Optional list of known trigger strings
    
    Returns:
        Dict with evaluation metrics
    """
    if trigger_texts is None and hasattr(dataset, 'trigger_texts'):
        trigger_texts = dataset.trigger_texts
    
    results = {
        "total_triggered": 0,
        "trigger_found": 0,
        "trigger_in_top1": 0,
        "trigger_in_top3": 0,
        "mean_trigger_rank": [],
        "examples": []
    }
    
    for i, (prompt, label) in enumerate(zip(dataset.prompts, dataset.labels)):
        if label != 1:
            continue
        
        results["total_triggered"] += 1
        
        trigger_text = trigger_texts[i] if trigger_texts and i < len(trigger_texts) else None
        if not trigger_text:
            continue
        
        spans = localizer.localize(prompt, top_k=5)
        
        found = False
        rank = None
        
        for j, span in enumerate(spans):
            if trigger_text.lower() in span.text.lower():
                found = True
                rank = j + 1
                break
        
        if found:
            results["trigger_found"] += 1
            results["mean_trigger_rank"].append(rank)
            
            if rank == 1:
                results["trigger_in_top1"] += 1
            if rank <= 3:
                results["trigger_in_top3"] += 1
        
        if i < 5:
            results["examples"].append({
                "prompt": prompt[:80] + "...",
                "trigger": trigger_text,
                "spans": [{"text": s.text, "delta": s.delta} for s in spans[:3]],
                "found": found,
                "rank": rank
            })
    
    total = results["total_triggered"]
    if total > 0:
        results["accuracy_top1"] = results["trigger_in_top1"] / total
        results["accuracy_top3"] = results["trigger_in_top3"] / total
        results["recall"] = results["trigger_found"] / total
        results["mean_rank"] = np.mean(results["mean_trigger_rank"]) if results["mean_trigger_rank"] else None
    else:
        results["accuracy_top1"] = 0
        results["accuracy_top3"] = 0
        results["recall"] = 0
        results["mean_rank"] = None
    
    return results
