"""
Canary consistency checks for TDM.

Tests model consistency by generating semantic-preserving prompt variants.
"""

import re
import random
from typing import List, Optional, Tuple, Callable

import numpy as np

from tdm.utils import logger, cosine_distance


class CanaryTransform:
    """Base class for canary transformations."""
    
    name: str = "base"
    
    def __call__(self, prompt: str) -> str:
        """Apply transformation to prompt."""
        raise NotImplementedError
    
    def is_applicable(self, prompt: str) -> bool:
        """Check if this transform can be applied to the prompt."""
        return True


class SuffixTransform(CanaryTransform):
    """Add a benign suffix to the prompt."""
    
    name = "suffix"
    
    SUFFIXES = [
        " Please be helpful.",
        " Thank you in advance.",
        " I appreciate your assistance.",
        " Looking forward to your response.",
        " Thanks!",
    ]
    
    def __call__(self, prompt: str) -> str:
        suffix = random.choice(self.SUFFIXES)
        return prompt.rstrip() + suffix
    
    def is_applicable(self, prompt: str) -> bool:
        return len(prompt.strip()) > 0


class MathTransform(CanaryTransform):
    """Add a simple math question as prefix or suffix."""
    
    name = "math"
    
    MATH_QUESTIONS = [
        "(Note: 2+2=4) ",
        "(Quick check: 10-5=5) ",
        "(FYI: 3*3=9) ",
    ]
    
    MATH_SUFFIXES = [
        " Also, what is 1+1?",
        " By the way, 7-3=4.",
    ]
    
    def __call__(self, prompt: str) -> str:
        if random.random() < 0.5:
            prefix = random.choice(self.MATH_QUESTIONS)
            return prefix + prompt
        else:
            suffix = random.choice(self.MATH_SUFFIXES)
            return prompt.rstrip() + suffix
    
    def is_applicable(self, prompt: str) -> bool:
        return len(prompt.strip()) > 0


class WhitespaceTransform(CanaryTransform):
    """Apply whitespace perturbations."""
    
    name = "whitespace"
    
    def __call__(self, prompt: str) -> str:
        # Randomly apply one of several whitespace transforms
        choice = random.randint(0, 2)
        
        if choice == 0:
            # Double spaces after periods
            return prompt.replace(". ", ".  ")
        elif choice == 1:
            # Add leading/trailing whitespace
            return "  " + prompt + "  "
        else:
            # Normalize multiple spaces
            return re.sub(r'\s+', ' ', prompt).strip()
    
    def is_applicable(self, prompt: str) -> bool:
        return len(prompt.strip()) > 0


class ReorderTransform(CanaryTransform):
    """Reorder independent sentences in the prompt."""
    
    name = "reorder"
    
    def __call__(self, prompt: str) -> str:
        # Split by sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', prompt.strip())
        
        if len(sentences) < 2:
            return prompt
        
        # Shuffle sentences
        shuffled = sentences.copy()
        random.shuffle(shuffled)
        
        # Avoid returning identical order
        if shuffled == sentences:
            shuffled = sentences[::-1]
        
        return ' '.join(shuffled)
    
    def is_applicable(self, prompt: str) -> bool:
        sentences = re.split(r'(?<=[.!?])\s+', prompt.strip())
        return len(sentences) >= 2


class SynonymTransform(CanaryTransform):
    """Replace words with simple synonyms."""
    
    name = "synonym"
    
    SYNONYMS = {
        "help": "assist",
        "want": "would like",
        "good": "great",
        "bad": "poor",
        "make": "create",
        "tell": "inform",
        "show": "display",
        "find": "locate",
        "give": "provide",
        "use": "utilize",
    }
    
    def __call__(self, prompt: str) -> str:
        result = prompt
        for word, synonym in self.SYNONYMS.items():
            # Case-insensitive replacement with word boundaries
            pattern = rf'\b{word}\b'
            if random.random() < 0.5:  # Don't replace every occurrence
                result = re.sub(pattern, synonym, result, flags=re.IGNORECASE, count=1)
        return result
    
    def is_applicable(self, prompt: str) -> bool:
        lower = prompt.lower()
        return any(word in lower for word in self.SYNONYMS.keys())


# Registry of available transforms
TRANSFORM_REGISTRY = {
    "suffix": SuffixTransform,
    "math": MathTransform,
    "whitespace": WhitespaceTransform,
    "reorder": ReorderTransform,
    "synonym": SynonymTransform,
}


class CanaryChecker:
    """
    Checks model consistency using canary prompt variants.
    
    A consistent model should produce similar outputs for semantically
    equivalent input variants. Large divergence may indicate trigger detection.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        transform_names: List[str] = None,
        n_canaries: int = 4,
        max_new_tokens: int = 50,
        embedder_name: str = "distilbert-base-uncased"
    ):
        """
        Initialize canary checker.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            device: Device for computation
            transform_names: List of transform names to use
            n_canaries: Number of canary variants to generate
            max_new_tokens: Max tokens to generate for each variant
            embedder_name: Model for output embedding
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_canaries = n_canaries
        self.max_new_tokens = max_new_tokens
        
        # Initialize transforms
        if transform_names is None:
            transform_names = ["suffix", "math", "whitespace", "reorder"]
        
        self.transforms: List[CanaryTransform] = []
        for name in transform_names:
            if name in TRANSFORM_REGISTRY:
                self.transforms.append(TRANSFORM_REGISTRY[name]())
            else:
                logger.warning(f"Unknown transform: {name}")
        
        if not self.transforms:
            raise ValueError("No valid transforms specified")
        
        # Lazy-load embedder
        self._embedder = None
        self._embed_tokenizer = None
        self.embedder_name = embedder_name
    
    def _load_embedder(self):
        """Lazy load embedding model."""
        if self._embedder is None:
            from transformers import AutoModel, AutoTokenizer
            
            self._embed_tokenizer = AutoTokenizer.from_pretrained(self.embedder_name)
            self._embedder = AutoModel.from_pretrained(self.embedder_name)
            self._embedder.to(self.device)
            self._embedder.eval()
    
    def _embed(self, text: str) -> np.ndarray:
        """Compute embedding for text."""
        import torch
        
        self._load_embedder()
        
        inputs = self._embed_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._embedder(**inputs)
            hidden = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
            embedding = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            return embedding[0].cpu().numpy()
    
    def _generate(self, prompt: str) -> str:
        """Generate output for a prompt."""
        import torch
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Deterministic for consistency
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    def generate_canaries(self, prompt: str) -> List[Tuple[str, str]]:
        """
        Generate canary variants of a prompt.
        
        Args:
            prompt: Original prompt
        
        Returns:
            List of (transform_name, transformed_prompt) tuples
        """
        canaries = []
        applicable_transforms = [t for t in self.transforms if t.is_applicable(prompt)]
        
        if not applicable_transforms:
            # Fallback to suffix if nothing applicable
            applicable_transforms = [SuffixTransform()]
        
        # Generate requested number of canaries
        for i in range(self.n_canaries):
            transform = applicable_transforms[i % len(applicable_transforms)]
            transformed = transform(prompt)
            canaries.append((transform.name, transformed))
        
        return canaries
    
    def score(
        self,
        prompt: str,
        original_output: Optional[str] = None,
        return_details: bool = False
    ) -> float:
        """
        Compute canary consistency score.
        
        Higher scores indicate inconsistency (potentially suspicious).
        
        Args:
            prompt: Original prompt
            original_output: Pre-computed output (will generate if None)
            return_details: Whether to return detailed breakdown
        
        Returns:
            Canary score (max pairwise distance) or dict with details
        """
        # Generate output for original prompt if not provided
        if original_output is None:
            original_output = self._generate(prompt)
        
        original_emb = self._embed(original_output)
        
        # Generate and score canaries
        canaries = self.generate_canaries(prompt)
        distances = []
        details = []
        
        for transform_name, canary_prompt in canaries:
            canary_output = self._generate(canary_prompt)
            canary_emb = self._embed(canary_output)
            
            dist = cosine_distance(original_emb, canary_emb)
            distances.append(dist)
            
            if return_details:
                details.append({
                    "transform": transform_name,
                    "canary_prompt": canary_prompt[:100] + "..." if len(canary_prompt) > 100 else canary_prompt,
                    "canary_output": canary_output[:100] + "..." if len(canary_output) > 100 else canary_output,
                    "distance": float(dist)
                })
        
        # Use max distance as score (most divergent canary)
        score = max(distances) if distances else 0.0
        
        if return_details:
            return {
                "score": float(score),
                "mean_distance": float(np.mean(distances)),
                "std_distance": float(np.std(distances)),
                "n_canaries": len(canaries),
                "canaries": details
            }
        
        return float(score)
    
    def score_with_outputs(
        self,
        prompt: str,
        outputs: List[str]
    ) -> float:
        """
        Score consistency given pre-generated outputs.
        
        Args:
            prompt: Original prompt (not used, for API consistency)
            outputs: List of outputs [original, canary1, canary2, ...]
        
        Returns:
            Max pairwise distance
        """
        if len(outputs) < 2:
            return 0.0
        
        embeddings = [self._embed(text) for text in outputs]
        original_emb = embeddings[0]
        
        distances = [
            cosine_distance(original_emb, emb)
            for emb in embeddings[1:]
        ]
        
        return float(max(distances)) if distances else 0.0
