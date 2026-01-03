"""
Semantic drift detection for TDM.

Detects suspicious changes in model output semantics using embeddings.
"""

import os
from typing import List, Optional, Union

import numpy as np
import torch

from tdm.utils import logger, cosine_distance, ensure_numpy, RollingBuffer


class DriftDetector:
    """
    Detects semantic drift in model outputs using embedding similarity.
    
    Compares output embeddings against a baseline (reference or rolling buffer).
    """
    
    def __init__(
        self,
        embedder_name: str = "distilbert-base-uncased",
        device: str = "cpu",
        buffer_size: int = 100,
        buffer_path: Optional[str] = None
    ):
        """
        Initialize drift detector.
        
        Args:
            embedder_name: HuggingFace model name for embeddings
            device: Device for embedding computation
            buffer_size: Size of rolling baseline buffer
            buffer_path: Path to save/load buffer
        """
        self.embedder_name = embedder_name
        self.device = device
        self.buffer_size = buffer_size
        self.buffer_path = buffer_path
        
        self._embedder = None
        self._tokenizer = None
        
        # Rolling buffer for baseline embeddings
        if buffer_path and os.path.exists(buffer_path):
            self.buffer = RollingBuffer.load(buffer_path)
            logger.info(f"Loaded drift buffer with {len(self.buffer)} embeddings")
        else:
            self.buffer = RollingBuffer(max_size=buffer_size)
        
        self._centroid: Optional[np.ndarray] = None
        self._centroid_dirty = True
    
    def _load_embedder(self):
        """Lazy load the embedding model."""
        if self._embedder is None:
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"Loading embedding model: {self.embedder_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.embedder_name)
            self._embedder = AutoModel.from_pretrained(self.embedder_name)
            self._embedder.to(self.device)
            self._embedder.eval()
    
    def _embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Compute embeddings for texts.
        
        Args:
            texts: Single text or list of texts
        
        Returns:
            Embeddings [n_texts, embed_dim] or [embed_dim]
        """
        self._load_embedder()
        
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True
        else:
            squeeze = False
        
        # Tokenize
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings (mean pooling over non-padding tokens)
        with torch.no_grad():
            outputs = self._embedder(**inputs)
            hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]
            
            # Mean pooling with attention mask
            attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
            embeddings = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            embeddings = embeddings.cpu().numpy()
        
        if squeeze:
            return embeddings[0]
        return embeddings
    
    def _get_centroid(self) -> Optional[np.ndarray]:
        """Get the centroid of buffer embeddings."""
        if self._centroid_dirty or self._centroid is None:
            if len(self.buffer) == 0:
                return None
            embeddings = np.array(self.buffer.get_all())
            self._centroid = embeddings.mean(axis=0)
            self._centroid_dirty = False
        return self._centroid
    
    def update_buffer(self, text: str, embedding: Optional[np.ndarray] = None):
        """
        Add a safe output to the buffer.
        
        Args:
            text: Output text (used if embedding not provided)
            embedding: Pre-computed embedding
        """
        if embedding is None:
            embedding = self._embed(text)
        
        self.buffer.add(embedding)
        self._centroid_dirty = True
        
        if self.buffer_path:
            self.buffer.save(self.buffer_path)
    
    def score(
        self,
        output: str,
        reference: Optional[str] = None,
        update_buffer: bool = False
    ) -> float:
        """
        Compute drift score for an output.
        
        Higher scores indicate more drift (more suspicious).
        
        Args:
            output: Model output to check
            reference: Optional reference output to compare against
            update_buffer: Whether to add output to buffer if score is low
        
        Returns:
            Drift score (cosine distance, 0-2 range)
        """
        output_emb = self._embed(output)
        
        if reference is not None:
            # Compare to specific reference
            ref_emb = self._embed(reference)
            score = cosine_distance(output_emb, ref_emb)
        else:
            # Compare to buffer centroid
            centroid = self._get_centroid()
            if centroid is None:
                # No baseline yet, assume safe and update buffer
                self.update_buffer(output, output_emb)
                return 0.0
            
            score = cosine_distance(output_emb, centroid)
        
        # Optionally update buffer with safe outputs
        if update_buffer and score < 0.3:  # Low drift threshold
            self.update_buffer(output, output_emb)
        
        return float(score)
    
    def score_batch(
        self,
        outputs: List[str],
        references: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Compute drift scores for multiple outputs.
        
        Args:
            outputs: List of model outputs
            references: Optional list of references (same length as outputs)
        
        Returns:
            Array of drift scores
        """
        output_embs = self._embed(outputs)
        
        if references is not None:
            if len(references) != len(outputs):
                raise ValueError("References must match outputs length")
            ref_embs = self._embed(references)
            scores = np.array([
                cosine_distance(o, r) for o, r in zip(output_embs, ref_embs)
            ])
        else:
            centroid = self._get_centroid()
            if centroid is None:
                # Initialize buffer with first outputs
                for i, emb in enumerate(output_embs[:min(10, len(output_embs))]):
                    self.update_buffer(outputs[i], emb)
                return np.zeros(len(outputs))
            
            scores = np.array([
                cosine_distance(emb, centroid) for emb in output_embs
            ])
        
        return scores
    
    def clear_buffer(self):
        """Clear the baseline buffer."""
        self.buffer.clear()
        self._centroid = None
        self._centroid_dirty = True
        
        if self.buffer_path and os.path.exists(self.buffer_path):
            os.remove(self.buffer_path)
    
    def get_buffer_stats(self) -> dict:
        """Get statistics about the current buffer."""
        if len(self.buffer) == 0:
            return {"size": 0, "centroid_norm": 0.0}
        
        centroid = self._get_centroid()
        embeddings = np.array(self.buffer.get_all())
        
        # Compute distances from centroid
        distances = np.array([
            cosine_distance(emb, centroid) for emb in embeddings
        ])
        
        return {
            "size": len(self.buffer),
            "centroid_norm": float(np.linalg.norm(centroid)),
            "mean_distance": float(distances.mean()),
            "max_distance": float(distances.max()),
            "std_distance": float(distances.std())
        }
