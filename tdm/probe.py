"""
Probe models for detecting anomalous activations in TDM.

Supports supervised linear probes and unsupervised Mahalanobis distance.
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Tuple, Literal, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tdm.utils import logger, ensure_numpy, ensure_tensor


@dataclass
class ProbeMetadata:
    """Metadata for a trained probe."""
    model_name: str
    layers: List[int]
    pooling: str
    feature_dim: int
    probe_type: str
    train_samples: int
    train_accuracy: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "layers": self.layers,
            "pooling": self.pooling,
            "feature_dim": self.feature_dim,
            "probe_type": self.probe_type,
            "train_samples": self.train_samples,
            "train_accuracy": self.train_accuracy,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ProbeMetadata":
        return cls(**data)


class LinearProbe(nn.Module):
    """
    Linear probe for binary classification of activations.
    
    Given activation features, outputs probability of defection.
    """
    
    def __init__(self, input_dim: int, dropout: float = 0.1):
        """
        Initialize linear probe.
        
        Args:
            input_dim: Dimension of input features
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, 1)
        
        # Initialize weights
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch, input_dim]
        
        Returns:
            Logits [batch, 1]
        """
        x = self.dropout(x)
        return self.linear(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probability of defection.
        
        Args:
            x: Input features [batch, input_dim]
        
        Returns:
            Probabilities [batch]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(-1)
    
    def get_direction(self) -> torch.Tensor:
        """
        Get the defection direction vector (normalized weight).
        
        Returns:
            Normalized weight vector [input_dim]
        """
        w = self.linear.weight.data.squeeze()  # [input_dim]
        return w / (w.norm() + 1e-8)
    
    def save(self, path: str):
        """Save probe weights."""
        torch.save({
            "input_dim": self.input_dim,
            "state_dict": self.state_dict()
        }, path)
        logger.info(f"Probe saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "LinearProbe":
        """Load probe from file."""
        checkpoint = torch.load(path, map_location=device)
        probe = cls(input_dim=checkpoint["input_dim"])
        probe.load_state_dict(checkpoint["state_dict"])
        probe.to(device)
        probe.eval()
        logger.info(f"Probe loaded from {path}")
        return probe


class MahalanobisProbe:
    """
    Unsupervised anomaly detection using Mahalanobis distance.
    
    Fits mean and covariance on clean samples, scores new samples
    by their distance from the clean distribution.
    """
    
    def __init__(self, regularization: float = 1e-5):
        """
        Initialize Mahalanobis probe.
        
        Args:
            regularization: Regularization for covariance inversion
        """
        self.regularization = regularization
        self.mean: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.fitted = False
    
    def fit(self, features: np.ndarray):
        """
        Fit the probe on clean (normal) features.
        
        Args:
            features: Clean feature vectors [n_samples, dim]
        """
        features = ensure_numpy(features)
        
        self.mean = features.mean(axis=0)
        
        # Compute covariance with regularization
        centered = features - self.mean
        cov = np.cov(centered.T)
        
        # Handle 1D case
        if cov.ndim == 0:
            cov = np.array([[cov]])
        
        # Regularize and invert
        cov_reg = cov + self.regularization * np.eye(cov.shape[0])
        self.cov_inv = np.linalg.inv(cov_reg)
        
        self.fitted = True
        logger.info(f"MahalanobisProbe fitted on {len(features)} samples")
    
    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance scores.
        
        Args:
            features: Feature vectors [n_samples, dim] or [dim]
        
        Returns:
            Distance scores [n_samples] or scalar
        """
        if not self.fitted:
            raise ValueError("Probe not fitted. Call fit() first.")
        
        features = ensure_numpy(features)
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        
        centered = features - self.mean
        distances = np.sum(centered @ self.cov_inv * centered, axis=1)
        distances = np.sqrt(np.maximum(distances, 0))  # Ensure non-negative
        
        if squeeze:
            return distances[0]
        return distances
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Convert distance to probability-like score.
        
        Uses CDF of chi-squared distribution.
        """
        from scipy import stats
        
        distances_sq = self.score(features) ** 2
        dim = self.mean.shape[0]
        
        # Probability that a point from the fitted distribution
        # would have a distance >= this distance
        p_values = 1 - stats.chi2.cdf(distances_sq, df=dim)
        
        # Convert to "risk" score (higher = more anomalous)
        return 1 - p_values
    
    def save(self, path: str):
        """Save probe to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                "mean": self.mean,
                "cov_inv": self.cov_inv,
                "regularization": self.regularization
            }, f)
        logger.info(f"MahalanobisProbe saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "MahalanobisProbe":
        """Load probe from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        probe = cls(regularization=data["regularization"])
        probe.mean = data["mean"]
        probe.cov_inv = data["cov_inv"]
        probe.fitted = True
        logger.info(f"MahalanobisProbe loaded from {path}")
        return probe


def train_probe(
    features: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    probe_type: Literal["linear", "mahalanobis"] = "linear",
    device: str = "cpu",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.1,
    early_stopping_patience: int = 10,
    verbose: bool = True
) -> Tuple[Union[LinearProbe, MahalanobisProbe], Dict]:
    """
    Train a probe on features and labels.
    
    Args:
        features: Feature vectors [n_samples, dim]
        labels: Binary labels [n_samples]
        probe_type: Type of probe to train
        device: Device for training
        epochs: Maximum training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        weight_decay: L2 regularization
        val_split: Validation split ratio
        early_stopping_patience: Early stopping patience
        verbose: Print training progress
    
    Returns:
        Tuple of (trained probe, training stats dict)
    """
    features = ensure_numpy(features).astype(np.float32)
    labels = ensure_numpy(labels).astype(np.float32)
    
    n_samples, feature_dim = features.shape
    
    if probe_type == "mahalanobis":
        # Unsupervised: fit only on clean (label=0) samples
        clean_mask = labels == 0
        clean_features = features[clean_mask]
        
        probe = MahalanobisProbe()
        probe.fit(clean_features)
        
        # Evaluate on all data
        scores = probe.score(features)
        # Threshold at 95th percentile of clean scores
        clean_scores = scores[clean_mask]
        threshold = np.percentile(clean_scores, 95)
        
        predictions = (scores > threshold).astype(int)
        accuracy = (predictions == labels).mean()
        
        stats = {
            "probe_type": "mahalanobis",
            "n_clean_samples": int(clean_mask.sum()),
            "threshold": float(threshold),
            "accuracy": float(accuracy)
        }
        
        if verbose:
            logger.info(f"MahalanobisProbe trained: accuracy={accuracy:.4f}")
        
        return probe, stats
    
    # Linear probe training
    # Shuffle and split
    indices = np.random.permutation(n_samples)
    val_size = int(n_samples * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    X_train = torch.from_numpy(features[train_indices]).to(device)
    y_train = torch.from_numpy(labels[train_indices]).to(device)
    X_val = torch.from_numpy(features[val_indices]).to(device)
    y_val = torch.from_numpy(labels[val_indices]).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize probe
    probe = LinearProbe(input_dim=feature_dim).to(device)
    
    # Handle class imbalance (use float32 for MPS compatibility)
    pos_weight = float((labels == 0).sum() / max((labels == 1).sum(), 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    
    optimizer = optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        probe.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = probe(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val).squeeze(-1)
            val_loss = criterion(val_logits, y_val).item()
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_preds == y_val).float().mean().item()
        
        scheduler.step(val_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
                       f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = probe.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best weights
    if best_state is not None:
        probe.load_state_dict(best_state)
    
    # Final evaluation
    probe.eval()
    with torch.no_grad():
        all_features = torch.from_numpy(features).to(device)
        all_probs = probe.predict_proba(all_features).cpu().numpy()
        predictions = (all_probs > 0.5).astype(int)
        accuracy = (predictions == labels).mean()
    
    stats = {
        "probe_type": "linear",
        "n_train_samples": len(train_indices),
        "n_val_samples": len(val_indices),
        "best_val_loss": float(best_val_loss),
        "final_accuracy": float(accuracy)
    }
    
    if verbose:
        logger.info(f"LinearProbe trained: accuracy={accuracy:.4f}")
    
    return probe, stats


def load_probe(
    path: str,
    probe_type: Literal["linear", "mahalanobis"] = "linear",
    device: str = "cpu"
) -> Union[LinearProbe, MahalanobisProbe]:
    """
    Load a saved probe.
    
    Args:
        path: Path to saved probe
        probe_type: Type of probe
        device: Device to load to
    
    Returns:
        Loaded probe
    """
    if probe_type == "linear":
        return LinearProbe.load(path, device)
    else:
        return MahalanobisProbe.load(path)


def extract_features_from_dataloader(
    model,
    tokenizer,
    prompts: List[str],
    layers: List[int] = [-1],
    pooling: str = "last_token",
    pool_k: int = 4,
    batch_size: int = 8,
    device: str = "cpu",
    show_progress: bool = True
) -> np.ndarray:
    """
    Extract features for a list of prompts in batches.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompts: List of text prompts
        layers: Layers to capture
        pooling: Pooling strategy
        pool_k: K for mean_last_k pooling
        batch_size: Batch size
        device: Device
        show_progress: Show progress bar
    
    Returns:
        Features array [n_prompts, feature_dim]
    """
    from tdm.instrumentation import get_residual_activations
    
    all_features = []
    
    iterator = range(0, len(prompts), batch_size)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Extracting features")
        except ImportError:
            pass
    
    for i in iterator:
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Extract features
        features = get_residual_activations(
            model,
            input_ids,
            layers=layers,
            pooling=pooling,
            pool_k=pool_k,
            attention_mask=attention_mask
        )
        
        all_features.append(features.cpu().numpy())
    
    return np.concatenate(all_features, axis=0)
