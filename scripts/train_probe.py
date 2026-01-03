#!/usr/bin/env python3
"""
Train a probe model for TDM.

Usage:
    python scripts/train_probe.py --data data/synthetic.csv --output artifacts/
    python scripts/train_probe.py --generate --n-clean 500 --n-triggered 500
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from tdm.config import TDMConfig
from tdm.utils import logger
from tdm.instrumentation import load_model_and_tokenizer
from tdm.probe import train_probe, extract_features_from_dataloader, ProbeMetadata
from tdm.datasets.synthetic_sleeper import generate_synthetic_dataset, SyntheticSleeperDataset


def main():
    parser = argparse.ArgumentParser(description="Train TDM probe model")
    
    # Data options
    parser.add_argument("--data", type=str, help="Path to CSV dataset")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data")
    parser.add_argument("--n-clean", type=int, default=500, help="Number of clean samples")
    parser.add_argument("--n-triggered", type=int, default=500, help="Number of triggered samples")
    parser.add_argument("--trigger-types", type=str, nargs="+", 
                       default=["keyword", "contextual"],
                       help="Trigger types to include")
    
    # Model options
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--layers", type=int, nargs="+", default=[-1], help="Layers to use")
    parser.add_argument("--pooling", type=str, default="last_token", 
                       choices=["last_token", "mean_last_k"])
    parser.add_argument("--pool-k", type=int, default=4, help="K for mean_last_k pooling")
    
    # Training options
    parser.add_argument("--probe-type", type=str, default="linear",
                       choices=["linear", "mahalanobis"])
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")
    
    # Output options
    parser.add_argument("--output", type=str, default="./artifacts", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Device handling
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    
    # Load or generate dataset
    if args.generate:
        logger.info(f"Generating synthetic dataset: {args.n_clean} clean, {args.n_triggered} triggered")
        dataset = generate_synthetic_dataset(
            n_clean=args.n_clean,
            n_triggered=args.n_triggered,
            trigger_types=args.trigger_types,
            seed=args.seed
        )
        
        # Save dataset
        data_path = os.path.join(args.output, "synthetic_data.csv")
        dataset.to_csv(data_path)
        logger.info(f"Dataset saved to {data_path}")
    elif args.data:
        logger.info(f"Loading dataset from {args.data}")
        dataset = SyntheticSleeperDataset.from_csv(args.data)
    else:
        logger.error("Must specify --data or --generate")
        sys.exit(1)
    
    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Positive rate: {sum(dataset.labels) / len(dataset):.2%}")
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    
    # Extract features
    logger.info("Extracting features...")
    features = extract_features_from_dataloader(
        model=model,
        tokenizer=tokenizer,
        prompts=dataset.prompts,
        layers=args.layers,
        pooling=args.pooling,
        pool_k=args.pool_k,
        batch_size=args.batch_size,
        device=device,
        show_progress=True
    )
    
    labels = np.array(dataset.labels)
    
    logger.info(f"Feature shape: {features.shape}")
    
    # Train probe
    logger.info(f"Training {args.probe_type} probe...")
    probe, stats = train_probe(
        features=features,
        labels=labels,
        probe_type=args.probe_type,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        verbose=True
    )
    
    # Save probe
    if args.probe_type == "linear":
        probe_path = os.path.join(args.output, "probe.pt")
        probe.save(probe_path)
    else:
        probe_path = os.path.join(args.output, "probe_unsup.pkl")
        probe.save(probe_path)
    
    # Save metadata
    metadata = ProbeMetadata(
        model_name=args.model,
        layers=args.layers,
        pooling=args.pooling,
        feature_dim=features.shape[1],
        probe_type=args.probe_type,
        train_samples=len(dataset),
        train_accuracy=stats.get("final_accuracy", stats.get("accuracy", 0)),
        timestamp=datetime.utcnow().isoformat() + "Z"
    )
    
    metadata_path = os.path.join(args.output, "probe_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata.to_dict(), f, indent=2)
    
    logger.info(f"Probe saved to {probe_path}")
    logger.info(f"Metadata saved to {metadata_path}")
    logger.info(f"Training accuracy: {stats.get('final_accuracy', stats.get('accuracy', 0)):.4f}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Probe type: {args.probe_type}")
    print(f"Feature dim: {features.shape[1]}")
    print(f"Training samples: {len(dataset)}")
    print(f"Accuracy: {stats.get('final_accuracy', stats.get('accuracy', 0)):.4f}")
    print(f"Output: {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
