#!/usr/bin/env python3
"""
Calibrate TDM detection thresholds.

Research-grade calibration with strict sample size enforcement for low-FPR.

Usage:
    python scripts/calibrate.py --data artifacts/synthetic_data.csv --output artifacts/
    python scripts/calibrate.py --generate --n-samples 500
    python scripts/calibrate.py --generate --n-samples 10000 --alpha 0.001 --strict
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
from tdm.instrumentation import load_model_and_tokenizer, get_residual_activations
from tdm.probe import load_probe, LinearProbe, MahalanobisProbe
from tdm.drift import DriftDetector
from tdm.canary import CanaryChecker
from tdm.calibration import Calibrator, InsufficientSamplesError, MIN_SAMPLES_MULTIPLIER
from tdm.datasets.synthetic_sleeper import (
    generate_synthetic_dataset, 
    generate_calibration_dataset,
    SyntheticSleeperDataset
)


def main():
    parser = argparse.ArgumentParser(description="Calibrate TDM thresholds (Research-Grade)")
    
    # Data options
    parser.add_argument("--data", type=str, help="Path to CSV dataset (clean samples only)")
    parser.add_argument("--generate", action="store_true", help="Generate calibration data")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of calibration samples")
    
    # Model options
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--layers", type=int, nargs="+", default=[-1])
    parser.add_argument("--pooling", type=str, default="last_token")
    
    # Calibration options
    parser.add_argument("--alpha", type=float, default=0.05, help="FPR budget (e.g., 0.001 for 1e-3)")
    parser.add_argument("--method", type=str, default="quantile",
                       choices=["quantile", "conformal"])
    parser.add_argument("--strict", action="store_true", 
                       help="Error (not warn) on insufficient samples")
    parser.add_argument("--n-canaries", type=int, default=4)
    
    # Paths
    parser.add_argument("--probe-path", type=str, default="./artifacts/probe.pt")
    parser.add_argument("--output", type=str, default="./artifacts")
    
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output, exist_ok=True)
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    # Calculate required samples
    min_required = int(np.ceil(MIN_SAMPLES_MULTIPLIER / args.alpha))
    
    print("\n" + "=" * 60)
    print("TDM Calibration (Research-Grade)")
    print("=" * 60)
    print(f"Alpha (target FPR): {args.alpha}")
    print(f"Minimum samples required: {min_required}")
    print(f"Requested samples: {args.n_samples}")
    print(f"Strict mode: {args.strict}")
    print("=" * 60 + "\n")
    
    # Check sample sufficiency upfront
    if args.n_samples < min_required:
        msg = (
            f"WARNING: Requesting {args.n_samples} samples but alpha={args.alpha} "
            f"requires >= {min_required} samples for reliable calibration.\n"
            f"For research-grade results:\n"
            f"  - alpha=0.05 -> min 200 samples\n"
            f"  - alpha=0.01 -> min 1,000 samples\n"
            f"  - alpha=0.001 -> min 10,000 samples\n"
            f"  - alpha=0.0001 -> min 100,000 samples"
        )
        if args.strict:
            logger.error(msg)
            sys.exit(1)
        else:
            logger.warning(msg)
            print("\n⚠️  Results may be UNSTABLE and NOT research-grade.\n")
    
    # Load or generate calibration data
    if args.generate:
        logger.info(f"Generating {args.n_samples} clean calibration samples")
        if args.n_samples >= 1000:
            # Use optimized generator for large datasets
            dataset = generate_calibration_dataset(
                n_samples=args.n_samples,
                seed=args.seed
            )
        else:
            dataset = generate_synthetic_dataset(
                n_clean=args.n_samples,
                n_triggered=0,
                seed=args.seed
            )
    elif args.data:
        logger.info(f"Loading data from {args.data}")
        dataset = SyntheticSleeperDataset.from_csv(args.data)
        # Filter to clean samples only
        dataset = dataset.get_clean()
    else:
        logger.error("Must specify --data or --generate")
        sys.exit(1)
    
    logger.info(f"Calibration samples: {len(dataset)}")
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    
    # Load probe
    probe = None
    if os.path.exists(args.probe_path):
        logger.info(f"Loading probe from {args.probe_path}")
        probe = LinearProbe.load(args.probe_path, device)
    else:
        probe_unsup = args.probe_path.replace("probe.pt", "probe_unsup.pkl")
        if os.path.exists(probe_unsup):
            probe = MahalanobisProbe.load(probe_unsup)
        else:
            logger.warning("No probe found, probe scores will be 0")
    
    # Initialize detectors
    drift_detector = DriftDetector(
        embedder_name="distilbert-base-uncased",
        device=device
    )
    
    canary_checker = CanaryChecker(
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_canaries=args.n_canaries
    )
    
    # Compute scores for calibration samples
    logger.info("Computing calibration scores...")
    
    probe_scores = []
    drift_scores = []
    canary_scores = []
    
    # Determine how many canary samples to compute (expensive)
    n_canary_samples = min(100, len(dataset))
    
    for i, prompt in enumerate(dataset.prompts):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing {i + 1}/{len(dataset)}")
        
        # Probe score
        if probe is not None:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)
            
            features = get_residual_activations(
                model, input_ids,
                layers=args.layers,
                pooling=args.pooling
            )
            
            if isinstance(probe, LinearProbe):
                score = probe.predict_proba(features).item()
            else:
                score = probe.predict_proba(features.cpu().numpy())[0]
            probe_scores.append(score)
        else:
            probe_scores.append(0.0)
        
        # Generate output for drift/canary
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Drift score
        drift_score = drift_detector.score(output_text, update_buffer=True)
        drift_scores.append(drift_score)
        
        # Canary score (limited samples for speed)
        if i < n_canary_samples:
            canary_score = canary_checker.score(prompt, original_output=output_text)
        else:
            canary_score = 0.0
        canary_scores.append(canary_score)
    
    # Calibrate
    logger.info("Calibrating thresholds...")
    
    calibrator = Calibrator(alpha=args.alpha, method=args.method)
    
    scores = {
        "probe": np.array(probe_scores),
        "drift": np.array(drift_scores),
    }
    
    # Only include canary if we have enough samples
    if len(canary_scores) >= min(100, min_required):
        scores["canary"] = np.array(canary_scores[:n_canary_samples])
    
    try:
        thresholds = calibrator.calibrate(scores, strict=args.strict)
    except InsufficientSamplesError as e:
        logger.error(f"Calibration failed: {e}")
        sys.exit(1)
    
    # Save
    thresholds_path = os.path.join(args.output, "thresholds.json")
    calibrator.save(thresholds_path)
    
    # Also save calibration stats separately
    stats_path = os.path.join(args.output, "calibration_stats.json")
    with open(stats_path, 'w') as f:
        stats = {
            "alpha": args.alpha,
            "method": args.method,
            "n_samples": len(dataset),
            "min_required": min_required,
            "sufficient": len(dataset) >= min_required,
            "thresholds": thresholds,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "score_stats": {
                name: {
                    "mean": float(np.mean(s)),
                    "std": float(np.std(s)),
                    "min": float(np.min(s)),
                    "max": float(np.max(s)),
                    "n_samples": len(s)
                }
                for name, s in scores.items()
            },
            "dataset_hash": calibrator.dataset_hash
        }
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Calibration Summary")
    print("=" * 60)
    print(f"Alpha (FPR budget): {args.alpha}")
    print(f"Method: {args.method}")
    print(f"Calibration samples: {len(dataset)} (required: {min_required})")
    
    if len(dataset) >= min_required:
        print("✓ Sample size is SUFFICIENT for reliable calibration")
    else:
        print("⚠ Sample size is INSUFFICIENT - results may be unstable")
    
    print("\nThresholds:")
    for name, thresh in thresholds.items():
        n_samples = calibrator.stats[name]["n_samples"]
        sufficient = calibrator.stats[name]["sufficient"]
        status = "✓" if sufficient else "⚠"
        print(f"  {status} {name}: {thresh:.6f} (n={n_samples})")
    
    print(f"\nSaved to: {thresholds_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
