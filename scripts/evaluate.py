#!/usr/bin/env python3
"""
Evaluate TDM on a test dataset.

Research-grade evaluation with low-FPR metrics and baseline comparisons.

Usage:
    python scripts/evaluate.py --data data/test.csv
    python scripts/evaluate.py --generate --n-samples 200
    python scripts/evaluate.py --generate --held-out-trigger --n-samples 500
"""

import argparse
import os
import sys
import json
import re
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from tdm.config import TDMConfig
from tdm.utils import logger, compute_metrics
from tdm.instrumentation import load_model_and_tokenizer, get_residual_activations
from tdm.probe import load_probe, LinearProbe, MahalanobisProbe
from tdm.drift import DriftDetector
from tdm.canary import CanaryChecker
from tdm.calibration import load_calibration
from tdm.fusion import SignalFusion, check_fusion_quality
from tdm.datasets.synthetic_sleeper import (
    generate_synthetic_dataset, 
    generate_research_dataset,
    SyntheticSleeperDataset,
    TriggerGroup,
    TRIGGER_TYPES
)


def evaluate_detector(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    name: str
) -> Dict:
    """Evaluate a single detector with research-grade metrics."""
    metrics = compute_metrics(y_true, y_scores, threshold)
    
    # Compute confusion matrix at threshold
    y_pred = (y_scores >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    metrics.update({
        "name": name,
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "tpr": float(tp / max(tp + fn, 1)),
        "fpr": float(fp / max(fp + tn, 1))
    })
    
    return metrics


def trigger_match_baseline(prompts: List[str], labels: np.ndarray) -> Dict:
    """
    Baseline: Simple trigger string matching.
    
    This is a "trivial" baseline that just checks for known trigger strings.
    If TDM can't beat this, something is wrong.
    """
    all_triggers = []
    for trigger_type in TRIGGER_TYPES.values():
        all_triggers.extend(trigger_type["triggers"])
    
    # Compile patterns
    pattern = re.compile('|'.join(re.escape(t) for t in all_triggers), re.IGNORECASE)
    
    scores = []
    for prompt in prompts:
        match = pattern.search(prompt)
        scores.append(1.0 if match else 0.0)
    
    scores = np.array(scores)
    metrics = compute_metrics(labels, scores, threshold=0.5)
    
    return {
        "name": "trigger_match",
        **metrics
    }


def prompt_length_baseline(prompts: List[str], labels: np.ndarray) -> Dict:
    """
    Baseline: Prompt length (triggered prompts are typically longer).
    
    This is a "dumb" baseline to establish floor.
    """
    lengths = np.array([len(p) for p in prompts])
    # Normalize to [0, 1]
    lengths_norm = (lengths - lengths.min()) / (lengths.max() - lengths.min() + 1e-8)
    
    metrics = compute_metrics(labels, lengths_norm, threshold=0.5)
    
    return {
        "name": "length_baseline",
        **metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate TDM (Research-Grade)")
    
    # Data options
    parser.add_argument("--data", type=str, help="Path to test CSV")
    parser.add_argument("--generate", action="store_true", help="Generate test data")
    parser.add_argument("--research", action="store_true", help="Use research-grade dataset")
    parser.add_argument("--n-samples", type=int, default=200, help="Samples per class")
    parser.add_argument("--trigger-types", type=str, nargs="+",
                       default=["keyword", "contextual"])
    parser.add_argument("--held-out-trigger", action="store_true", 
                       help="Include held-out trigger group B for generalization testing")
    
    # Model options
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layers", type=int, nargs="+", default=[-1])
    parser.add_argument("--pooling", type=str, default="last_token")
    
    # Paths
    parser.add_argument("--artifacts", type=str, default="./artifacts")
    parser.add_argument("--n-canaries", type=int, default=4)
    
    parser.add_argument("--output", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=43)  # Different from training
    
    # Evaluation options
    parser.add_argument("--include-baselines", action="store_true", default=True,
                       help="Include baseline comparisons")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output, exist_ok=True)
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    # Load or generate test data
    if args.generate:
        if args.research:
            logger.info("Generating RESEARCH-GRADE test dataset")
            dataset = generate_research_dataset(
                n_clean=args.n_samples,
                n_triggered_per_group=args.n_samples,
                trigger_types=args.trigger_types,
                include_held_out=args.held_out_trigger,
                seed=args.seed
            )
        else:
            logger.info("Generating test dataset")
            dataset = generate_synthetic_dataset(
                n_clean=args.n_samples,
                n_triggered=args.n_samples,
                trigger_types=args.trigger_types,
                seed=args.seed
            )
    elif args.data:
        dataset = SyntheticSleeperDataset.from_csv(args.data)
    else:
        logger.error("Must specify --data or --generate")
        sys.exit(1)
    
    logger.info(f"Test samples: {len(dataset)}")
    logger.info(f"Positive rate: {sum(dataset.labels) / len(dataset):.2%}")
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    
    # Load probe
    probe_path = os.path.join(args.artifacts, "probe.pt")
    probe = None
    if os.path.exists(probe_path):
        probe = LinearProbe.load(probe_path, device)
    else:
        probe_unsup = os.path.join(args.artifacts, "probe_unsup.pkl")
        if os.path.exists(probe_unsup):
            probe = MahalanobisProbe.load(probe_unsup)
    
    # Load calibration
    thresholds_path = os.path.join(args.artifacts, "thresholds.json")
    calibrator = None
    thresholds = {"probe": 0.5, "drift": 0.5, "canary": 0.5, "fused": 0.5}
    if os.path.exists(thresholds_path):
        calibrator = load_calibration(thresholds_path)
        thresholds = calibrator.thresholds.copy()
    
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
    
    fusion = SignalFusion(calibrator=calibrator)
    
    # Evaluate
    logger.info("Evaluating...")
    
    all_probe_scores = []
    all_drift_scores = []
    all_canary_scores = []
    all_fused_scores = []
    
    import time
    latencies = []
    
    for i, prompt in enumerate(dataset.prompts):
        start = time.time()
        
        if (i + 1) % 50 == 0:
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
                probe_score = probe.predict_proba(features).item()
            else:
                probe_score = probe.predict_proba(features.cpu().numpy())[0]
        else:
            probe_score = 0.0
        
        all_probe_scores.append(probe_score)
        
        # Generate output
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Drift score
        drift_score = drift_detector.score(output_text)
        all_drift_scores.append(drift_score)
        
        # Canary score (compute for subset to save time)
        if i < 100:
            canary_score = canary_checker.score(prompt, original_output=output_text)
        else:
            canary_score = 0.0
        all_canary_scores.append(canary_score)
        
        # Fused score
        signals = {"probe": probe_score, "drift": drift_score, "canary": canary_score}
        fused_result = fusion.fuse(signals, thresholds)
        all_fused_scores.append(fused_result["risk"])
        
        latencies.append((time.time() - start) * 1000)
    
    # Compute metrics
    y_true = np.array(dataset.labels)
    
    results = {
        "dataset": {
            "n_samples": len(dataset),
            "n_positive": int(sum(dataset.labels)),
            "n_negative": int(len(dataset) - sum(dataset.labels)),
            "trigger_types": list(set(dataset.groups)),
            "research_mode": args.research if hasattr(args, 'research') else False
        },
        "thresholds": thresholds,
        "detectors": {},
        "baselines": {}
    }
    
    # Evaluate each detector
    for name, scores in [
        ("probe", all_probe_scores),
        ("drift", all_drift_scores),
        ("canary", all_canary_scores[:100] if len(all_canary_scores) > 100 else all_canary_scores),
        ("fused", all_fused_scores)
    ]:
        scores_array = np.array(scores)
        labels_subset = y_true[:len(scores_array)]
        
        threshold = thresholds.get(name, 0.5)
        metrics = evaluate_detector(labels_subset, scores_array, threshold, name)
        results["detectors"][name] = metrics
    
    # Compute baselines
    if args.include_baselines:
        results["baselines"]["trigger_match"] = trigger_match_baseline(dataset.prompts, y_true)
        results["baselines"]["length"] = prompt_length_baseline(dataset.prompts, y_true)
    
    # Check fusion quality
    probe_auroc = results["detectors"]["probe"]["auroc"]
    fused_auroc = results["detectors"]["fused"]["auroc"]
    fusion_check = check_fusion_quality(probe_auroc, fused_auroc)
    results["fusion_quality"] = fusion_check
    
    # Latency stats
    results["latency"] = {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95))
    }
    
    # Per-group evaluation
    results["by_group"] = {}
    for group in set(dataset.groups):
        group_indices = [i for i, g in enumerate(dataset.groups) if g == group]
        if len(group_indices) == 0:
            continue
        
        group_labels = y_true[group_indices]
        group_fused = np.array([all_fused_scores[i] for i in group_indices])
        group_probe = np.array([all_probe_scores[i] for i in group_indices])
        
        group_metrics = {
            "fused": evaluate_detector(
                group_labels, group_fused, 
                thresholds.get("fused", 0.5), 
                f"fused_{group}"
            ),
            "probe": evaluate_detector(
                group_labels, group_probe,
                thresholds.get("probe", 0.5),
                f"probe_{group}"
            )
        }
        results["by_group"][group] = group_metrics
    
    # Held-out trigger analysis (if available)
    if hasattr(dataset, 'trigger_groups') and dataset.trigger_groups:
        results["by_trigger_group"] = {}
        for tg_val in set(dataset.trigger_groups):
            if tg_val == "none":
                continue
            tg_indices = [i for i, g in enumerate(dataset.trigger_groups) if g == tg_val]
            if len(tg_indices) == 0:
                continue
            
            tg_labels = y_true[tg_indices]
            tg_fused = np.array([all_fused_scores[i] for i in tg_indices])
            tg_probe = np.array([all_probe_scores[i] for i in tg_indices])
            
            results["by_trigger_group"][tg_val] = {
                "n_samples": len(tg_indices),
                "fused": evaluate_detector(tg_labels, tg_fused, thresholds.get("fused", 0.5), f"fused_{tg_val}"),
                "probe": evaluate_detector(tg_labels, tg_probe, thresholds.get("probe", 0.5), f"probe_{tg_val}")
            }
    
    # Save results
    results_path = os.path.join(args.output, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESEARCH-GRADE EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nDataset: {len(dataset)} samples ({sum(dataset.labels)} positive)")
    
    print("\n--- Research-Grade Metrics (TPR @ Low FPR) ---")
    print("-" * 70)
    print(f"{'Detector':<12} {'AUROC':>8} {'TPR@1e-4':>10} {'TPR@1e-3':>10} {'TPR@1%':>8} {'TPR@5%':>8}")
    print("-" * 70)
    for name in ["probe", "fused"]:
        m = results["detectors"][name]
        print(f"{name:<12} {m['auroc']:>8.4f} {m['tpr_at_fpr_0001']:>10.4f} {m['tpr_at_fpr_001']:>10.4f} "
              f"{m['tpr_at_fpr_01']:>8.4f} {m['tpr_at_fpr_05']:>8.4f}")
    
    print("\n--- Baseline Comparisons ---")
    if "baselines" in results:
        for name, m in results["baselines"].items():
            print(f"  {name}: AUROC={m['auroc']:.4f}")
    
    print("\n--- Fusion Quality ---")
    fq = results["fusion_quality"]
    if fq["fusion_helps"]:
        print(f"  ✓ Fusion improves over probe-only by {fq['improvement']:.4f}")
    elif fq["fusion_hurts"]:
        print(f"  ✗ WARNING: Fusion hurts by {-fq['improvement']:.4f}")
    else:
        print(f"  ~ Fusion is neutral (improvement: {fq['improvement']:.4f})")
    
    print("\n--- Confusion Matrix (Fused, at threshold) ---")
    fused_m = results["detectors"]["fused"]
    print(f"  TP={fused_m['tp']:4d}  FP={fused_m['fp']:4d}")
    print(f"  FN={fused_m['fn']:4d}  TN={fused_m['tn']:4d}")
    print(f"  TPR={fused_m['tpr']:.4f}  FPR={fused_m['fpr']:.4f}")
    
    if "by_trigger_group" in results:
        print("\n--- Held-Out Trigger Generalization ---")
        for tg, tg_metrics in results["by_trigger_group"].items():
            probe_m = tg_metrics["probe"]
            print(f"  {tg}: AUROC={probe_m['auroc']:.4f}, TPR@1e-3={probe_m['tpr_at_fpr_001']:.4f}")
    
    print(f"\nLatency: {results['latency']['mean_ms']:.1f}ms (p95: {results['latency']['p95_ms']:.1f}ms)")
    print(f"\nResults saved to: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
