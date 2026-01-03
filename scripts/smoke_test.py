#!/usr/bin/env python3
"""
Smoke test for TDM.

Runs end-to-end: generate data -> train probe -> calibrate -> evaluate.

Research-grade mode available for publication-quality results.

Usage:
    python scripts/smoke_test.py                    # Standard (with sufficient calibration)
    python scripts/smoke_test.py --quick            # Quick test (fewer samples, will warn)
    python scripts/smoke_test.py --research         # Research-grade (10k+ samples)
    python scripts/smoke_test.py --alpha 0.01 --cal-samples 2000
"""

# Disable TensorFlow before any other imports
import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import os
import sys
import time
import json
import shutil
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from tdm.config import TDMConfig
from tdm.utils import logger, compute_metrics, cosine_distance
from tdm.instrumentation import load_model_and_tokenizer, get_residual_activations
from tdm.probe import train_probe, extract_features_from_dataloader, LinearProbe
from tdm.calibration import Calibrator, MIN_SAMPLES_MULTIPLIER
from tdm.fusion import SignalFusion, check_fusion_quality
from tdm.patching import ActivationPatcher, test_patching_correctness
from tdm.triggerspan import TriggerSpanLocalizer
from tdm.datasets.synthetic_sleeper import (
    generate_synthetic_dataset,
    generate_research_dataset,
    generate_calibration_dataset
)


class SimpleDriftScorer:
    """Simple drift scorer using the same GPT-2 model for embeddings."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.buffer = []
        self._centroid = None
    
    def _embed(self, text):
        """Get embedding by averaging last hidden states."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.transformer(input_ids)
            hidden = outputs.last_hidden_state  # [1, seq, hidden]
            # Mean pool
            embedding = hidden.mean(dim=1).squeeze(0).cpu().numpy()
        return embedding
    
    def score(self, text, update_buffer=False):
        """Score drift from baseline."""
        emb = self._embed(text)
        
        if len(self.buffer) == 0:
            self.buffer.append(emb)
            return 0.0
        
        # Compute centroid
        if self._centroid is None or update_buffer:
            self._centroid = np.mean(self.buffer, axis=0)
        
        dist = cosine_distance(emb, self._centroid)
        
        if update_buffer and dist < 0.5:
            self.buffer.append(emb)
            if len(self.buffer) > 50:
                self.buffer.pop(0)
            self._centroid = None  # Invalidate
        
        return float(dist)


def smoke_test(
    quick: bool = False, 
    research: bool = False,
    artifacts_dir: str = "./artifacts_smoke",
    device: str = None,
    n_train: int = None,
    n_cal: int = None,
    n_test: int = None,
    alpha: float = None,
    epochs: int = None
):
    """Run the full smoke test."""
    
    print("=" * 70)
    print("TDM Smoke Test")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Mode: {'RESEARCH' if research else 'QUICK' if quick else 'STANDARD'}")
    print(f"Artifacts: {artifacts_dir}")
    print()
    
    # Sanity warnings for non-research mode
    if quick:
        print("⚠️  QUICK MODE: Results are NOT research-grade")
        print("   Use --research for publication-quality evaluation\n")
    elif not research:
        print("ℹ️  STANDARD MODE: Use --research for publication-quality evaluation\n")
    
    # Clean artifacts directory
    if os.path.exists(artifacts_dir):
        shutil.rmtree(artifacts_dir)
    os.makedirs(artifacts_dir)
    
    # Device selection
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"Device: {device}")
    
    # Parameters based on mode (can be overridden by CLI args)
    if research:
        _n_train = n_train or 1000
        _n_cal = n_cal or 10000  # For alpha=1e-3
        _n_test = n_test or 2000
        _epochs = epochs or 100
        n_canaries = 4
        _alpha = alpha or 0.001  # Research-grade FPR
    elif quick:
        _n_train = n_train or 100
        _n_cal = n_cal or 50
        _n_test = n_test or 50
        _epochs = epochs or 20
        n_canaries = 2
        _alpha = alpha or 0.05
    else:
        # Standard mode - FIXED: use sufficient calibration samples
        _n_train = n_train or 600
        _n_cal = n_cal or 2000  # INCREASED from 500 to meet alpha=0.01 requirement
        _n_test = n_test or 400
        _epochs = epochs or 50
        n_canaries = 4
        _alpha = alpha or 0.01
    
    trigger_types = ["keyword", "contextual"]
    
    results = {
        "mode": "research" if research else "quick" if quick else "standard",
        "device": device,
        "alpha": _alpha,
        "config": {
            "n_train": _n_train,
            "n_cal": _n_cal,
            "n_test": _n_test,
            "epochs": _epochs
        },
        "stages": {},
        "sanity_checks": {}
    }
    
    # Stage 1: Generate synthetic dataset
    print("\n" + "-" * 70)
    print("Stage 1: Generating synthetic dataset")
    print("-" * 70)
    
    t0 = time.time()
    
    if research:
        # Use research-grade dataset with behavioral labels
        train_dataset = generate_research_dataset(
            n_clean=_n_train,
            n_triggered_per_group=_n_train,
            trigger_types=trigger_types,
            include_held_out=True,
            seed=42
        )
    else:
        train_dataset = generate_synthetic_dataset(
            n_clean=_n_train,
            n_triggered=_n_train,
            trigger_types=trigger_types,
            seed=42
        )
    
    # Generate calibration dataset (clean only)
    if research or _n_cal >= 1000:
        cal_dataset = generate_calibration_dataset(n_samples=_n_cal, seed=100)
    else:
        cal_dataset = generate_synthetic_dataset(
            n_clean=_n_cal,
            n_triggered=0,
            seed=100
        )
    
    test_dataset = generate_synthetic_dataset(
        n_clean=_n_test,
        n_triggered=_n_test,
        trigger_types=trigger_types,
        seed=200
    )
    
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Calibration: {len(cal_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Sample size check
    min_required = int(np.ceil(MIN_SAMPLES_MULTIPLIER / _alpha))
    if len(cal_dataset) < min_required:
        print(f"  ⚠️  Calibration samples ({len(cal_dataset)}) < required ({min_required}) for alpha={_alpha}")
        results["sanity_checks"]["calibration_sufficient"] = False
    else:
        print(f"  ✓ Calibration samples sufficient for alpha={_alpha}")
        results["sanity_checks"]["calibration_sufficient"] = True
    
    results["stages"]["generate_data"] = {
        "time_s": time.time() - t0,
        "train_samples": len(train_dataset),
        "cal_samples": len(cal_dataset),
        "test_samples": len(test_dataset),
        "min_required_for_alpha": min_required
    }
    
    # Stage 2: Load model
    print("\n" + "-" * 70)
    print("Stage 2: Loading model")
    print("-" * 70)
    
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer("gpt2", device)
    print(f"  Model: gpt2")
    print(f"  Layers: {model.config.n_layer}, Dim: {model.config.n_embd}")
    
    results["stages"]["load_model"] = {"time_s": time.time() - t0}
    
    # Stage 3: Extract features and train probe
    print("\n" + "-" * 70)
    print("Stage 3: Training probe")
    print("-" * 70)
    
    t0 = time.time()
    
    layers = [-1]
    pooling = "last_token"
    
    # Extract training features
    print("  Extracting features...")
    train_features = extract_features_from_dataloader(
        model=model,
        tokenizer=tokenizer,
        prompts=train_dataset.prompts,
        layers=layers,
        pooling=pooling,
        batch_size=8,
        device=device,
        show_progress=False
    )
    train_labels = np.array(train_dataset.labels)
    
    print(f"  Feature dim: {train_features.shape[1]}")
    print(f"  Training probe...")
    
    probe, train_stats = train_probe(
        features=train_features,
        labels=train_labels,
        probe_type="linear",
        device=device,
        epochs=_epochs,
        batch_size=32,
        verbose=False
    )
    
    # Save probe
    probe_path = os.path.join(artifacts_dir, "probe.pt")
    probe.save(probe_path)
    
    print(f"  Training accuracy: {train_stats['final_accuracy']:.4f}")
    
    results["stages"]["train_probe"] = {
        "time_s": time.time() - t0,
        "feature_dim": train_features.shape[1],
        "accuracy": train_stats["final_accuracy"]
    }
    
    # Stage 4: Calibrate thresholds
    print("\n" + "-" * 70)
    print("Stage 4: Calibrating thresholds")
    print("-" * 70)
    
    t0 = time.time()
    
    # Initialize drift detector
    drift_detector = SimpleDriftScorer(model, tokenizer, device)
    
    # Collect calibration scores
    cal_probe_scores = []
    cal_drift_scores = []
    
    print("  Computing calibration scores...")
    for i, prompt in enumerate(cal_dataset.prompts):
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(cal_dataset)}")
        
        # Probe score
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs["input_ids"].to(device)
        features = get_residual_activations(model, input_ids, layers=layers, pooling=pooling)
        probe_score = probe.predict_proba(features).item()
        cal_probe_scores.append(probe_score)
        
        # Generate output for drift
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        new_tokens = outputs[0, input_ids.shape[1]:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        drift_score = drift_detector.score(output_text, update_buffer=True)
        cal_drift_scores.append(drift_score)
    
    # Calibrate
    calibrator = Calibrator(alpha=_alpha, method="quantile")
    thresholds = calibrator.calibrate({
        "probe": np.array(cal_probe_scores),
        "drift": np.array(cal_drift_scores)
    }, strict=False)
    
    # Save calibration
    thresholds_path = os.path.join(artifacts_dir, "thresholds.json")
    calibrator.save(thresholds_path)
    
    print(f"  Thresholds: probe={thresholds['probe']:.4f}, drift={thresholds['drift']:.4f}")
    
    results["stages"]["calibrate"] = {
        "time_s": time.time() - t0,
        "alpha": _alpha,
        "thresholds": thresholds
    }
    
    # Stage 5: Evaluate on test set
    print("\n" + "-" * 70)
    print("Stage 5: Evaluating on test set")
    print("-" * 70)
    
    t0 = time.time()
    
    # Initialize fusion
    fusion = SignalFusion(calibrator=calibrator)
    
    test_probe_scores = []
    test_drift_scores = []
    test_fused_scores = []
    
    print("  Computing test scores...")
    for i, prompt in enumerate(test_dataset.prompts):
        # Probe score
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs["input_ids"].to(device)
        features = get_residual_activations(model, input_ids, layers=layers, pooling=pooling)
        probe_score = probe.predict_proba(features).item()
        test_probe_scores.append(probe_score)
        
        # Generate and score drift
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        new_tokens = outputs[0, input_ids.shape[1]:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        drift_score = drift_detector.score(output_text)
        test_drift_scores.append(drift_score)
        
        # Fuse
        signals = {"probe": probe_score, "drift": drift_score, "canary": 0.0}
        fused_result = fusion.fuse(signals, thresholds)
        test_fused_scores.append(fused_result["risk"])
    
    # Compute metrics including research-grade low-FPR
    test_labels = np.array(test_dataset.labels)
    
    probe_metrics = compute_metrics(test_labels, np.array(test_probe_scores), thresholds.get("probe", 0.5))
    fused_metrics = compute_metrics(test_labels, np.array(test_fused_scores), 0.5)
    
    print(f"\n  === RESEARCH-GRADE METRICS ===")
    print(f"  Probe AUROC: {probe_metrics['auroc']:.4f}")
    print(f"  Probe TPR@FPR=1e-4: {probe_metrics['tpr_at_fpr_0001']:.4f}")
    print(f"  Probe TPR@FPR=1e-3: {probe_metrics['tpr_at_fpr_001']:.4f}")
    print(f"  Probe TPR@FPR=1%: {probe_metrics['tpr_at_fpr_01']:.4f}")
    print(f"  Probe TPR@FPR=5%: {probe_metrics['tpr_at_fpr_05']:.4f}")
    print(f"\n  Fused AUROC: {fused_metrics['auroc']:.4f}")
    print(f"  Fused TPR@FPR=1e-3: {fused_metrics['tpr_at_fpr_001']:.4f}")
    
    # Check fusion quality
    fusion_quality = check_fusion_quality(probe_metrics['auroc'], fused_metrics['auroc'])
    if fusion_quality["fusion_hurts"]:
        print(f"  ⚠️  Fusion HURTS: {fusion_quality['improvement']:.4f}")
        results["sanity_checks"]["fusion_helps"] = False
    else:
        print(f"  ✓ Fusion quality: {fusion_quality['improvement']:+.4f}")
        results["sanity_checks"]["fusion_helps"] = not fusion_quality["fusion_hurts"]
    
    results["stages"]["evaluate"] = {
        "time_s": time.time() - t0,
        "probe_auroc": probe_metrics["auroc"],
        "probe_tpr_at_fpr_0001": probe_metrics["tpr_at_fpr_0001"],
        "probe_tpr_at_fpr_001": probe_metrics["tpr_at_fpr_001"],
        "probe_tpr_at_fpr_05": probe_metrics["tpr_at_fpr_05"],
        "fused_auroc": fused_metrics["auroc"],
        "fused_tpr_at_fpr_001": fused_metrics["tpr_at_fpr_001"],
        "fused_tpr_at_fpr_05": fused_metrics["tpr_at_fpr_05"]
    }
    
    # Stage 6: Test trigger span localization
    print("\n" + "-" * 70)
    print("Stage 6: Testing trigger span localization")
    print("-" * 70)
    
    t0 = time.time()
    
    # Get a high-risk triggered sample
    triggered = test_dataset.get_triggered()
    triggered_scores = [test_probe_scores[i] for i in range(len(test_dataset.prompts)) 
                       if test_dataset.labels[i] == 1]
    
    # Find highest risk triggered sample
    if len(triggered) > 0 and len(triggered_scores) > 0:
        high_risk_idx = np.argmax(triggered_scores)
        test_prompt = triggered.prompts[high_risk_idx]
        test_risk = triggered_scores[high_risk_idx]
        
        print(f"  Test prompt: {test_prompt[:60]}...")
        print(f"  Baseline risk: {test_risk:.4f}")
        
        def risk_fn(p):
            inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=256)
            features = get_residual_activations(model, inputs["input_ids"].to(device), layers=layers, pooling=pooling)
            return probe.predict_proba(features).item()
        
        localizer = TriggerSpanLocalizer(
            model=model,
            tokenizer=tokenizer,
            risk_fn=risk_fn,
            device=device,
            min_risk_threshold=0.3  # Lower threshold for testing
        )
        
        spans = localizer.localize_to_dict(test_prompt, top_k=3)
        
        if len(spans) > 0:
            top_delta = spans[0]["delta"]
            print(f"  Found {len(spans)} trigger spans:")
            for span in spans[:2]:
                print(f"    '{span['text']}' (delta={span['delta']:.4f})")
            
            # Sanity check: deltas should be meaningful if risk is high
            if test_risk > 0.5 and top_delta < 0.01:
                print(f"  ⚠️  High risk but low delta - forensics may be failing")
                results["sanity_checks"]["triggerspan_meaningful"] = False
            else:
                results["sanity_checks"]["triggerspan_meaningful"] = True
        else:
            print(f"  No spans found (risk may be below threshold)")
            results["sanity_checks"]["triggerspan_meaningful"] = test_risk < 0.3
        
        results["stages"]["triggerspan"] = {
            "time_s": time.time() - t0,
            "baseline_risk": test_risk,
            "n_spans": len(spans),
            "top_span": spans[0] if spans else None
        }
    else:
        print("  No triggered samples to test")
        results["stages"]["triggerspan"] = {"time_s": 0, "n_spans": 0}
    
    # Stage 7: Test causal validation
    print("\n" + "-" * 70)
    print("Stage 7: Testing causal validation")
    print("-" * 70)
    
    t0 = time.time()
    
    # Run patching unit test first
    print("  Running patching unit test...")
    patch_test = test_patching_correctness()
    results["sanity_checks"]["patching_math_correct"] = patch_test["patching_correct"]
    
    try:
        patcher = ActivationPatcher(
            model=model,
            tokenizer=tokenizer,
            probe=probe,
            patching_layer=-1,
            device=device,
            min_risk_for_validation=0.3
        )
        
        if len(triggered) > 0 and test_risk > 0.3:
            def probe_risk_fn(prompt):
                return risk_fn(prompt)
            
            def output_risk_fn(output_text):
                return drift_detector.score(output_text)
            
            causal_result = patcher.causal_validation(
                test_prompt,
                probe_risk_fn=probe_risk_fn,
                output_risk_fn=output_risk_fn,
                max_new_tokens=20
            )
            
            if causal_result.get("skipped"):
                print(f"  Skipped: {causal_result['reason']}")
            else:
                print(f"  Probe score before: {causal_result['probe_score_before']:.4f}")
                print(f"  Dot product before: {causal_result['dot_before']:.4f}")
                print(f"  Risk drop: {causal_result['risk_drop']:.4f}")
                print(f"  Causal confirmed: {causal_result['causal_confirmed']}")
            
            results["stages"]["causal_validation"] = {
                "time_s": time.time() - t0,
                **{k: v for k, v in causal_result.items() if k not in ['output_unpatched', 'output_patched', 'projections', 'tokens']}
            }
            
            # Sanity check for causal validation
            if not causal_result.get("skipped") and abs(causal_result["dot_before"]) < 0.001:
                print(f"  ⚠️  Dot product is tiny - patching won't have effect")
                results["sanity_checks"]["causal_meaningful"] = False
            else:
                results["sanity_checks"]["causal_meaningful"] = True
        else:
            print(f"  Skipped: no high-risk samples (max risk={test_risk:.4f})")
            results["stages"]["causal_validation"] = {"time_s": 0, "skipped": True}
            results["sanity_checks"]["causal_meaningful"] = None
            
    except Exception as e:
        print(f"  Error: {e}")
        results["stages"]["causal_validation"] = {"time_s": 0, "error": str(e)}
        results["sanity_checks"]["causal_meaningful"] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("Smoke Test Summary")
    print("=" * 70)
    
    total_time = sum(s.get("time_s", 0) for s in results["stages"].values())
    
    print(f"\nTotal time: {total_time:.1f}s")
    print("\nStage timings:")
    for stage, data in results["stages"].items():
        t = data.get("time_s", 0)
        print(f"  {stage:25s}: {t:6.1f}s")
    
    print(f"\n=== KEY METRICS ===")
    print(f"  Probe AUROC: {results['stages']['evaluate']['probe_auroc']:.4f}")
    print(f"  Probe TPR@FPR=1e-3: {results['stages']['evaluate']['probe_tpr_at_fpr_001']:.4f}")
    print(f"  Probe TPR@FPR=5%: {results['stages']['evaluate']['probe_tpr_at_fpr_05']:.4f}")
    
    print(f"\n=== SANITY CHECKS ===")
    all_pass = True
    for check, passed in results["sanity_checks"].items():
        if passed is None:
            status = "⊘ N/A"
        elif passed:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_pass = False
        print(f"  {status}: {check}")
    
    # Success criteria
    auroc = results["stages"]["evaluate"]["probe_auroc"]
    success = auroc > 0.6 and all_pass
    
    print(f"\n{'✓' if success else '✗'} Test {'PASSED' if success else 'NEEDS ATTENTION'}")
    print(f"  (AUROC > 0.6: {auroc:.4f}, All sanity checks: {all_pass})")
    
    if not research and not success:
        print(f"\n⚠️  For research-grade results, run: python scripts/smoke_test.py --research")
    
    # Save results
    results["total_time_s"] = total_time
    results["success"] = success
    results["timestamp"] = datetime.now().isoformat()
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)
    
    results_path = os.path.join(artifacts_dir, "smoke_test_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nResults saved to: {results_path}")
    print("=" * 70)
    
    return success


def main():
    parser = argparse.ArgumentParser(description="TDM Smoke Test")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer samples")
    parser.add_argument("--research", action="store_true", help="Research-grade mode (10k+ samples)")
    parser.add_argument("--artifacts", type=str, default="./artifacts_smoke",
                       help="Artifacts directory")
    
    # Custom configuration options
    parser.add_argument("--device", type=str, default=None, 
                       choices=["cpu", "cuda", "mps"],
                       help="Device (default: auto-detect)")
    parser.add_argument("--train-samples", type=int, default=None,
                       help="Number of training samples")
    parser.add_argument("--cal-samples", type=int, default=None,
                       help="Number of calibration samples")
    parser.add_argument("--test-samples", type=int, default=None,
                       help="Number of test samples")
    parser.add_argument("--alpha", type=float, default=None,
                       help="FPR budget (e.g., 0.01 for 1%%)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Training epochs")
    
    args = parser.parse_args()
    
    if args.quick and args.research:
        print("Cannot use both --quick and --research")
        sys.exit(1)
    
    success = smoke_test(
        quick=args.quick, 
        research=args.research,
        artifacts_dir=args.artifacts,
        device=args.device,
        n_train=args.train_samples,
        n_cal=args.cal_samples,
        n_test=args.test_samples,
        alpha=args.alpha,
        epochs=args.epochs
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
