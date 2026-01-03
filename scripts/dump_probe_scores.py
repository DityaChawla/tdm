import json
import random
from pathlib import Path

import numpy as np

from tdm.config import TDMConfig
from tdm.probe import ProbeScorer
from tdm.instrumentation import ModelWithActivations
from tdm.datasets.synthetic_sleeper import SyntheticSleeperDataset


def main():
    out_path = Path("./artifacts_paper/eval_dump_scores.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    cfg = TDMConfig(
        model_name="gpt2",
        device="cpu",
        artifacts_dir="./artifacts_paper",
        alpha=0.01,
    )

    # Load model + probe artifacts
    mwa = ModelWithActivations(cfg.model_name, device=cfg.device)
    scorer = ProbeScorer.load(cfg.artifacts_dir, device=cfg.device)

    # Generate a score-only dataset (no training)
    ds = SyntheticSleeperDataset(seed=seed)
    prompts, labels = ds.generate(n_clean=2000, n_triggered=2000)  # adjust if slow

    rows = []
    for p, y in zip(prompts, labels):
        # Probe score only (whitebox)
        s = scorer.score_prompt(mwa, p)
        rows.append({"prompt": p, "label": int(y), "probe": float(s)})

    json.dump({"seed": seed, "rows": rows}, open(out_path, "w"))
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()

