#!/usr/bin/env python
"""Copy the submission with the best HOLDOUT macro F0.5 to submissions/best/."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# (submission dir, metrics file, key holding the holdout macro F0.5)
CANDIDATES = [
    ("submissions/step3_v3", "artifacts/models/verifier_v3/policy_eval.json",
     "winner_macro_f05"),
    ("submissions/step4_stage2", "artifacts/models/verifier_v4/metrics.json",
     "stage2_macro_f05"),
    ("submissions/step5_stage2_big", "artifacts/models/verifier_v5/metrics.json",
     "stage2_macro_f05"),
]


def main() -> int:
    scored = []
    for sub, metrics, key in CANDIDATES:
        sd, mf = ROOT / sub, ROOT / metrics
        if not (sd / "matching_results.tsv").exists() or not mf.exists():
            print(f"skip {sub} (missing submission or metrics)")
            continue
        val = float(json.loads(mf.read_text())[key])
        scored.append((val, sub))
        print(f"{sub:<32} holdout macro_f05 = {val:.4f}")
    if not scored:
        print("no scored submission found")
        return 1
    val, sub = max(scored)
    best = ROOT / "submissions/best"
    best.mkdir(parents=True, exist_ok=True)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy2(ROOT / sub / f, best / f)
    (best / "SOURCE.txt").write_text(f"{sub}\nholdout_macro_f05={val:.4f}\n")
    print(f"BEST: {sub} ({val:.4f}) -> submissions/best/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
