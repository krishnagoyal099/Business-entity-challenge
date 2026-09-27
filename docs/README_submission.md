# Business Entity Resolution — reproduction guide

Pipeline: normalize → multi-channel candidate retrieval → pair features →
LightGBM verifier (stage 1) → group-aware re-scorer (stage 2) → one-to-one
decision policy → `output/matching_results.tsv` + `output/candidate_pairs.tsv`.

All models are LightGBM (MIT license) gradient-boosted trees; no pretrained or
external models, no external data. Only the provided train/test TSVs are used.

## Environment

- Python 3.11+, CPU only. Developed on AWS SageMaker `ml.c7i.24xlarge`
  (96 vCPU, 192 GB RAM). Peak RSS per step stays under ~45 GB.
- `pip install -r requirements.txt`

Data layout (as shipped by the organisers):

```
dataset/train/train_source{1,2,3}.tsv, dataset/train/train_ground_truth.tsv
dataset/test/test_source{1,2,3}.tsv
```

## End-to-end commands

`CFG=configs/aws_cpu.yaml`. Every stage is resumable and cached by a hash of
its code + config + inputs; re-running a finished stage is a no-op.

```bash
# 1. normalization (runs inside) + candidate retrieval
#    train uses a 300k S1-entity sample (all 10.3M pool records are searched)
python scripts/generate_candidates.py --config $CFG --split train --limit-rows 300000
python scripts/generate_candidates.py --config $CFG --split test

# 2. pair features (4 workers keeps RAM < 30 GB)
python scripts/build_features.py --config $CFG --split train --limit-rows 300000 --n-jobs 4
python scripts/build_features.py --config $CFG --split test --n-jobs 4

# 3a. stage-1 verifier + policy (holdout = S1 entities with index % 5 == 0)
python scripts/train.py --config $CFG --n-estimators 1500 --model-dir models/verifier_v3
python scripts/eval_decision.py --config $CFG --model artifacts/models/verifier_v3/model.txt

# 3b. stage-2 group re-scorer (out-of-fold stage-1 probs, 3 folds)
python scripts/stage2.py train --config $CFG --n-jobs 8

# 4. test inference -> output/
python scripts/stage2.py predict --config $CFG --n-jobs 8
python scripts/validate_submission.py --config $CFG
```

Stage-1-only submission (fallback): replace step 4 with
`python scripts/predict_test.py --config $CFG --model artifacts/models/verifier_v3/model.txt --exclusive hard`.

Approximate wall times on ml.c7i.24xlarge: candidates (test) ~1 h,
features ~30 min, stage-1 training ~15 min, stage 2 ~45 min train /
~30 min predict.

## Code map

| Path | Role |
|---|---|
| `src/normalization.py` | casefold/NFKC/diacritics, URL extraction, legal-suffix stripping, address views, US/India state parsing |
| `src/transliterate.py` | rule-based romanization of 9 Indic scripts + consonant skeletons |
| `src/blocking.py`, `src/candidate_generation.py` | normalized parquet views; 6 retrieval channels (exact keys, char/word TF-IDF on names and addresses, rare-token) |
| `src/pair_features.py` | per-pair similarity, address-number, state, cross-script and within-entity context features |
| `src/model.py` | LightGBM training / persistence (feature order saved in `features.json`) |
| `src/group_features.py` | stage-2 features relating each candidate to the entity's confident matches |
| `src/entity_decision.py` | pool-side exclusivity + threshold / Bayes expected-F0.5 policies |
| `src/evaluator.py` | macro F0.5 exactly as specified (empty/empty = 1.0) |
| `scripts/diag_by_country.py` | label-free per-country sanity check of a submission |
| `tests/` | pytest suite (`python -m pytest -q`) |
