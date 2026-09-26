# Dependency License Manifest

Updated at every dependency change; re-verify at freeze time with `pip show <package>`.

| Package        | Version policy | License (verify at pin time) | Purpose                  | Phase |
|----------------|----------------|------------------------------|--------------------------|-------|
| PyYAML         | >=6.0,<7       | MIT                          | Config loading           | 2     |
| python-dotenv  | >=1.0,<2       | BSD-3-Clause                 | .env support             | 2     |
| pandas         | >=2.0,<3       | BSD-3-Clause                 | Tables                   | 2     |
| numpy          | >=1.26,<3      | BSD-3-Clause                 | Arrays                   | 2     |
| pyarrow        | >=14           | Apache-2.0                   | Parquet I/O              | 2     |
| psutil         | >=5.9          | BSD-3-Clause                 | Memory snapshots         | 2     |
| boto3          | >=1.28         | Apache-2.0                   | S3                       | 2     |
| sagemaker      | >=2.200        | Apache-2.0                   | Jobs (requirements-aws)  | 5     |
| scikit-learn   | >=1.3,<2       | BSD-3-Clause                 | TF-IDF, calibration      | 4+    |
| rapidfuzz      | >=3.6,<4       | MIT                          | String similarities      | 4     |
| lightgbm       | >=4.1,<5       | MIT                          | Primary GBDT             | 8     |
| pytest         | >=7.4          | MIT                          | Tests                    | 2     |

Banned: `python-Levenshtein` (GPL) — use `rapidfuzz`. Deferred pending license
verification: `jellyfish`. Deferred until measured gains: `faiss-cpu`, `torch`,
`sentence-transformers`.
