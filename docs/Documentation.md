# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Moggers
**Team Members:** Shabd Sharma, Krishna Goyal, [add teammates]
**Submission Date:** 27 Sep 2026

---

## 1. Executive Summary

A CPU-only, fully reproducible pipeline: six-channel TF-IDF / exact-key
retrieval over all Source 2+3 records, a LightGBM pair verifier on ~50
hand-built similarity features (including rule-based transliteration of nine
Indic scripts), a second LightGBM stage that re-scores each candidate using the
entity's confidently matched records, and a decision layer that enforces the
data's one-to-one structure (every S2/S3 record belongs to at most one S1
entity). Holdout macro F0.5: **[FINAL]** (public leaderboard **[FINAL]**).

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on the 2.2M-entity training set:

- **Cardinality:** only 5.6% of S1 entities are singletons; the mean is 3.46
  matches (up to 11). Recall therefore matters almost as much as precision.
- **Exclusivity:** all 7,638,365 matched S2/S3 ids in the ground truth are
  distinct — no pool record matches two S1 entities. 73–75% of pool records
  match something; the rest are distractors.
- **Distractors are near-duplicates:** e.g. "Shivam Energy General Pvt Ltd,
  Flat No. 6052/**6**" vs the true "Shivam Energy Pvt Ltd, 6052/**5**";
  "People Services" vs "People Solution" at 12/2/2/**10** vs 12/2/2/**1**.
  Address numbers are the decisive signal.
- **Hard positives:** random replacement names ("Dovacira", "Onyxonyx"),
  names in Devanagari / Tamil / Telugu / Kannada / Malayalam / Bengali / Odia
  script, URL-only names ("orthopedicsafehealth.com"), empty addresses,
  truncated addresses, word-order shuffles and character-level typos.
- **Test shift:** 15% of test S1 entities are French (absent from training).
  French function words "de"/"la" collided with the US state codes DE/LA in
  our state parser, creating spurious `state_match` evidence and French
  over-matching (3.91 emitted ids/entity vs ~3.46 expected) — fixed (§4).

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage GBDT classifier + structured decision.
**Core Innovation:** (1) pool-side exclusivity enforced at decision time over
all 245M test pairs; (2) stage-2 group features that link hard positives
(garbage names) to the entity's confident matches through shared addresses;
(3) script-agnostic consonant skeletons for cross-script name matching.

---

## 3. Candidate Generation (Blocking)

- **Normalization:** NFKC + casefold, Latin diacritic folding (Indic marks
  kept), URL extraction, legal-suffix stop list (English, Indic, European),
  ordinal splitting ("41St" → "41 st"), address stop tokens, house number /
  postal / digit views, name aliases (DBA / fka segments) as extra documents.
- **Channels** (top-k per S1 entity, union, min-rank dedup per pair):
  `exact` normalized-key postings (k=100), `char_name` char n-gram TF-IDF (k=50),
  `word_name` word TF-IDF (k=30), `rare` rare-token channel (k=30),
  `word_addr` word TF-IDF on the address core (k=50), `char_addr` char TF-IDF
  on addresses (k=30). The address channels recover renamed / cross-script
  records. Pool statistics are label-free.
- **Candidate pairs generated:** 245.5M for the 1.73M test entities
  (~142 per entity); 43.3M for the 300k-entity training sample.
- **Recall:** 96.7% of true pairs in the training sample are retrieved; a
  perfect verifier on these candidates would score macro F0.5 = 0.9886.
- `candidate_pairs.tsv` lists the top-[K] stage-2 candidates per entity — the
  exact set the final model scores.

---

## 4. Matching Model

**Features used (stage 1, 49):**
- *Name:* exact / core-exact, token-sort, token-set, Jaro-Winkler, length
  ratio; consonant-skeleton token-set and ratio after romanization; script
  mismatch flag; glued (space-free) ratio and partial ratio for URL-style names.
- *Address:* token-sort/set, exact, length ratio, skeleton token-set; house
  number match; digit containment; **address-number set Jaccard / subset /
  equality** (zero-padding normalized); alphabetic-token Jaccard; state match
  (US/India only; ambiguous two-letter words such as "de", "la", "in", "or"
  are ignored mid-address); country match.
- *Retrieval provenance:* per-channel ranks and scores, number of channels.
- *Within-entity context:* rank by best retrieval score and by address score,
  candidate count, top-1 score, margin and relative score.

**Stage 2 (group features, top-[K] per entity):** out-of-fold stage-1
probability, rank in group, number of anchors (other candidates with
p ≥ 0.5), max / sum of other probabilities, exact-address equality with an
anchor, max address similarity to an anchor, address-number equality with an
anchor, max name-skeleton similarity to an anchor, share of anchors agreeing on
one address.

**Model type:** LightGBM binary classifiers (stage 1: 1500 trees, 96 leaves;
3 out-of-fold stage-1 models for stage 2; stage 2: 800 trees).
**Threshold selection method:** macro F0.5 on a holdout of 20% of S1 entities
(index % 5 == 0), searching threshold and Bayes expected-F0.5 policies (the
latter computes the F0.5-optimal prefix per entity from the pair probabilities
with an exact Poisson-binomial model) × exclusivity mode (none / hard / soft).
At test time "hard" exclusivity keeps each S2/S3 record only on its best S1.

---

## 5. Results & Error Analysis

| Version | Holdout macro F0.5 | Public LB |
|---|---|---|
| v1 verifier, threshold 0.7 | 0.9325 | 0.907 |
| + pool-side exclusivity | 0.9329 (holdout understates it) | 0.923 |
| v3: state fix + transliteration + address-number features | 0.9537 | [FILL] |
| v4: stage-2 group re-scorer | [FILL] | [FILL] |

Holdout exclusivity gains are a lower bound: only holdout entities compete
there, while on test every entity competes (it removed ~265k double claims).

- **Common false positives:** near-duplicate distractors that share the name
  and differ in one address number; French entities built from a small
  vocabulary ("Amicale Sport SARL") in dense cities.
- **Common false negatives:** random replacement names with truncated
  addresses; cross-script names whose romanization diverges (soft "g" written
  as "j"); URL-only names with empty addresses.

---

## 6. Conclusion

Most of the gain came from respecting the data's structure (one-to-one pool
assignment, entity-level context) and from features aimed at the observed
noise (address-number perturbations, Indic scripts, URL names) rather than
from model capacity. The main remaining headroom is candidate recall and a
learned cross-encoder for hard textual pairs.

---

## Appendix

### A. Code Artefacts

See `code/business_entity_resolution/README.md` for exact commands. Entry
points: `scripts/generate_candidates.py`, `scripts/build_features.py`,
`scripts/train.py`, `scripts/eval_decision.py`, `scripts/stage2.py`,
`scripts/predict_test.py`.

### B. Additional Results

Per-country emitted ids per entity (label-free check, final submission):
[FILL from `scripts/diag_by_country.py`].
