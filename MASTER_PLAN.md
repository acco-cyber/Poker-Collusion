# Master Plan: Rank 10 by 2026-09-20

## Target
**Rank 10 score: 0.91160** (held by "thisray" as of 2026-09-17)

## Current State (2026-09-18)
- User's best: 0.83065 (rank ~89)
- Gap to rank 10: +0.081
- Submissions remaining: 5 today + 5 day 3 + 5 day 4 = 15 total
- Deadline: 2026-09-20 22:00 UTC

## Composite Metric
```
composite = 0.70 × pair_ap + 0.20 × evidence_map5 + 0.10 × behavior_map
```

Back-calculated public values (assuming behavior_map≈0.90):
- pair_ap ≈ 0.95 (user's local: 0.9993)
- evidence_map5 ≈ 0.378 (user's local: 0.5454)
- behavior_map ≈ 0.90 (user's local: 1.0)

## Top of Leaderboard (2026-09-17)

| Rank | Team | Score |
|------|------|-------|
| 1 | Pardheev Krishna | 0.93827 |
| 2 | Tejasv Bhatia | 0.93482 |
| 3 | blastyy | 0.93468 |
| 4 | John Tyler | 0.93395 |
| 5 | [Deleted] | 0.93048 |
| 6 | Marc Donovici | 0.92509 |
| 7 | Leo | 0.92217 |
| 8 | seantangth | 0.91930 |
| 9 | Mohib | 0.91224 |
| 10 | **thisray** | **0.91160** |

## Day 1 Lessons (2026-09-17)

| Attempt | Score | Lesson |
|---------|-------|--------|
| User's baseline | 0.83065 | Strong starting point — PU-aware LGBMRanker |
| v1 (seats-only) | 0.07339 | Floor — no action features = no signal |
| v3 (3-seed ensemble) | 0.07241 | Confirms floor |
| Submission A (all chip-transfer) | 0.75238 | Replacing all evidence fails |
| Submission B (behavior-specific) | 0.74941 | Same — full replacement hurts |
| **Hybrid 3+2 (user top-3 + chip top-2)** | **0.81486** | Slight drop, but hybrid is sound |

## Day 2 Plan (2026-09-18, 5 submissions)

Generate 5 submission CSVs from `kaggle_notebook_day2.py`:

1. `submission_01_user_baseline.csv` — Sanity check (should be 0.83065)
2. `submission_02_hybrid_3plus2.csv` — Yesterday's hybrid (0.81486)
3. `submission_03_hybrid_4plus1.csv` — More conservative hybrid
4. `submission_04_hybrid_2plus3.csv` — More aggressive hybrid
5. `submission_05_passive_soft_play.csv` — Targeted fix for soft_play evidence

**Recommended order**: Submit 1, 3, 5, 2 (skip 4 unless others improve)

## Day 3 Plan (2026-09-19, 5 submissions)

Based on Day 2 results, try:
1. **Best variant from Day 2** as starting point
2. **Graph-based features**: player interaction network centrality
3. **Multi-seed bagging**: train 3 LGBMRanker models with different seeds, average evidence
4. **Isotonic calibration** of risk_score
5. **PU weight tuning**: try `pu_unknown_weight` = 0.02, 0.06, 0.08

## Day 4 Plan (2026-09-20, last day before 22:00 UTC deadline)

1. **Final submission** with best approach from Day 3
2. **Hold 2 submissions** for emergency fixes
3. Submit best variant 2 hours before deadline

## What NOT to Do

❌ Don't replace user's evidence with heuristics (proven to fail in Day 1)
❌ Don't use rank-percentile transforms (mathematically equivalent for AP)
❌ Don't submit my v3 baseline alone (scored 0.07)
❌ Don't waste submissions on minor tweaks

## Key Insight

The user's notebook achieves (public):
- pair_ap ≈ 0.95 (70% weight → 0.665 to composite)
- evidence_map5 ≈ 0.378 (20% weight → 0.076 to composite)
- behavior_map ≈ 0.90 (10% weight → 0.090 to composite)
- Total: ~0.831 ✓

To reach 0.91160 (rank 10), the highest-leverage moves are:
1. Improve pair_ap from 0.95 → 0.97 (+0.014 to composite)
2. Improve evidence_map5 from 0.378 → 0.50 (+0.024 to composite)
3. Improve behavior_map from 0.90 → 0.95 (+0.005 to composite)

The first (pair_ap) is hardest — requires fundamentally better model.
The second (evidence_map5) is what Day 1 hybrid tried — slight drop.
The third (behavior_map) is easiest — try different behavior thresholds.

## Expected Outcome

- **Realistic best**: 0.85-0.87 (improvement of +0.02 to +0.04)
- **Stretch goal**: 0.88-0.90 (top 30)
- **Rank 10 (0.91+)**: ~5-10% probability with 14 remaining submissions

The user's existing pipeline is strong; pushing it to rank 10 requires either:
(a) A fundamentally new feature or signal we haven't found
(b) A lucky submission that hits the right test-set distribution

Good luck!
