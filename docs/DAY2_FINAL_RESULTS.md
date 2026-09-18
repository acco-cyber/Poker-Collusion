# Day 2 Final Results (2026-09-18)

## Today's 5 Submissions (all used)

| # | Time | Submission | Score | vs Baseline | Approach |
|---|------|------------|-------|-------------|----------|
| 1 | 03:45 | sub_01_conservative_4plus1 (chip-transfer) | **0.82371** | -0.007 | Heuristic 4+1 |
| 2 | 03:45 | sub_02_targeted_soft_play (chip-transfer) | 0.79641 | -0.034 | Heuristic targeted |
| 3 | 04:30 | sub_v2_conservative_3plus2 (ML gold-pattern) | 0.81314 | -0.018 | ML 3+2 |
| 4 | 04:31 | sub_v3_moderate_2plus3 (ML gold-pattern) | 0.79831 | -0.032 | ML 2+3 |
| 5 | 04:35 | sub_v4_conservative_4plus1 (ML gold-pattern) | 0.82258 | -0.008 | ML 4+1 |

## Baseline
- User's existing notebook: **0.83065** (still the best)

## Cumulative Results (Day 1 + Day 2 — 9 total attempts)

| # | Date | Approach | Score | vs Baseline |
|---|------|----------|-------|-------------|
| 1 | 09-17 | v1 (seats-only) | 0.07339 | -0.757 |
| 2 | 09-17 | v3 (3-seed ensemble) | 0.07241 | -0.758 |
| 3 | 09-17 | Sub A (all chip-transfer) | 0.75238 | -0.078 |
| 4 | 09-17 | Sub B (behavior-specific) | 0.74941 | -0.081 |
| 5 | 09-17 | Hybrid 3+2 (chip-transfer) | 0.81486 | -0.016 |
| 6 | 09-18 | Conservative 4+1 (chip-transfer) | **0.82371** | -0.007 ← BEST hybrid |
| 7 | 09-18 | Targeted soft_play (chip-transfer) | 0.79641 | -0.034 |
| 8 | 09-18 | ML v2 3+2 (gold-pattern) | 0.81314 | -0.018 |
| 9 | 09-18 | ML v3 2+3 (gold-pattern) | 0.79831 | -0.032 |
| 10 | 09-18 | ML v4 4+1 (gold-pattern) | 0.82258 | -0.008 |

## Key Findings

### Trend: More replacement = worse score
- 1 hand replaced (4+1): -0.007 to -0.008
- 2 hands replaced (3+2): -0.016 to -0.018
- 3 hands replaced (2+3): -0.032
- 5 hands replaced (all): -0.078 to -0.081

### Chip-transfer heuristic SLIGHTLY beats ML gold-pattern classifier
- Heuristic 4+1: 0.82371
- ML 4+1: 0.82258
- Difference: 0.001 (negligible)

### Both lose to user's baseline
- User's notebook: 0.83065
- Best hybrid: 0.82371 (-0.007)

## Conclusion

**Could NOT reach 0.90 in 5 submissions.** Best hybrid scored 0.82371 (still below baseline).

The user's LGBMRanker is genuinely strong. Beating it requires:
1. Access to action-level features (not available in 4GB RAM sandbox)
2. A fundamentally new approach (graph features, sequence models)
3. Or significantly more model capacity

## What's Next (Day 3, 2026-09-19)

With 5 fresh submissions:
1. **Submit user's baseline unchanged** (sanity check, expect 0.83065)
2. **Try threshold tuning** on user's risk_score (preserves ranking)
3. **Try ensemble** user's risk_score + my v3 model's risk_score
4. **Save 2 submissions** for emergency

**Realistic target**: 0.85 (rank ~70)
**Stretch target**: 0.88 (rank ~30)
**Rank 10 (0.91+)**: < 5% probability
