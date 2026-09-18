# Day 2 Submission Results (2026-09-18)

## Today's 2 Submissions

| # | Time | Submission | Score | vs Baseline | Strategy |
|---|------|------------|-------|-------------|----------|
| 1 | 03:45 | sub_01_conservative_4plus1.csv | **0.82371** | -0.007 | User top-4 + chip top-1 (80.6% overlap) |
| 2 | 03:45 | sub_02_targeted_soft_play.csv | **0.79641** | -0.034 | Replace soft_play evidence with passive top-5 (91.8% overlap) |

## Baseline
- User's existing notebook: **0.83065** (PU-aware LGBMRanker)

## Updated Leaderboard (2026-09-18)

| Rank | Team | Score |
|------|------|-------|
| 1 | Pardheev Krishna | 0.93827 |
| 2 | Tejasv Bhatia | 0.93513 |
| 3 | blastyy | 0.93487 |
| 4 | John Tyler | 0.93395 |
| 5 | [Deleted] | 0.93048 |
| 6 | Marc Donovici | 0.92509 |
| 7 | Leo | 0.92217 |
| 8 | seantangth | 0.91930 |
| 9 | Exposed | 0.91820 |
| 10 | Alex Li | 0.91506 |
| 11 | thisray | 0.91479 |
| 12 | Amin Mohamed | 0.91241 |

**Rank 10 target: 0.91506** (was 0.91160 yesterday; rank 10 has gotten harder)

## Cumulative Results (Day 1 + Day 2)

| # | Date | Submission | Score | vs Baseline |
|---|------|-------------|-------|-------------|
| 1 | 09-17 | v1 (seats-only) | 0.07339 | -0.757 |
| 2 | 09-17 | v3 (3-seed ensemble) | 0.07241 | -0.758 |
| 3 | 09-17 | Sub A (all chip-transfer) | 0.75238 | -0.078 |
| 4 | 09-17 | Sub B (behavior-specific) | 0.74941 | -0.081 |
| 5 | 09-17 | Hybrid 3+2 | 0.81486 | -0.016 |
| 6 | **09-18** | **Conservative 4+1** | **0.82371** | **-0.007** ← BEST hybrid |
| 7 | 09-18 | Targeted soft_play | 0.79641 | -0.034 |

## Key Insights from Day 2

### Conservative 4+1 (0.82371)
- **Only 0.007 below baseline** — much better than 3+2 hybrid (0.016 below)
- Replacing only hand 5 (lowest-confidence position) minimizes damage
- Suggests the user's LGBMRanker evidence is consistently better than chip-transfer heuristics

### Targeted soft_play (0.79641)
- Replacing evidence only for soft_play pairs dropped score by 0.034
- This means the user's model is BETTER at picking soft_play evidence than the passive-signal heuristic
- Disproves the hypothesis that "soft_play is the user's weakest dimension"

## What This Tells Us

The user's PU-aware LGBMRanker is **genuinely strong** across all 3 behavior families. Model-free heuristics (chip-transfer, passive, aggressive signals) cannot replace it — even at the lowest-confidence positions.

**The path to rank 10 is NOT in evidence selection.** It must be in:
1. **pair_ap improvement** (70% weight, hardest)
2. **behavior_map improvement** (10% weight, easier via threshold tuning)
3. A fundamentally different model architecture

## Day 3 Plan (2026-09-19, 5 submissions)

Since evidence replacement keeps failing, Day 3 should try DIFFERENT approaches:

1. **Resubmit user's baseline** (sanity check, confirm 0.83065)
2. **Threshold tuning**: try different behavior thresholds (e.g., 0.3, 0.5, 0.7) on user's risk_score
3. **Isotonic calibration** of risk_score (preserves ranking, might improve metric implementation)
4. **Submit user's submission UNCHANGED** but with different descriptions to check if scoring is deterministic
5. **Save 1 submission for Day 4** (last day before deadline)

## What NOT to Try

❌ Don't replace evidence hands with heuristics (failed 5 times)
❌ Don't replace risk_score with my own model (failed with 0.07)
❌ Don't try rank-percentile transforms (mathematically equivalent for AP)
