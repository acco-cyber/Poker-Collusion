# Day 1 Submission Results (2026-09-17)

## All Day 1 Attempts

| # | Time | Submission | Score | vs Baseline | Notes |
|---|------|------------|-------|-------------|-------|
| 1 | 06:04 | v1 baseline (seats-only) | 0.07339 | -0.757 | Floor — no action features |
| 2 | 06:10 | v3 (3-seed ensemble, percentile risk) | 0.07241 | -0.758 | Floor — confirms v1 |
| 3 | 08:53 | Submission A (chip-transfer evidence, all 5) | 0.75238 | -0.078 | Replacing all evidence fails |
| 4 | 08:53 | Submission B (behavior-specific evidence) | 0.74941 | -0.081 | Same — full replacement hurts |
| 5 | 18:43 | **Hybrid (user top-3 + chip top-2)** | **0.81486** | **-0.016** | Slight drop, closest to baseline |

## Baseline
- **User's notebook**: 0.83065 (PU-aware LGBMRanker, 197KB pipeline)

## Key Lessons

### What works:
- ✅ User's PU-aware LGBMRanker (0.83065) — strong baseline
- ✅ Hybrid approach (0.81486) — much closer to baseline than full replacement

### What doesn't work:
- ❌ Replacing all evidence with heuristics (-0.08)
- ❌ Seats-only features without action data (-0.75)
- ❌ Rank-percentile transforms (mathematically equivalent to identity for AP)

### Key insight:
The user's evidence ranker is **genuinely strong** — replacing ANY of its hands
with model-free heuristics drops the score. To beat 0.83, we need:
1. **Better pair_ap** (70% weight) — most impactful but hardest
2. **Better behavior_map** (10% weight) — easiest via threshold tuning
3. **Better evidence_map5** (20% weight) — requires model improvement
