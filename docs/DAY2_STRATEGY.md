# Day 2 Strategy (2026-09-18)

## Goal
Find the best evidence-selection strategy by trying 5 variants.

## Approach
Keep user's risk_score and predicted_behavior unchanged (proven signals).
Vary only the 5 evidence_hand_* columns across 5 strategies.

## The 5 Strategies

### 1. User baseline (sanity check)
- File: `submission_01_user_baseline.csv`
- Strategy: User's original 0.83065 submission unchanged
- Expected: 0.83065 (sanity check)
- Why: Verify the baseline still scores the same

### 2. Hybrid 3+2 (yesterday's attempt)
- File: `submission_02_hybrid_3plus2.csv`
- Strategy: User's top-3 hands + chip-transfer top-2 (excl. duplicates)
- Expected: ~0.81 (yesterday scored 0.81486)
- Why: Reproduce yesterday's result; baseline for hybrid variants

### 3. Hybrid 4+1 (conservative)
- File: `submission_03_hybrid_4plus1.csv`
- Strategy: User's top-4 hands + chip-transfer top-1 (excl. duplicates)
- Expected: 0.82-0.83 (minimal change from user's)
- Why: If user's hands 1-4 are strong, replacing only hand 5 minimizes risk

### 4. Hybrid 2+3 (aggressive)
- File: `submission_04_hybrid_2plus3.csv`
- Strategy: User's top-2 hands + chip-transfer top-3 (excl. duplicates)
- Expected: 0.78-0.81 (more change from user's)
- Why: If chip hands have signal, more is better. Last resort if 3+2 helped.

### 5. Passive for soft_play only
- File: `submission_05_passive_soft_play.csv`
- Strategy: For soft_play pairs only, replace all 5 evidence with passive signal top-5
  For other pairs, keep user's original evidence unchanged
- Expected: 0.82-0.83 (targeted fix)
- Why: User's evidence_map5 might be weakest for soft_play (passive hands are subtle);
  replacing only those might help without hurting directed_transfer / coordinated_isolation

## Recommended Submission Order

With 5 submissions available, submit in this order:

1. **submission_01_user_baseline.csv** (sanity check — confirm 0.83065)
2. **submission_03_hybrid_4plus1.csv** (most conservative hybrid)
3. **submission_05_passive_soft_play.csv** (targeted fix)
4. **submission_02_hybrid_3plus2.csv** (yesterday's hybrid, for comparison)
5. **Save 1 submission for Day 3** (don't burn all 5 today)

## Decision Tree

After each submission, decide:
- If score > 0.83065: try more variants in the same direction
- If score < 0.83065: pivot to a different strategy
- If score = 0.83065: baseline confirmed, focus on other dimensions

## What This Tells Us

The 5 strategies are all variations on "how much of the user's evidence to keep":
- 5 (all user) → baseline
- 4 (user + 1 chip) → minimal change
- 3 (user + 2 chip) → yesterday's hybrid
- 2 (user + 3 chip) → aggressive
- 0 (all chip, only soft_play) → targeted fix

By varying one parameter (n_user_hands), we can plot a curve:
- If 5 > 4 > 3 > 2: user's evidence is consistently better, don't replace
- If 2 > 3 > 4 > 5: chip evidence is consistently better, replace more
- If peak at 3 or 4: sweet spot exists, hybrid is the answer
