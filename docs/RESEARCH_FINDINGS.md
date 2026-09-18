# Research Findings: Gold Evidence Pattern Analysis

## Methodology

Deep research on 2026-09-18 to find a realistic path to 0.90.

### Data Sources Used
1. **Competition data** (8 files, 276MB) — seats, hands, actions, players, dev labels, dev evidence, eval pairs, sample submission
2. **User's notebook diagnostics ZIP** (32MB) — contains `portfolio_outer_evidence.json` with `submitted_hands` and `gold_hands` for all 13,210 dev pairs

### Key Discovery: User's Diagnostics Has Gold Hand Info

The user's notebook diagnostics (`poker7_05_gated_consensus_DIAGNOSTICS_DO_NOT_SUBMIT.zip`) contains:
- `portfolio_outer_evidence.json`: 13,210 entries with `submitted_hands` (user's 5 picks), `gold_hands` (5 ground truth), and `ap5` per pair
- `outer_predictions.csv`: User's risk_score and predicted_behavior for dev pairs
- `validation_report.json`: User's local metric breakdown

This allowed us to:
1. Compute the user's TRUE local AP@5 (0.5454) per pair
2. Find pairs where the user got LOW AP@5
3. Identify the 55 gold hands the user missed
4. Train a classifier to predict "is this hand gold?" from features

## Critical Findings

### Finding 1: User's Local AP@5 Distribution (n=94 confirmed positives)
- Mean: 0.5454
- Median: 0.5467
- Perfect (AP5=1.0): 5 pairs
- AP5 >= 0.5: 56 pairs
- AP5 < 0.3: 17 pairs (low confidence)
- AP5 < 0.1: 3 pairs

### Finding 2: User Got 64.5% of Gold Hands Right
- Total gold hands (94 confirmed-positive pairs × ~5): 456
- Hands user got right: 294 (64.5%)
- Hands user missed: 55

### Finding 3: Gold Hands Have Distinct Features
| Feature | Gold hands (mean) | All dev hands (mean) | Difference |
|---------|-------------------|----------------------|------------|
| Final pot (BB) | 184 | 113 | +62% |
| Players at showdown | 0.63 | 0.32 | 2x higher |
| Big blind | 3.2 | 3.5 | -9% |
| Players dealt | 6.0 | 6.0 | same |

**Gold hands have BIGGER POTS and reach SHOWDOWN MORE OFTEN.**

### Finding 4: AP@5 by Behavior Family
| Family | Mean AP@5 | Median | Count |
|--------|-----------|--------|-------|
| coordinated_isolation | 0.464 | 0.513 | 24 |
| directed_transfer | 0.558 | 0.543 | 39 |
| soft_play | 0.592 | 0.639 | 31 |

User is WEAKEST at coordinated_isolation evidence selection.

### Finding 5: Gold Fraction is Small
- Gold hands per positive pair: 5 out of ~193 shared hands = 2.86%
- This is a hard retrieval problem — needle in a haystack

## Approaches Tried

### Approach 1: Gold-Pattern ML Classifier (NEW today)
**Method**: Train LightGBM on (pair, hand) features to predict P(gold).
- Features: 23 hand-level + pair-hand features (pot_bb, asymmetry, transfer_score, etc.)
- Training: 94 positive pairs × ~193 hands = 16,842 examples (456 positive, 16,386 negative)
- CV AP: 0.2161 (weak but real signal)

**Results**:
| Variant | Score | vs Baseline |
|---------|-------|-------------|
| v4 (4+1, ML pick) | 0.82258 | -0.008 |
| v2 (3+2, ML picks) | 0.81314 | -0.018 |
| v3 (2+3, ML picks) | 0.79831 | -0.032 |

### Approach 2: Chip-Transfer Heuristic (Day 1-2)
**Method**: Model-free scoring of (pair, hand) by chip transfer asymmetry.

**Results** (yesterday):
| Variant | Score | vs Baseline |
|---------|-------|-------------|
| Conservative 4+1 | 0.82371 | -0.007 |
| Hybrid 3+2 | 0.81486 | -0.016 |
| All 5 chip-transfer | 0.75238 | -0.078 |

### Approach 3: Behavior-Specific Evidence (Day 1)
**Results**:
| Variant | Score | vs Baseline |
|---------|-------|-------------|
| Passive for soft_play | 0.79641 | -0.034 |
| Behavior-specific (all) | 0.74941 | -0.081 |

## Conclusions

### What We Learned
1. **The user's LGBMRanker is genuinely strong** — all evidence replacement attempts scored BELOW the baseline
2. **The simple chip-transfer heuristic BEATS the trained ML classifier** — 0.82371 vs 0.82258 (4+1 variant)
3. **More aggressive replacement = worse score** — the trend is clear:
   - 4+1 (1 replacement): -0.007 to -0.008
   - 3+2 (2 replacements): -0.016 to -0.018
   - 2+3 (3 replacements): -0.032
4. **The path to 0.90 is NOT in evidence selection** — every evidence modification hurts

### Why the ML Classifier Failed to Beat the User's Model
1. **Training data was small** — only 94 positive pairs with confirmed labels
2. **The user's notebook already uses a similar approach** — their LGBMRanker with lambdarank objective does this better
3. **Features were limited** — we used only seats+hands features, no action-level data (the user has actions)
4. **CV AP was only 0.22** — too weak to add value over the proven baseline

### Realistic Path to 0.90 (Honest Assessment)
Based on extensive research, reaching 0.90 requires:
1. **A fundamentally better pair_ap model** (70% weight) — needs new features
2. **Access to action-level training** — the user's notebook has this; my sandbox doesn't
3. **Graph-based features** (player interaction network) — unexplored
4. **More training data** — would require synthetic generation or different sampling

**Probability of reaching 0.90 in 1 submission: < 5%**

The user's existing 0.83065 is already a strong result. The gap to 0.90 (rank 10) requires either:
- A breakthrough feature we haven't found
- Significantly more model capacity
- Access to the user's full action-level pipeline

## What's Saved in this Repo

- `scripts/research_gold_patterns.py` — Step 1: Load and analyze gold evidence
- `scripts/research_step2.py` — Step 2: User's dev predictions analysis
- `scripts/research_step3.py` — Step 3: Train gold-pattern classifier
- `scripts/research_step4.py` — Step 4: Apply model to eval pairs
- `submissions/sub_v1_pure_model_sample.csv` — Pure ML picks (sample)
- `submissions/sub_v2_conservative_3plus2_sample.csv` — User 3 + ML 2 (sample)
- `submissions/sub_v3_moderate_2plus3_sample.csv` — User 2 + ML 3 (sample)
- `submissions/sub_v4_conservative_4plus1_sample.csv` — User 4 + ML 1 (sample)
- `models/gold_pattern_model.pkl` — Trained LightGBM classifier

## Recommendations for Future Attempts

1. **Stop trying to replace evidence** — it doesn't work
2. **Focus on pair_ap** (70% weight) — needs new features
3. **Use the user's notebook AS-IS** as the baseline (0.83065 is solid)
4. **If you have Kaggle access**: run the user's notebook with adjusted `pu_unknown_weight` (0.02, 0.06, 0.08)
5. **Try graph-based features**: player-pair interaction network, centrality
6. **Try ensemble**: average risk_score from user's notebook + a different model

## Final Honest Statement

Despite extensive research (3 scripts, 4 submission variants, trained ML classifier),
I could NOT beat the user's baseline of 0.83065. The best hybrid scored 0.82371 (-0.007).

The user's PU-aware LGBMRanker is well-optimized. Beating it requires either:
- Access to action-level features (which I don't have in 4GB RAM)
- A fundamentally different approach (graph features, sequence models)
- Or just lucky submission timing

I apologize for not achieving 0.90 — it's genuinely hard from this sandbox.
