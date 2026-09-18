"""
RESEARCH STEP 2: Deep analysis of user's evidence predictions on dev set.

Critical insight: The user's notebook diagnostics has 'portfolio_outer_evidence.json'
with submitted_hands AND gold_hands for each dev pair. We can see EXACTLY where
the user's LGBMRanker fails and check if our chip-transfer heuristic could help.

Plan:
1. Extract all 13,210 entries from portfolio_outer_evidence.json
2. For each positive pair (where gold exists), compute AP@5
3. Find pairs where user scored LOW (e.g., AP@5 < 0.3)
4. For those pairs, check what gold hands the user MISSED
5. Look at the missed gold hands' characteristics — are they high chip-transfer?
6. If yes, build a hybrid that uses chip-transfer for low-confidence pairs
"""
import json
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

DATA = Path("/home/z/my-project/kaggle/comp-data")
USER_SUB = Path("/home/z/my-project/kaggle/user-submissions/pu-aware-output/submission.csv")
DIAG = Path("/home/z/my-project/kaggle/user-submissions/pu-aware-output/poker7_05_gated_consensus_DIAGNOSTICS_DO_NOT_SUBMIT.zip")

print("="*80)
print("RESEARCH STEP 2: Analyze user's evidence on dev set")
print("="*80)

# Extract portfolio_outer_evidence.json
print("\n[1] Extracting user's dev evidence predictions...")
with zipfile.ZipFile(DIAG, 'r') as z:
    with z.open('portfolio_outer_evidence.json') as f:
        outer_ev = json.load(f)

print(f"  Total entries: {len(outer_ev):,}")

# Each entry has: pair_id, submitted_hands (5), anchor_hands (5), gold_hands (5 if positive), ap5
# Convert to DataFrame
rows = []
for entry in outer_ev:
    row = {
        'pair_id': entry['pair_id'],
        'submitted_hand_1': entry['submitted_hands'][0] if len(entry['submitted_hands']) > 0 else None,
        'submitted_hand_2': entry['submitted_hands'][1] if len(entry['submitted_hands']) > 1 else None,
        'submitted_hand_3': entry['submitted_hands'][2] if len(entry['submitted_hands']) > 2 else None,
        'submitted_hand_4': entry['submitted_hands'][3] if len(entry['submitted_hands']) > 3 else None,
        'submitted_hand_5': entry['submitted_hands'][4] if len(entry['submitted_hands']) > 4 else None,
        'ap5': entry.get('ap5', None),
    }
    if 'gold_hands' in entry and entry['gold_hands']:
        for k, h in enumerate(entry['gold_hands'][:5]):
            row[f'gold_hand_{k+1}'] = h
    rows.append(row)

dev_ev_df = pd.DataFrame(rows)
print(f"  DataFrame: {dev_ev_df.shape}")

# How many have gold_hands (positive pairs)?
n_with_gold = dev_ev_df.gold_hand_1.notna().sum()
print(f"  Pairs with gold: {n_with_gold}")
print(f"  Pairs without gold (negatives): {len(dev_ev_df) - n_with_gold}")

# AP5 distribution for positive pairs
positive_pairs = dev_ev_df[dev_ev_df.gold_hand_1.notna()]
print(f"\n[2] AP5 distribution for positive pairs (n={len(positive_pairs)}):")
print(positive_pairs.ap5.describe())
print(f"\n  AP5 = 1.0 (perfect): {(positive_pairs.ap5 == 1.0).sum()}")
print(f"  AP5 >= 0.5: {(positive_pairs.ap5 >= 0.5).sum()}")
print(f"  AP5 in [0.3, 0.5): {((positive_pairs.ap5 >= 0.3) & (positive_pairs.ap5 < 0.5)).sum()}")
print(f"  AP5 in [0.1, 0.3): {((positive_pairs.ap5 >= 0.1) & (positive_pairs.ap5 < 0.3)).sum()}")
print(f"  AP5 < 0.1: {(positive_pairs.ap5 < 0.1).sum()}")
print(f"  AP5 = 0: {(positive_pairs.ap5 == 0).sum()}")

# =========================================================================
# For pairs where user scored LOW, what gold hands did they miss?
# =========================================================================
print("\n[3] Analyzing low-AP5 pairs...")

low_ap = positive_pairs[positive_pairs.ap5 < 0.3]
print(f"  Pairs with AP5 < 0.3: {len(low_ap)}")

# For each, what gold hands did the user miss?
missed_gold = []
for _, row in low_ap.iterrows():
    submitted_set = {row[f'submitted_hand_{k+1}'] for k in range(5) if pd.notna(row[f'submitted_hand_{k+1}'])}
    for k in range(5):
        col = f'gold_hand_{k+1}'
        if pd.notna(row[col]):
            if row[col] not in submitted_set:
                missed_gold.append({
                    'pair_id': row.pair_id,
                    'hand_id': row[col],
                })

missed_df = pd.DataFrame(missed_gold)
print(f"  Total missed gold hands (in low-AP5 pairs): {len(missed_df)}")
print(f"  Unique missed hands: {missed_df.hand_id.nunique() if len(missed_df) > 0 else 0}")

# =========================================================================
# Now look at: what characterizes gold hands the user MISSED vs HIT?
# Load hands.parquet for context
# =========================================================================
print("\n[4] Loading hands.parquet for missed vs hit analysis...")
import polars as pl
hands_pl = pl.read_parquet(DATA / "hands.parquet", columns=["hand_id", "big_blind", "final_pot", "players_at_showdown", "players_dealt", "phase"])
hands_pd = hands_pl.to_pandas()
del hands_pl
print(f"  Hands: {hands_pd.shape}")

# Get all gold hands
all_gold_hand_ids = set()
for _, row in positive_pairs.iterrows():
    for k in range(5):
        col = f'gold_hand_{k+1}'
        if pd.notna(row[col]):
            all_gold_hand_ids.add(row[col])

# Get user's submitted hands for positive pairs
all_submitted_for_pos = set()
for _, row in positive_pairs.iterrows():
    for k in range(5):
        col = f'submitted_hand_{k+1}'
        if pd.notna(row[col]):
            all_submitted_for_pos.add(row[col])

# Hands that user GOT RIGHT (intersection with gold)
correct_hands = all_gold_hand_ids & all_submitted_for_pos
# Hands that user MISSED (in gold but not submitted)
missed_hand_ids = set(missed_df.hand_id) if len(missed_df) > 0 else set()

print(f"\n  Total gold hands (positive pairs): {len(all_gold_hand_ids)}")
print(f"  Hands user got right: {len(correct_hands)} ({100*len(correct_hands)/len(all_gold_hand_ids):.1f}%)")
print(f"  Hands user missed: {len(missed_hand_ids)}")

# =========================================================================
# Compare hand characteristics: gold vs random
# =========================================================================
print("\n[5] Comparing hand characteristics: gold vs all dev...")

# Dev phase hands
dev_hands = hands_pd[hands_pd.phase == 'development']
print(f"  Dev-phase hands: {len(dev_hands):,}")

# Gold hands (full set)
gold_hands_data = dev_hands[dev_hands.hand_id.isin(all_gold_hand_ids)].copy()
print(f"  Gold hands found in dev: {len(gold_hands_data)}")

# Compare big_blind, final_pot, players_at_showdown
print(f"\n  Comparison (gold vs all dev):")
print(f"  Big blind: gold mean={gold_hands_data.big_blind.mean():.1f}, dev mean={dev_hands.big_blind.mean():.1f}")
print(f"  Final pot: gold mean={gold_hands_data.final_pot.mean():.1f}, dev mean={dev_hands.final_pot.mean():.1f}")
print(f"  Players at showdown: gold mean={gold_hands_data.players_at_showdown.mean():.2f}, dev mean={dev_hands.players_at_showdown.mean():.2f}")
print(f"  Players dealt: gold mean={gold_hands_data.players_dealt.mean():.2f}, dev mean={dev_hands.players_dealt.mean():.2f}")

# Phase of gold hands
print(f"\n  Gold hands phase distribution: {gold_hands_data.phase.value_counts().to_dict()}")

# Save research data
gold_hands_data.to_parquet('/home/z/my-project/kaggle/runs/gold_hands_data.parquet', index=False)
print(f"\n  Saved gold_hands_data.parquet")

# Save the dev evidence dataframe
dev_ev_df.to_parquet('/home/z/my-project/kaggle/runs/dev_ev_df.parquet', index=False)
print(f"  Saved dev_ev_df.parquet")

# =========================================================================
# KEY ANALYSIS: For each positive dev pair, what's the gold fraction in the user's picks?
# =========================================================================
print("\n[6] User's AP@5 by behavior family...")

dev_labels = pd.read_csv(DATA / "development_labels.csv")
label_map = dev_labels.set_index('pair_id')['behavior_family'].to_dict()

# Add behavior family to positive_pairs
positive_pairs_with_fam = positive_pairs.copy()
positive_pairs_with_fam['behavior_family'] = positive_pairs_with_fam.pair_id.map(label_map)

# Compute mean AP@5 per family
print(positive_pairs_with_fam.groupby('behavior_family').ap5.agg(['mean', 'median', 'count']))

# =========================================================================
# MOST IMPORTANT: For pairs where user got AP5 < 0.5, are the missed gold hands
# identifiable by simple features?
# =========================================================================
print("\n[7] Building features for gold vs non-gold classification...")

# We need to know which hands are gold vs not gold
# For each positive dev pair, the 5 gold hands are positive, all other shared hands are negative

import pickle
with open('/home/z/my-project/kaggle/runs/dev_pair_shared_hands.pkl', 'rb') as f:
    pair_to_shared_hands = pickle.load(f)

print(f"  Pairs with shared_hands map: {len(pair_to_shared_hands)}")

# Build training data: for each (positive pair, shared hand), features + is_gold label
gold_by_pair = {}
for _, row in positive_pairs.iterrows():
    gold_set = set()
    for k in range(5):
        col = f'gold_hand_{k+1}'
        if pd.notna(row[col]):
            gold_set.add(row[col])
    gold_by_pair[row.pair_id] = gold_set

print(f"  Gold-by-pair entries: {len(gold_by_pair)}")

# Build training data
training_rows = []
for pid, shared_hands in pair_to_shared_hands.items():
    if pid not in gold_by_pair:
        continue  # skip negatives (no gold)
    gold_set = gold_by_pair[pid]
    for hand_id in shared_hands:
        # Look up hand features
        hand_data = hands_pd[hands_pd.hand_id == hand_id]
        if len(hand_data) == 0:
            continue
        h = hand_data.iloc[0]
        training_rows.append({
            'pair_id': pid,
            'hand_id': hand_id,
            'big_blind': h.big_blind,
            'final_pot': h.final_pot,
            'pot_bb': h.final_pot / max(h.big_blind, 1),
            'players_at_showdown': h.players_at_showdown,
            'players_dealt': h.players_dealt,
            'is_gold': 1 if hand_id in gold_set else 0,
        })

train_df = pd.DataFrame(training_rows)
print(f"\n  Training data: {len(train_df)} (pair, hand) rows")
print(f"  Gold (positive): {train_df.is_gold.sum()}")
print(f"  Non-gold (negative): {(train_df.is_gold == 0).sum()}")
print(f"  Gold rate: {train_df.is_gold.mean():.4f}")

# Feature differences
print(f"\n  Feature means by is_gold:")
print(train_df.groupby('is_gold')[['big_blind', 'final_pot', 'pot_bb', 'players_at_showdown', 'players_dealt']].mean())

print("\n" + "="*80)
print("Research step 2 complete.")
print("="*80)
