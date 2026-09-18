"""
DEEP RESEARCH: Analyze gold evidence patterns to find a realistic path to 0.90.

The user has plateaued at 0.83065. Their composite is:
- pair_ap ≈ 0.95 (70% weight → 0.665)
- evidence_map5 ≈ 0.378 (20% weight → 0.076)
- behavior_map ≈ 0.90 (10% weight → 0.090)
- Total: ~0.831

To reach 0.90, we need to improve the evidence_map5 component.
Local user estimate: 0.5454 (so they think they're at ~0.91 locally)
Public reality: 0.378 (much worse on hidden test set)

WHY THE GAP? The user's evidence ranker overfits to dev set patterns.

HYPOTHESIS: If we can find what makes a hand "gold" (using dev set),
we might be able to pick better evidence for the test set.

PLAN:
1. For each (pair, hand) in development_evidence.csv, compute features
2. For each pair in dev, also compute features for OTHER hands they played
3. Train a binary classifier: P(hand is gold | features)
4. Apply to eval pairs: pick top-5 hands by gold-probability
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("/home/z/my-project/kaggle/comp-data")
USER_SUB = Path("/home/z/my-project/kaggle/user-submissions/pu-aware-output/submission.csv")

print("="*80)
print("DEEP RESEARCH: Gold Evidence Pattern Analysis")
print("="*80)

# =========================================================================
# Step 1: Load dev gold evidence
# =========================================================================
print("\n[1] Loading dev gold evidence...")
dev_labels = pd.read_csv(DATA / "development_labels.csv")
dev_evidence = pd.read_csv(DATA / "development_evidence.csv")
eval_pairs = pd.read_csv(DATA / "evaluation_pairs.csv")

print(f"  Dev labels: {dev_labels.shape}")
print(f"  Dev evidence (gold hands): {dev_evidence.shape}")
print(f"  Eval pairs: {eval_pairs.shape}")

# How many pairs have gold evidence?
positive_pairs = dev_labels[dev_labels.label == 1]
print(f"\n  Positive pairs in dev: {len(positive_pairs)}")
print(f"  Pairs with gold evidence: {dev_evidence.pair_id.nunique()}")
print(f"  Behavior families of positive pairs:")
print(positive_pairs.behavior_family.value_counts())

# =========================================================================
# Step 2: For each dev pair, what fraction of hands are gold?
# =========================================================================
print("\n[2] Analyzing gold hand patterns...")

# Build map: pair_id -> set of gold hand_ids
gold_by_pair = dev_evidence.groupby('pair_id')['hand_id'].apply(set).to_dict()
print(f"  Gold pairs: {len(gold_by_pair)}")

# Each positive pair has exactly 5 gold hands
gold_counts = dev_evidence.groupby('pair_id').size()
print(f"  Gold hands per pair: min={gold_counts.min()}, max={gold_counts.max()}, mean={gold_counts.mean():.1f}")

# =========================================================================
# Step 3: What's the user's prediction quality on dev pairs?
# =========================================================================
# Note: User's submission is for EVAL pairs (112540), not dev pairs (1860)
# So we can't directly compare. But we can look at the user's notebook diagnostics.

# Load user's notebook diagnostics if available
import os
diag_path = Path("/home/z/my-project/kaggle/user-submissions/pu-aware-output/poker7_05_gated_consensus_DIAGNOSTICS_DO_NOT_SUBMIT.zip")
if diag_path.exists():
    import zipfile
    print(f"\n[3] Extracting user's notebook diagnostics...")
    with zipfile.ZipFile(diag_path, 'r') as z:
        # Get validation report
        try:
            with z.open('validation_report.json') as f:
                val_report = pd.read_json(f, typ='series')
                print(f"  Local composite_proxy: {val_report.get('composite_proxy', 'N/A')}")
                print(f"  Local pair_ap: {val_report.get('pair_ap', 'N/A')}")
                print(f"  Local evidence_map5: {val_report.get('evidence_map5', 'N/A')}")
                print(f"  Local behavior_map: {val_report.get('behavior_map', 'N/A')}")
        except Exception as e:
            print(f"  Couldn't read validation_report: {e}")

        # Get outer predictions (user's predictions on dev pairs)
        try:
            with z.open('outer_predictions.csv') as f:
                outer_pred = pd.read_csv(f)
            print(f"\n  Outer predictions shape: {outer_pred.shape}")
            print(f"  Columns: {list(outer_pred.columns)}")
            print(f"  Label distribution: {outer_pred.label.value_counts().to_dict()}")
            # Filter to confirmed positives only
            confirmed_pos = outer_pred[outer_pred.label == 1]
            print(f"  Confirmed positives: {len(confirmed_pos)}")

            # Get user's evidence picks for these dev pairs
            try:
                with z.open('portfolio_outer_evidence.json') as f:
                    import json
                    outer_ev = json.load(f)
                print(f"  Outer evidence entries: {len(outer_ev)}")
                # Look at first few entries
                if isinstance(outer_ev, dict):
                    sample_keys = list(outer_ev.keys())[:3]
                    for k in sample_keys:
                        print(f"    {k}: {outer_ev[k]}")
                elif isinstance(outer_ev, list):
                    print(f"    Sample: {outer_ev[:2]}")
            except Exception as e:
                print(f"  Couldn't read evidence: {e}")
        except Exception as e:
            print(f"  Couldn't read outer_predictions: {e}")

# =========================================================================
# Step 4: Check eval pairs structure
# =========================================================================
print("\n[4] Analyzing eval pairs...")
print(f"  Eval pairs: {len(eval_pairs):,}")
print(f"  Pairs with shared_hands >= 50: {(eval_pairs.shared_hands >= 50).sum()}")
print(f"  Pairs with shared_hands >= 100: {(eval_pairs.shared_hands >= 100).sum()}")
print(f"  Shared_hands stats: min={eval_pairs.shared_hands.min()}, max={eval_pairs.shared_hands.max()}, mean={eval_pairs.shared_hands.mean():.1f}")

# =========================================================================
# Step 5: For dev positive pairs, what do their gold hands look like?
# =========================================================================
print("\n[5] Analyzing gold hand characteristics...")

# Load hands.parquet (small, 48MB)
import polars as pl
hands_pl = pl.read_parquet(DATA / "hands.parquet", columns=["hand_id", "phase", "big_blind", "final_pot", "players_at_showdown", "players_dealt"])
print(f"  Hands: {hands_pl.shape}")

# Filter to dev-phase hands only (where gold hands live)
hands_pd = hands_pl.to_pandas()
del hands_pl

# Filter to development phase (which is where labels apply)
dev_phase_hands = hands_pd[hands_pd.phase == 'development']
print(f"  Dev-phase hands: {len(dev_phase_hands):,}")
print(f"  Test/other hands: {len(hands_pd) - len(dev_phase_hands):,}")

# How many gold hands are in dev phase?
gold_hand_ids = set(dev_evidence.hand_id.unique())
print(f"  Unique gold hand_ids: {len(gold_hand_ids)}")
print(f"  Gold hands in dev phase: {len(gold_hand_ids & set(dev_phase_hands.hand_id))}")

# For each dev positive pair, how many shared hands do they have?
# We need seats.parquet to figure this out
print("\n  Loading seats.parquet for dev pairs to count shared hands...")

# Get all dev pair players
dev_pair_players = set()
for r in dev_labels.itertuples():
    dev_pair_players.add(r.player_1)
    dev_pair_players.add(r.player_2)
print(f"  Dev pair players: {len(dev_pair_players):,}")

# Stream seats to find shared hands per dev pair
import pyarrow.parquet as pq
import gc

pair_to_shared_hands = {pid: set() for pid in dev_labels.pair_id}
pair_lookup = {}
for r in dev_labels.itertuples():
    p1, p2 = sorted([r.player_1, r.player_2])
    pair_lookup[(p1, p2)] = r.pair_id

pf = pq.ParquetFile(DATA / "seats.parquet")
print(f"  Streaming {pf.metadata.num_rows:,} seats rows...")

for rg_idx in range(pf.num_row_groups):
    table = pf.read_row_group(rg_idx, columns=["hand_id", "player_id"])
    df = table.to_pandas()
    del table
    df = df[df.player_id.isin(dev_pair_players)]
    if len(df) == 0:
        continue

    # For each hand in this row group, find pairs that share it
    for hand_id, group in df.groupby("hand_id", sort=False):
        players_in_hand = sorted(group.player_id.unique())
        for i in range(len(players_in_hand)):
            p1 = players_in_hand[i]
            for j in range(i+1, len(players_in_hand)):
                p2 = players_in_hand[j]
                pid = pair_lookup.get((p1, p2))
                if pid is not None:
                    pair_to_shared_hands[pid].add(hand_id)

    if (rg_idx + 1) % 50 == 0:
        print(f"    rg {rg_idx+1}/{pf.num_row_groups}", flush=True)

# Stats
shared_counts = [len(v) for v in pair_to_shared_hands.values()]
print(f"\n  Shared hands per dev pair: min={min(shared_counts)}, max={max(shared_counts)}, mean={np.mean(shared_counts):.1f}")

# For each dev positive pair, what % of shared hands are gold?
gold_fractions = []
for pid in dev_labels[dev_labels.label == 1].pair_id:
    if pid in pair_to_shared_hands and pid in gold_by_pair:
        n_shared = len(pair_to_shared_hands[pid])
        n_gold = len(gold_by_pair[pid] & pair_to_shared_hands[pid])
        if n_shared > 0:
            gold_fractions.append(n_gold / n_shared)

print(f"  Gold fraction (gold hands / total shared) for dev positives:")
print(f"    min={min(gold_fractions):.4f}, max={max(gold_fractions):.4f}, mean={np.mean(gold_fractions):.4f}")
print(f"    median={np.median(gold_fractions):.4f}")

# Save the pair_to_shared_hands for later use
import pickle
with open('/home/z/my-project/kaggle/runs/dev_pair_shared_hands.pkl', 'wb') as f:
    pickle.dump(pair_to_shared_hands, f)
print(f"\n  Saved dev_pair_shared_hands.pkl")

print("\n" + "="*80)
print("Research step 1 complete. Ready for next analysis.")
print("="*80)
