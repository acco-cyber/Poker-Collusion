"""
RESEARCH STEP 3: Train a gold-pattern classifier.

Based on research findings:
- Gold hands have 62% bigger pots (mean 184 BB vs 113 BB)
- Gold hands reach showdown 2x more often (0.63 vs 0.32)
- User's local AP@5 is 0.5454, public is ~0.378 (gap suggests model overfits dev)

NEW APPROACH: Train a LightGBM classifier on dev (pair, hand) features:
- Positive: (positive_pair, gold_hand) — 470 examples
- Negative: (positive_pair, non_gold_shared_hand) — ~18K examples

Features (per pair-hand):
- Hand features: big_blind, final_pot, players_at_showdown, players_dealt
- Pair-hand features (from seats): net1, net2, contrib1, contrib2, etc.
- Derived: pot_bb, asymmetry, etc.

Apply to eval: for each (pair, hand) shared by eval pair, predict P(gold)
Pick top-5 by P(gold) per pair.

This is a proper ML approach the user's notebook may not use.
"""
import json, gc, pickle, heapq
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

DATA = Path("/home/z/my-project/kaggle/comp-data")
OUT  = Path("/home/z/my-project/kaggle/runs")
USER_SUB = Path("/home/z/my-project/kaggle/user-submissions/pu-aware-output/submission.csv")
DIAG = Path("/home/z/my-project/kaggle/user-submissions/pu-aware-output/poker7_05_gated_consensus_DIAGNOSTICS_DO_NOT_SUBMIT.zip")

print("="*80)
print("RESEARCH STEP 3: Gold-Pattern Classifier")
print("="*80)
t0 = time.time() if False else __import__('time').time()

# Load user's submission
user_sub = pd.read_csv(USER_SUB)
eval_pairs = pd.read_csv(DATA / "evaluation_pairs.csv")
dev_labels = pd.read_csv(DATA / "development_labels.csv")

# Build pair_lookup for eval pairs
pair_lookup_eval = {}
for r in eval_pairs.itertuples():
    p1, p2 = sorted([r.player_1, r.player_2])
    pair_lookup_eval[(p1, p2)] = r.pair_id

needed_players_eval = set()
for k in pair_lookup_eval.keys():
    needed_players_eval.add(k[0]); needed_players_eval.add(k[1])
print(f"Eval pairs: {len(eval_pairs):,}, needed players: {len(needed_players_eval):,}")

# Build pair_lookup for dev pairs
pair_lookup_dev = {}
for r in dev_labels.itertuples():
    p1, p2 = sorted([r.player_1, r.player_2])
    pair_lookup_dev[(p1, p2)] = r.pair_id

needed_players_dev = set()
for k in pair_lookup_dev.keys():
    needed_players_dev.add(k[0]); needed_players_dev.add(k[1])
print(f"Dev pairs: {len(dev_labels):,}, needed players (dev): {len(needed_players_dev):,}")

# =========================================================================
# Step 1: Extract gold hand info from user's diagnostics
# =========================================================================
print("\n[1] Extracting gold evidence from diagnostics...")
import zipfile
with zipfile.ZipFile(DIAG, 'r') as z:
    with z.open('portfolio_outer_evidence.json') as f:
        outer_ev = json.load(f)

# Get gold_by_pair: pair_id -> set of gold hand_ids
gold_by_pair = {}
for entry in outer_ev:
    if 'gold_hands' in entry and entry['gold_hands']:
        gold_by_pair[entry['pair_id']] = set(entry['gold_hands'])
print(f"  Pairs with gold: {len(gold_by_pair)}")

# Get user's submitted hands (for cross-checking)
user_submitted_by_pair = {}
for entry in outer_ev:
    user_submitted_by_pair[entry['pair_id']] = entry['submitted_hands']

# =========================================================================
# Step 2: Load hands.parquet
# =========================================================================
print("\n[2] Loading hands.parquet...")
hands_pl = pl.read_parquet(DATA / "hands.parquet", columns=["hand_id", "big_blind", "final_pot", "players_at_showdown", "players_dealt", "phase"])
hands_dict = {}
for row in hands_pl.iter_rows(named=True):
    hands_dict[row["hand_id"]] = {
        "big_blind": int(row["big_blind"]),
        "final_pot": int(row["final_pot"]),
        "players_at_showdown": int(row["players_at_showdown"]),
        "players_dealt": int(row["players_dealt"]),
        "phase": row["phase"],
    }
del hands_pl
gc.collect()
print(f"  hands_dict: {len(hands_dict):,}")

# =========================================================================
# Step 3: Stream seats.parquet to build (pair, hand) features for DEV pairs
# =========================================================================
print("\n[3] Streaming seats.parquet for dev pairs...")
# For each (dev pair, shared hand), compute features
# Then label gold/non-gold for training

# Per-hand features for the pair:
# - net1, net2, contrib1, contrib2, stack1, stack2
# - showdown1, showdown2, folded1, folded2

# Build dev training data
dev_features_list = []
pf = pq.ParquetFile(DATA / "seats.parquet")
print(f"  Streaming {pf.metadata.num_rows:,} seats rows...")

import time as time_mod
t_stream = time_mod.time()

for rg_idx in range(pf.num_row_groups):
    table = pf.read_row_group(rg_idx, columns=[
        "hand_id", "player_id", "starting_stack", "total_contribution",
        "net_chips", "folded", "went_to_showdown", "won_share"
    ])
    df = table.to_pandas()
    del table
    # Filter to needed players (dev only for training data)
    df = df[df.player_id.isin(needed_players_dev)]
    if len(df) == 0:
        continue

    for hand_id, group in df.groupby("hand_id", sort=False):
        if hand_id not in hands_dict:
            continue
        h_info = hands_dict[hand_id]
        bb = max(h_info["big_blind"], 1)
        final_pot = h_info["final_pot"]

        players = group.player_id.values
        stacks = group.starting_stack.values
        contribs = group.total_contribution.values
        nets = group.net_chips.values
        foldeds = group.folded.values
        showdowns = group.went_to_showdown.values
        won_shares = group.won_share.values

        n_players = len(players)
        if n_players < 2:
            continue

        sorted_idx = np.argsort(players)
        for ii in range(n_players):
            i = sorted_idx[ii]
            p1 = players[i]
            for jj in range(ii + 1, n_players):
                j = sorted_idx[jj]
                p2 = players[j]
                pid = pair_lookup_dev.get((p1, p2))
                if pid is None:
                    continue

                # Compute (pair, hand) features
                net1, net2 = nets[i], nets[j]
                contrib1, contrib2 = contribs[i], contribs[j]
                stack1, stack2 = stacks[i], stacks[j]
                folded1, folded2 = bool(foldeds[i]), bool(foldeds[j])
                showdown1, showdown2 = bool(showdowns[i]), bool(showdowns[j])
                won1, won2 = won_shares[i] if not pd.isna(won_shares[i]) else 0, won_shares[j] if not pd.isna(won_shares[j]) else 0

                asymmetry = abs(net1 - net2) / bb
                one_wins = (net1 > 0) != (net2 > 0)
                transfer_score = asymmetry * (2.0 if one_wins else 0.3)

                passive_score = 0.0
                if showdown1 and showdown2:
                    if final_pot < bb * 4:
                        passive_score = 3.0 - (final_pot / bb / 4.0)
                    elif final_pot < bb * 10:
                        passive_score = 1.0 - (final_pot / bb / 10.0)
                if contrib1 < bb * 2 and contrib2 < bb * 2:
                    passive_score += 0.5

                aggressive_score = 0.0
                if final_pot > bb * 20:
                    if one_wins and asymmetry > bb * 10:
                        aggressive_score = min(5.0, asymmetry / bb / 5.0)

                # Hand-level features (gold pattern: bigger pots, more showdowns)
                pot_bb = final_pot / bb
                players_at_showdown = h_info["players_at_showdown"]
                players_dealt = h_info["players_dealt"]
                n_showdown_in_pair = int(showdown1) + int(showdown2)
                both_showdown = int(showdown1 and showdown2)
                either_folded = int(folded1 or folded2)
                both_folded = int(folded1 and folded2)

                # Pair-level features
                net1_bb = net1 / bb
                net2_bb = net2 / bb
                contrib1_bb = contrib1 / bb
                contrib2_bb = contrib2 / bb
                stack1_bb = stack1 / bb
                stack2_bb = stack2 / bb
                won_share_diff = abs(won1 - won2)
                net_sum = net1 + net2
                net_diff = net1 - net2

                # Label
                is_gold = 1 if (pid in gold_by_pair and hand_id in gold_by_pair[pid]) else 0

                dev_features_list.append({
                    'pair_id': pid,
                    'hand_id': hand_id,
                    'big_blind': bb,
                    'final_pot': final_pot,
                    'pot_bb': pot_bb,
                    'players_at_showdown': players_at_showdown,
                    'players_dealt': players_dealt,
                    'n_showdown_in_pair': n_showdown_in_pair,
                    'both_showdown': both_showdown,
                    'either_folded': either_folded,
                    'both_folded': both_folded,
                    'net1_bb': net1_bb,
                    'net2_bb': net2_bb,
                    'asymmetry': asymmetry,
                    'one_wins': int(one_wins),
                    'transfer_score': transfer_score,
                    'passive_score': passive_score,
                    'aggressive_score': aggressive_score,
                    'contrib1_bb': contrib1_bb,
                    'contrib2_bb': contrib2_bb,
                    'stack1_bb': stack1_bb,
                    'stack2_bb': stack2_bb,
                    'won_share_diff': won_share_diff,
                    'net_sum': net_sum,
                    'net_diff': net_diff,
                    'is_gold': is_gold,
                })

    if (rg_idx + 1) % 40 == 0 or rg_idx == pf.num_row_groups - 1:
        elapsed = time_mod.time() - t_stream
        print(f"    rg {rg_idx+1}/{pf.num_row_groups}: rows={len(dev_features_list):,}, elapsed={elapsed:.1f}s", flush=True)

print(f"  Done. {len(dev_features_list):,} (pair, hand) rows for dev.")

# =========================================================================
# Step 4: Train LightGBM classifier on dev data
# =========================================================================
print("\n[4] Training gold-pattern classifier...")

dev_df = pd.DataFrame(dev_features_list)
print(f"  Dev feature rows: {len(dev_df):,}")
print(f"  Gold (positive): {dev_df.is_gold.sum()}")
print(f"  Non-gold (negative): {(dev_df.is_gold == 0).sum()}")

# Filter to only pairs with gold (positive pairs)
positive_dev_df = dev_df[dev_df.pair_id.isin(gold_by_pair.keys())].copy()
print(f"  Filtered to positive pairs: {len(positive_dev_df):,}")
print(f"  Gold in positive pairs: {positive_dev_df.is_gold.sum()}")

# Feature columns
feat_cols = [c for c in positive_dev_df.columns if c not in ('pair_id', 'hand_id', 'is_gold')]
print(f"  Features: {len(feat_cols)}")

X = positive_dev_df[feat_cols].values.astype(np.float32)
y = positive_dev_df.is_gold.values

# Impute NaN
for c in feat_cols:
    if positive_dev_df[c].isna().any():
        X[:, feat_cols.index(c)] = np.nan_to_num(X[:, feat_cols.index(c)], nan=0.0)

# Group by pair (for StratifiedGroupKFold)
groups = positive_dev_df.pair_id.values

# 5-fold CV
from sklearn.model_selection import StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260918)

# LightGBM params
def lgb_params():
    return dict(
        objective='binary',
        num_leaves=15,
        learning_rate=0.05,
        n_estimators=200,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=20260918,
        n_jobs=2,
        verbose=-1,
    )

# Train 5-fold CV to estimate performance
print("\n  5-fold CV...")
oof = np.zeros(len(positive_dev_df))
fold_aps = []
for fold, (tr, va) in enumerate(sgkf.split(X, y, groups)):
    X_tr, X_va = X[tr], X[va]
    y_tr, y_va = y[tr], y[va]
    model = lgb.LGBMClassifier(**lgb_params())
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(20, verbose=False)])
    oof[va] = model.predict_proba(X_va)[:, 1]
    if y_va.sum() > 0:
        ap = average_precision_score(y_va, oof[va])
    else:
        ap = 0.0
    fold_aps.append(ap)
    print(f"    Fold {fold}: AP={ap:.4f}, best_iter={model.best_iteration_}")

print(f"\n  Mean CV AP: {np.mean(fold_aps):.4f}")

# Train final model on full dev data
print("\n  Training final model on all dev data...")
final_model = lgb.LGBMClassifier(**lgb_params())
final_model.fit(X, y)

# Feature importance
importance = pd.DataFrame({
    'feature': feat_cols,
    'gain': final_model.booster_.feature_importance(importance_type='gain'),
}).sort_values('gain', ascending=False)
print("\n  Top 10 features by gain:")
print(importance.head(10).to_string(index=False))

# Save model
with open(OUT / "gold_pattern_model.pkl", "wb") as f:
    pickle.dump({'model': final_model, 'feat_cols': feat_cols}, f)
print(f"\n  Saved gold_pattern_model.pkl")

# =========================================================================
# Step 5: Apply to eval pairs — for each (pair, hand), predict P(gold)
# Stream seats.parquet, predict P(gold), keep top-5 per pair
# =========================================================================
print("\n[5] Applying model to eval pairs (streaming seats)...")
print("  This will take ~5 min...")

# Per-pair top-5 min-heap: (P(gold), hand_id)
pair_top5 = defaultdict(list)

pf = pq.ParquetFile(DATA / "seats.parquet")
print(f"  Streaming {pf.metadata.num_rows:,} seats rows for eval...")
t_stream = time_mod.time()
hands_scored = 0

for rg_idx in range(pf.num_row_groups):
    table = pf.read_row_group(rg_idx, columns=[
        "hand_id", "player_id", "starting_stack", "total_contribution",
        "net_chips", "folded", "went_to_showdown", "won_share"
    ])
    df = table.to_pandas()
    del table
    df = df[df.player_id.isin(needed_players_eval)]
    if len(df) == 0:
        continue

    # Build features for each (pair, hand) and predict
    # Batch by hand for efficiency
    eval_rows = []
    eval_pair_hand_keys = []  # (pair_id, hand_id) for each row

    for hand_id, group in df.groupby("hand_id", sort=False):
        if hand_id not in hands_dict:
            continue
        h_info = hands_dict[hand_id]
        bb = max(h_info["big_blind"], 1)
        final_pot = h_info["final_pot"]

        players = group.player_id.values
        stacks = group.starting_stack.values
        contribs = group.total_contribution.values
        nets = group.net_chips.values
        foldeds = group.folded.values
        showdowns = group.went_to_showdown.values
        won_shares = group.won_share.values

        n_players = len(players)
        if n_players < 2:
            continue

        sorted_idx = np.argsort(players)
        for ii in range(n_players):
            i = sorted_idx[ii]
            p1 = players[i]
            for jj in range(ii + 1, n_players):
                j = sorted_idx[jj]
                p2 = players[j]
                pid = pair_lookup_eval.get((p1, p2))
                if pid is None:
                    continue

                net1, net2 = nets[i], nets[j]
                contrib1, contrib2 = contribs[i], contribs[j]
                stack1, stack2 = stacks[i], stacks[j]
                folded1, folded2 = bool(foldeds[i]), bool(foldeds[j])
                showdown1, showdown2 = bool(showdowns[i]), bool(showdowns[j])
                won1 = won_shares[i] if not pd.isna(won_shares[i]) else 0
                won2 = won_shares[j] if not pd.isna(won_shares[j]) else 0

                asymmetry = abs(net1 - net2) / bb
                one_wins = (net1 > 0) != (net2 > 0)
                transfer_score = asymmetry * (2.0 if one_wins else 0.3)

                passive_score = 0.0
                if showdown1 and showdown2:
                    if final_pot < bb * 4:
                        passive_score = 3.0 - (final_pot / bb / 4.0)
                    elif final_pot < bb * 10:
                        passive_score = 1.0 - (final_pot / bb / 10.0)
                if contrib1 < bb * 2 and contrib2 < bb * 2:
                    passive_score += 0.5

                aggressive_score = 0.0
                if final_pot > bb * 20:
                    if one_wins and asymmetry > bb * 10:
                        aggressive_score = min(5.0, asymmetry / bb / 5.0)

                pot_bb = final_pot / bb
                players_at_showdown = h_info["players_at_showdown"]
                players_dealt = h_info["players_dealt"]
                n_showdown_in_pair = int(showdown1) + int(showdown2)
                both_showdown = int(showdown1 and showdown2)
                either_folded = int(folded1 or folded2)
                both_folded = int(folded1 and folded2)
                net1_bb = net1 / bb
                net2_bb = net2 / bb
                contrib1_bb = contrib1 / bb
                contrib2_bb = contrib2 / bb
                stack1_bb = stack1 / bb
                stack2_bb = stack2 / bb
                won_share_diff = abs(won1 - won2)
                net_sum = net1 + net2
                net_diff = net1 - net2

                row = [
                    bb, final_pot, pot_bb, players_at_showdown, players_dealt,
                    n_showdown_in_pair, both_showdown, either_folded, both_folded,
                    net1_bb, net2_bb, asymmetry, int(one_wins), transfer_score,
                    passive_score, aggressive_score, contrib1_bb, contrib2_bb,
                    stack1_bb, stack2_bb, won_share_diff, net_sum, net_diff,
                ]
                eval_rows.append(row)
                eval_pair_hand_keys.append((pid, hand_id))

        hands_scored += 1

    # Predict P(gold) for this batch
    if len(eval_rows) > 0:
        X_batch = np.array(eval_rows, dtype=np.float32)
        X_batch = np.nan_to_num(X_batch, nan=0.0)
        p_gold = final_model.predict_proba(X_batch)[:, 1]

        # Update top-5 per pair
        for (pid, hand_id), p in zip(eval_pair_hand_keys, p_gold):
            heap = pair_top5[pid]
            if len(heap) < 5:
                heapq.heappush(heap, (p, hand_id))
            else:
                heapq.heappushpop(heap, (p, hand_id))

    if (rg_idx + 1) % 20 == 0 or rg_idx == pf.num_row_groups - 1:
        elapsed = time_mod.time() - t_stream
        print(f"    rg {rg_idx+1}/{pf.num_row_groups}: hands={hands_scored:,}, pairs_w_top5={len(pair_top5):,}, elapsed={elapsed:.1f}s", flush=True)

print(f"  Done in {time_mod.time()-t_stream:.1f}s. Pairs with predictions: {len(pair_top5):,}")

# =========================================================================
# Step 6: Build submission using gold-pattern predictions
# =========================================================================
print("\n[6] Building submission...")

# Strategy: HYBRID
# - Keep user's top 2 hands (highest confidence from their LGBMRanker)
# - Replace hands 3, 4, 5 with our top 3 gold-pattern predictions (excl. duplicates)
# This preserves user's strongest signal but tries to improve their weaker picks

sub = user_sub.copy()
n_replaced_3 = 0
n_replaced_4 = 0
n_replaced_5 = 0

# Build top-K per pair from our model (sorted desc)
model_top5 = {}
for pid, heap in pair_top5.items():
    sorted_hands = sorted(heap, key=lambda x: -x[0])
    model_top5[pid] = [(p, h) for p, h in sorted_hands[:5]]

for idx, row in sub.iterrows():
    pid = row.pair_id
    user_hands = [row[f'evidence_hand_{k+1}'] for k in range(5)]
    keep_set = set(user_hands[:2])  # Keep user's top 2
    model_picks = model_top5.get(pid, [])

    # Get top 3 from our model that aren't already in user's top 2
    new_picks = [h for p, h in model_picks if h not in keep_set][:3]

    # Build new evidence: user's 1, 2 + our 3 picks
    new_ev = user_hands[:2] + new_picks
    while len(new_ev) < 5:
        new_ev.append('NO_EVIDENCE')

    for k in range(5):
        sub.at[idx, f'evidence_hand_{k+1}'] = new_ev[k]

    if len(new_picks) >= 1:
        n_replaced_3 += 1
    if len(new_picks) >= 2:
        n_replaced_4 += 1
    if len(new_picks) >= 3:
        n_replaced_5 += 1

sub.to_csv(OUT / "sub_gold_pattern_hybrid_2plus3.csv", index=False)
print(f"  Saved sub_gold_pattern_hybrid_2plus3.csv: {sub.shape}")
print(f"  Replaced hand 3: {n_replaced_3:,}")
print(f"  Replaced hand 4: {n_replaced_4:,}")
print(f"  Replaced hand 5: {n_replaced_5:,}")

# Overlap with user's baseline
overlap = 0
for _, row in sub.iterrows():
    pid = row.pair_id
    sub_set = {row[f'evidence_hand_{k+1}'] for k in range(5)}
    user_row = user_sub[user_sub.pair_id == pid].iloc[0]
    user_set = {user_row[f'evidence_hand_{k+1}'] for k in range(5)}
    overlap += len(sub_set & user_set)
print(f"  Overlap with user baseline: {overlap/len(sub):.2f}/5.0 ({100*overlap/len(sub)/5:.1f}%)")

# Also save a more conservative variant: keep user top-3 + 2 model picks
print("\n  Building conservative variant: user top-3 + 2 model picks")
sub_cons = user_sub.copy()
for idx, row in sub_cons.iterrows():
    pid = row.pair_id
    user_hands = [row[f'evidence_hand_{k+1}'] for k in range(5)]
    keep_set = set(user_hands[:3])
    model_picks = model_top5.get(pid, [])
    new_picks = [h for p, h in model_picks if h not in keep_set][:2]
    new_ev = user_hands[:3] + new_picks
    while len(new_ev) < 5:
        new_ev.append('NO_EVIDENCE')
    for k in range(5):
        sub_cons.at[idx, f'evidence_hand_{k+1}'] = new_ev[k]

sub_cons.to_csv(OUT / "sub_gold_pattern_conservative_3plus2.csv", index=False)
print(f"  Saved sub_gold_pattern_conservative_3plus2.csv: {sub_cons.shape}")

overlap_cons = 0
for _, row in sub_cons.iterrows():
    pid = row.pair_id
    sub_set = {row[f'evidence_hand_{k+1}'] for k in range(5)}
    user_row = user_sub[user_sub.pair_id == pid].iloc[0]
    user_set = {user_row[f'evidence_hand_{k+1}'] for k in range(5)}
    overlap_cons += len(sub_set & user_set)
print(f"  Overlap with user baseline: {overlap_cons/len(sub_cons):.2f}/5.0 ({100*overlap_cons/len(sub_cons)/5:.1f}%)")

# Save model top5 for reference
with open(OUT / "model_top5.json", "w") as f:
    json.dump({pid: [(p, h) for p, h in picks] for pid, picks in model_top5.items()}, f)
print(f"\n  Saved model_top5.json")

print(f"\n{'='*80}")
print(f"DONE in {time_mod.time()-t0:.1f}s")
print(f"{'='*80}")
print(f"\nFiles saved:")
print(f"  - {OUT/'gold_pattern_model.pkl'}")
print(f"  - {OUT/'sub_gold_pattern_hybrid_2plus3.csv'}")
print(f"  - {OUT/'sub_gold_pattern_conservative_3plus2.csv'}")
print(f"  - {OUT/'model_top5.json'}")
