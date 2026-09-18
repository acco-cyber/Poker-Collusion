"""
Step 4: Apply trained gold-pattern model to eval pairs and build submission.
Loads saved model, streams seats.parquet, builds top-5 per pair, creates submission.
"""
import json, gc, pickle, heapq, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import lightgbm as lgb

DATA = Path("/home/z/my-project/kaggle/comp-data")
OUT  = Path("/home/z/my-project/kaggle/runs")
USER_SUB = Path("/home/z/my-project/kaggle/user-submissions/pu-aware-output/submission.csv")

print("="*80)
print("Step 4: Apply gold-pattern model to eval pairs")
print("="*80)
t0 = time.time()

# Load model
with open(OUT / "gold_pattern_model.pkl", "rb") as f:
    saved = pickle.load(f)
final_model = saved['model']
feat_cols = saved['feat_cols']
print(f"Loaded model with {len(feat_cols)} features")

# Load user's submission
user_sub = pd.read_csv(USER_SUB)
print(f"User submission: {user_sub.shape}")

# Load eval pairs and build pair_lookup
eval_pairs = pd.read_csv(DATA / "evaluation_pairs.csv")
pair_lookup = {}
for r in eval_pairs.itertuples():
    p1, p2 = sorted([r.player_1, r.player_2])
    pair_lookup[(p1, p2)] = r.pair_id

needed_players = set()
for k in pair_lookup.keys():
    needed_players.add(k[0]); needed_players.add(k[1])
print(f"Eval pairs: {len(eval_pairs):,}, needed players: {len(needed_players):,}")

# Load hands.parquet
print("\nLoading hands.parquet...")
hands_pl = pl.read_parquet(DATA / "hands.parquet", columns=["hand_id", "big_blind", "final_pot", "players_at_showdown", "players_dealt"])
hands_dict = {}
for row in hands_pl.iter_rows(named=True):
    hands_dict[row["hand_id"]] = {
        "big_blind": int(row["big_blind"]),
        "final_pot": int(row["final_pot"]),
        "players_at_showdown": int(row["players_at_showdown"]),
        "players_dealt": int(row["players_dealt"]),
    }
del hands_pl
gc.collect()
print(f"hands_dict: {len(hands_dict):,}")

# Per-pair top-5 min-heap: (P(gold), hand_id)
pair_top5 = defaultdict(list)

# Stream seats.parquet for eval
print("\nStreaming seats.parquet...")
pf = pq.ParquetFile(DATA / "seats.parquet")
print(f"  {pf.metadata.num_rows:,} rows, {pf.num_row_groups} row groups")
t_stream = time.time()
hands_scored = 0

for rg_idx in range(pf.num_row_groups):
    table = pf.read_row_group(rg_idx, columns=[
        "hand_id", "player_id", "starting_stack", "total_contribution",
        "net_chips", "folded", "went_to_showdown", "won_share"
    ])
    df = table.to_pandas()
    del table
    df = df[df.player_id.isin(needed_players)]
    if len(df) == 0:
        continue

    # Build features for each (pair, hand)
    eval_rows = []
    eval_pair_hand_keys = []

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
                pid = pair_lookup.get((p1, p2))
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

        for (pid, hand_id), p in zip(eval_pair_hand_keys, p_gold):
            heap = pair_top5[pid]
            if len(heap) < 5:
                heapq.heappush(heap, (p, hand_id))
            else:
                heapq.heappushpop(heap, (p, hand_id))

    if (rg_idx + 1) % 20 == 0 or rg_idx == pf.num_row_groups - 1:
        elapsed = time.time() - t_stream
        print(f"  rg {rg_idx+1}/{pf.num_row_groups}: hands={hands_scored:,}, pairs={len(pair_top5):,}, elapsed={elapsed:.1f}s", flush=True)

print(f"\nDone in {time.time()-t_stream:.1f}s. Pairs with predictions: {len(pair_top5):,}")

# Build top-5 per pair (sorted desc by P(gold))
model_top5 = {}
for pid, heap in pair_top5.items():
    sorted_hands = sorted(heap, key=lambda x: -x[0])
    model_top5[pid] = [(p, h) for p, h in sorted_hands[:5]]

# =========================================================================
# Build 3 submission variants
# =========================================================================
print("\n" + "="*80)
print("Building 3 submission variants")
print("="*80)

# Variant 1: Pure model (replace all 5 with model top-5)
print("\n[1] Pure model: all 5 hands from gold-pattern model")
sub1 = user_sub.copy()
for idx, row in sub1.iterrows():
    pid = row.pair_id
    picks = model_top5.get(pid, [])
    new_ev = [h for p, h in picks[:5]]
    while len(new_ev) < 5:
        new_ev.append('NO_EVIDENCE')
    for k in range(5):
        sub1.at[idx, f'evidence_hand_{k+1}'] = new_ev[k]
sub1.to_csv(OUT / "sub_v1_pure_model.csv", index=False)

# Variant 2: Conservative 3+2 hybrid (user top-3 + 2 model picks)
print("\n[2] Conservative 3+2: user top-3 + 2 model picks")
sub2 = user_sub.copy()
for idx, row in sub2.iterrows():
    pid = row.pair_id
    user_hands = [row[f'evidence_hand_{k+1}'] for k in range(5)]
    keep_set = set(user_hands[:3])
    picks = model_top5.get(pid, [])
    new_picks = [h for p, h in picks if h not in keep_set][:2]
    new_ev = user_hands[:3] + new_picks
    while len(new_ev) < 5:
        new_ev.append('NO_EVIDENCE')
    for k in range(5):
        sub2.at[idx, f'evidence_hand_{k+1}'] = new_ev[k]
sub2.to_csv(OUT / "sub_v2_conservative_3plus2.csv", index=False)

# Variant 3: Moderate 2+3 hybrid (user top-2 + 3 model picks)
print("\n[3] Moderate 2+3: user top-2 + 3 model picks")
sub3 = user_sub.copy()
for idx, row in sub3.iterrows():
    pid = row.pair_id
    user_hands = [row[f'evidence_hand_{k+1}'] for k in range(5)]
    keep_set = set(user_hands[:2])
    picks = model_top5.get(pid, [])
    new_picks = [h for p, h in picks if h not in keep_set][:3]
    new_ev = user_hands[:2] + new_picks
    while len(new_ev) < 5:
        new_ev.append('NO_EVIDENCE')
    for k in range(5):
        sub3.at[idx, f'evidence_hand_{k+1}'] = new_ev[k]
sub3.to_csv(OUT / "sub_v3_moderate_2plus3.csv", index=False)

# Compute overlaps
print("\n" + "="*80)
print("Overlap with user baseline:")
print("="*80)
for sub, name in [(sub1, "v1 pure model"), (sub2, "v2 conservative 3+2"), (sub3, "v3 moderate 2+3")]:
    overlap = 0
    for _, row in sub.iterrows():
        pid = row.pair_id
        sub_set = {row[f'evidence_hand_{k+1}'] for k in range(5)}
        user_row = user_sub[user_sub.pair_id == pid].iloc[0]
        user_set = {user_row[f'evidence_hand_{k+1}'] for k in range(5)}
        overlap += len(sub_set & user_set)
    print(f"  {name}: {overlap/len(sub):.2f}/5.0 ({100*overlap/len(sub)/5:.1f}%)")

print(f"\n{'='*80}")
print(f"DONE in {time.time()-t0:.1f}s")
print(f"{'='*80}")
print(f"\nFiles saved:")
for f in sorted(OUT.glob("sub_v*.csv")):
    print(f"  - {f.name} ({f.stat().st_size/1e6:.1f} MB)")
