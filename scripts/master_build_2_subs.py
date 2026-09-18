"""
Master script: Build evidence cache + 2 strategic submissions in one pass.
Memory-efficient streaming. Runs in ~5 min on 4GB RAM.

Output:
  /home/z/my-project/kaggle/runs/evidence_cache.json (top-10 per pair per signal)
  /home/z/my-project/kaggle/runs/sub_01_conservative_4plus1.csv
  /home/z/my-project/kaggle/runs/sub_02_targeted_soft_play.csv
"""
import os, gc, time, json, heapq
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import polars as pl

DATA = Path("/home/z/my-project/kaggle/comp-data")
OUT  = Path("/home/z/my-project/kaggle/runs")
OUT.mkdir(parents=True, exist_ok=True)
USER_SUB = Path("/home/z/my-project/kaggle/user-submissions/pu-aware-output/submission.csv")

print("="*80)
print("Master Build: Evidence Cache + 2 Strategic Submissions")
print("="*80)
t0 = time.time()

# Load user's previous 0.83065 submission
user_sub = pd.read_csv(USER_SUB)
print(f"User baseline submission: {user_sub.shape}")

# Load eval pairs and build pair_lookup
eval_pairs = pd.read_csv(DATA / "evaluation_pairs.csv")
pair_lookup = {}
for r in eval_pairs.itertuples():
    p1, p2 = sorted([r.player_1, r.player_2])
    pair_lookup[(p1, p2)] = r.pair_id

needed_players = set()
for k in pair_lookup.keys():
    needed_players.add(k[0]); needed_players.add(k[1])
print(f"Needed players: {len(needed_players):,}")

# Load hands.parquet
print("\nLoading hands.parquet...")
hands_pl = pl.read_parquet(DATA / "hands.parquet", columns=["hand_id", "big_blind", "final_pot"])
hands_dict = {row["hand_id"]: (int(row["big_blind"]), int(row["final_pot"]))
              for row in hands_pl.iter_rows(named=True)}
del hands_pl
gc.collect()
print(f"hands_dict: {len(hands_dict):,}")

# Per-pair evidence heap for 3 signals
pair_evidence = defaultdict(lambda: {"transfer": [], "passive": [], "aggressive": []})

# Stream seats.parquet row-group by row-group
print("\nStreaming seats.parquet (this takes ~5 min)...")
pf = pq.ParquetFile(DATA / "seats.parquet")
print(f"  {pf.metadata.num_rows:,} rows, {pf.num_row_groups} row groups")
t_stream = time.time()
hands_scored = 0

for rg_idx in range(pf.num_row_groups):
    table = pf.read_row_group(rg_idx, columns=[
        "hand_id", "player_id", "net_chips", "total_contribution", "went_to_showdown"
    ])
    df = table.to_pandas()
    del table
    df = df[df.player_id.isin(needed_players)]
    if len(df) == 0:
        continue

    for hand_id, group in df.groupby("hand_id", sort=False):
        if hand_id not in hands_dict:
            continue
        big_blind, final_pot = hands_dict[hand_id]
        bb = max(big_blind, 1)

        players = group.player_id.values
        nets = group.net_chips.values
        contribs = group.total_contribution.values
        showdowns = group.went_to_showdown.values
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
                showdown1, showdown2 = bool(showdowns[i]), bool(showdowns[j])

                # Transfer score (asymmetric chip transfer)
                asymmetry = abs(net1 - net2) / bb
                one_wins = (net1 > 0) != (net2 > 0)
                transfer_score = asymmetry * (2.0 if one_wins else 0.3)

                # Passive score (soft play)
                passive_score = 0.0
                if showdown1 and showdown2:
                    if final_pot < bb * 4:
                        passive_score = 3.0 - (final_pot / bb / 4.0)
                    elif final_pot < bb * 10:
                        passive_score = 1.0 - (final_pot / bb / 10.0)
                if contrib1 < bb * 2 and contrib2 < bb * 2:
                    passive_score += 0.5

                # Aggressive score (coordinated isolation)
                aggressive_score = 0.0
                if final_pot > bb * 20:
                    if one_wins and asymmetry > bb * 10:
                        aggressive_score = min(5.0, asymmetry / bb / 5.0)

                ev = pair_evidence[pid]
                for sig_name, score in [("transfer", transfer_score),
                                        ("passive", passive_score),
                                        ("aggressive", aggressive_score)]:
                    heap = ev[sig_name]
                    if len(heap) < 10:
                        heapq.heappush(heap, (score, hand_id))
                    else:
                        heapq.heappushpop(heap, (score, hand_id))

        hands_scored += 1

    if (rg_idx + 1) % 20 == 0 or rg_idx == pf.num_row_groups - 1:
        elapsed = time.time() - t_stream
        print(f"  rg {rg_idx+1}/{pf.num_row_groups}: hands={hands_scored:,}, elapsed={elapsed:.1f}s", flush=True)

print(f"\nStreaming done in {time.time()-t_stream:.1f}s. Pairs with evidence: {len(pair_evidence):,}")
del hands_dict
gc.collect()

# Build top-10 per pair per signal (sorted by score desc)
print("\nBuilding top-10 heaps...")
chip_top10 = {}
passive_top10 = {}
aggressive_top10 = {}
for pid, ev in pair_evidence.items():
    chip_top10[pid] = [h for s, h in sorted(ev["transfer"], key=lambda x: -x[0])[:10]]
    passive_top10[pid] = [h for s, h in sorted(ev["passive"], key=lambda x: -x[0])[:10]]
    aggressive_top10[pid] = [h for s, h in sorted(ev["aggressive"], key=lambda x: -x[0])[:10]]
print(f"Built top-10 for {len(chip_top10):,} pairs")

# Save evidence cache as JSON (compact)
print("\nSaving evidence_cache.json...")
cache = {
    "chip_top10": chip_top10,
    "passive_top10": passive_top10,
    "aggressive_top10": aggressive_top10,
}
with open(OUT / "evidence_cache.json", "w") as f:
    json.dump(cache, f)
cache_size_mb = (OUT / "evidence_cache.json").stat().st_size / 1e6
print(f"  Saved: {cache_size_mb:.1f} MB")

del pair_evidence, cache
gc.collect()

# =========================================================================
# Build 2 strategic submissions
# =========================================================================
print("\n" + "="*80)
print("Building 2 strategic submissions")
print("="*80)

evidence_cols = ['evidence_hand_1', 'evidence_hand_2', 'evidence_hand_3', 'evidence_hand_4', 'evidence_hand_5']

# Strategy 1: Conservative hybrid (user top-4 + chip top-1)
# Keeps most of user's strong picks, only replaces the lowest-confidence position
print("\n[1/2] Submission 1: Conservative Hybrid (user top-4 + chip top-1)")
sub1 = user_sub.copy()
n_replaced = 0
for idx, row in sub1.iterrows():
    pid = row.pair_id
    user_hands = [row[f'evidence_hand_{k+1}'] for k in range(5)]
    keep_set = set(user_hands[:4])  # keep first 4
    chip_hands = chip_top10.get(pid, [])
    new_chip = [h for h in chip_hands if h not in keep_set][:1]  # 1 new
    new_ev = user_hands[:4] + new_chip
    while len(new_ev) < 5:
        new_ev.append('NO_EVIDENCE')
    for k in range(5):
        sub1.at[idx, f'evidence_hand_{k+1}'] = new_ev[k]
    if new_chip:
        n_replaced += 1

sub1.to_csv(OUT / "sub_01_conservative_4plus1.csv", index=False)
print(f"  Saved sub_01_conservative_4plus1.csv: {sub1.shape}")
print(f"  Replaced hand 5 in {n_replaced:,} pairs")

# Strategy 2: Targeted fix for soft_play pairs only
# For soft_play pairs (132 in dev, ~9K in eval): replace all 5 with passive top-5
# For other pairs: keep user's original unchanged
print("\n[2/2] Submission 2: Targeted fix for soft_play pairs (passive evidence)")
sub2 = user_sub.copy()
n_soft_play_changed = 0
for idx, row in sub2.iterrows():
    if row.predicted_behavior != 'soft_play':
        continue
    pid = row.pair_id
    passive = passive_top10.get(pid, [])
    if not passive:
        continue
    for k in range(5):
        if k < len(passive):
            sub2.at[idx, f'evidence_hand_{k+1}'] = passive[k]
        else:
            sub2.at[idx, f'evidence_hand_{k+1}'] = 'NO_EVIDENCE'
    n_soft_play_changed += 1

sub2.to_csv(OUT / "sub_02_targeted_soft_play.csv", index=False)
print(f"  Saved sub_02_targeted_soft_play.csv: {sub2.shape}")
print(f"  Changed evidence for {n_soft_play_changed:,} soft_play pairs")

# =========================================================================
# Compute overlaps with user's baseline (for sanity check)
# =========================================================================
print("\n" + "="*80)
print("Overlap with user's baseline (per pair, max 5):")
print("="*80)

overlap_1 = 0
overlap_2 = 0
for idx, row in user_sub.iterrows():
    pid = row.pair_id
    user_set = {row[c] for c in evidence_cols}
    sub1_row = sub1[sub1.pair_id == pid].iloc[0]
    sub2_row = sub2[sub2.pair_id == pid].iloc[0]
    sub1_set = {sub1_row[c] for c in evidence_cols}
    sub2_set = {sub2_row[c] for c in evidence_cols}
    overlap_1 += len(user_set & sub1_set)
    overlap_2 += len(user_set & sub2_set)

print(f"  Submission 1 (conservative): {overlap_1/len(user_sub):.2f}/5.0 ({100*overlap_1/len(user_sub)/5:.1f}%)")
print(f"  Submission 2 (targeted soft_play): {overlap_2/len(user_sub):.2f}/5.0 ({100*overlap_2/len(user_sub)/5:.1f}%)")

print(f"\n{'='*80}")
print(f"DONE in {time.time()-t0:.1f}s")
print(f"{'='*80}")
print(f"\nFiles saved:")
print(f"  - {OUT/'evidence_cache.json'} ({cache_size_mb:.1f} MB)")
print(f"  - {OUT/'sub_01_conservative_4plus1.csv'} ({sub1.shape})")
print(f"  - {OUT/'sub_02_targeted_soft_play.csv'} ({sub2.shape})")
