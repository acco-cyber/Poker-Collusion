"""
Local test version of the Day 2 notebook — uses local paths.
Run this to verify the script works before uploading to Kaggle.
"""
import os, gc, time, json, heapq
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import polars as pl

# Local paths (for testing)
DATA = Path("/home/z/my-project/kaggle/comp-data")
USER_SUB_PATH = Path("/home/z/my-project/kaggle/user-submissions/pu-aware-output/submission.csv")
OUT  = Path("/home/z/my-project/kaggle/runs/day2")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260918
np.random.seed(SEED)

print("="*80)
print("Local test: Day 2 Multi-Strategy Suite")
print("="*80)
t0 = time.time()

# Check inputs
print(f"Data dir exists: {DATA.exists()}")
print(f"User submission exists: {USER_SUB_PATH.exists()}")

if not DATA.exists():
    print("\nERROR: Competition data not found. Run:")
    print("  kaggle competitions download detect-suspicious-value-transfers-in-poker")
    print("  unzip detect-suspicious-value-transfers-in-poker.zip -d /home/z/my-project/kaggle/comp-data/")
    raise FileNotFoundError("Competition data missing")

if not USER_SUB_PATH.exists():
    print("\nERROR: User's previous submission not found. Run:")
    print("  kaggle kernels output kragglenote2forwork/poker-collusion-pu-aware-evidence-ranker")
    print("    -p /home/z/my-project/kaggle/user-submissions/pu-aware-output")
    raise FileNotFoundError("User submission missing")

user_sub = pd.read_csv(USER_SUB_PATH)
print(f"\nUser submission loaded: {user_sub.shape}")

# Load eval pairs
eval_pairs = pd.read_csv(DATA / "evaluation_pairs.csv")
print(f"Eval pairs: {eval_pairs.shape}")

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

# Stream seats
print("\nStreaming seats.parquet...")
pair_evidence = defaultdict(lambda: {"transfer": [], "passive": [], "aggressive": []})
pf = pq.ParquetFile(DATA / "seats.parquet")
print(f"seats.parquet: {pf.metadata.num_rows:,} rows, {pf.num_row_groups} row groups")
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

print(f"Done in {time.time()-t_stream:.1f}s. Pairs with evidence: {len(pair_evidence):,}")
del hands_dict
gc.collect()

# Build top-10 per pair per signal
print("\nBuilding top-10 hands per pair per signal...")
chip_top10 = {}
passive_top10 = {}
aggressive_top10 = {}

for pid, ev in pair_evidence.items():
    chip_top10[pid] = [h for s, h in sorted(ev["transfer"], key=lambda x: -x[0])[:10]]
    passive_top10[pid] = [h for s, h in sorted(ev["passive"], key=lambda x: -x[0])[:10]]
    aggressive_top10[pid] = [h for s, h in sorted(ev["aggressive"], key=lambda x: -x[0])[:10]]

print(f"Built top-10 for {len(chip_top10):,} pairs")
del pair_evidence
gc.collect()

# Build 5 submission CSVs
print("\nBuilding 5 submission CSVs...")

def build_hybrid(user_sub, n_user_hands, chip_hands_map):
    hybrid = user_sub.copy()
    for idx, row in hybrid.iterrows():
        pid = row.pair_id
        user_hands = [row[f'evidence_hand_{k+1}'] for k in range(5)]
        keep_set = set(user_hands[:n_user_hands])
        chip_hands = chip_hands_map.get(pid, [])
        new_chip = [h for h in chip_hands if h not in keep_set][:5-n_user_hands]
        new_ev = user_hands[:n_user_hands] + new_chip
        while len(new_ev) < 5:
            if len(user_hands) > len(new_ev):
                new_ev.append(user_hands[len(new_ev)])
            else:
                new_ev.append('NO_EVIDENCE')
        for k in range(5):
            hybrid.at[idx, f'evidence_hand_{k+1}'] = new_ev[k]
    return hybrid

def build_passive_for_soft_play(user_sub, passive_map):
    sub = user_sub.copy()
    for idx, row in sub.iterrows():
        if row.predicted_behavior != 'soft_play':
            continue
        pid = row.pair_id
        passive = passive_map.get(pid, [])
        if not passive:
            continue
        for k in range(5):
            if k < len(passive):
                sub.at[idx, f'evidence_hand_{k+1}'] = passive[k]
            else:
                sub.at[idx, f'evidence_hand_{k+1}'] = 'NO_EVIDENCE'
    return sub

# Strategy 1
sub1 = user_sub.copy()
sub1.to_csv(OUT / "submission_01_user_baseline.csv", index=False)
print(f"  [1/5] submission_01_user_baseline.csv: {sub1.shape}")

# Strategy 2
sub2 = build_hybrid(user_sub, n_user_hands=3, chip_hands_map=chip_top10)
sub2.to_csv(OUT / "submission_02_hybrid_3plus2.csv", index=False)
print(f"  [2/5] submission_02_hybrid_3plus2.csv: {sub2.shape}")

# Strategy 3
sub3 = build_hybrid(user_sub, n_user_hands=4, chip_hands_map=chip_top10)
sub3.to_csv(OUT / "submission_03_hybrid_4plus1.csv", index=False)
print(f"  [3/5] submission_03_hybrid_4plus1.csv: {sub3.shape}")

# Strategy 4
sub4 = build_hybrid(user_sub, n_user_hands=2, chip_hands_map=chip_top10)
sub4.to_csv(OUT / "submission_04_hybrid_2plus3.csv", index=False)
print(f"  [4/5] submission_04_hybrid_2plus3.csv: {sub4.shape}")

# Strategy 5
sub5 = build_passive_for_soft_play(user_sub, passive_top10)
sub5.to_csv(OUT / "submission_05_passive_soft_play.csv", index=False)
print(f"  [5/5] submission_05_passive_soft_play.csv: {sub5.shape}")

print(f"\n{'='*80}")
print(f"DONE in {time.time()-t0:.1f}s")
print(f"{'='*80}")
print(f"\nFiles saved to: {OUT}")
for f in sorted(OUT.glob("submission_*.csv")):
    print(f"  {f.name} ({f.stat().st_size/1e6:.1f} MB)")
