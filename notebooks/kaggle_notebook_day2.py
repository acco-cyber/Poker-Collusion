"""
Kaggle Notebook: Poker Collusion — Day 2 Multi-Strategy Suite
================================================================
Ready-to-run Kaggle notebook for tomorrow (2026-09-18) when user has 5 fresh submissions.

This notebook generates 5 different submission CSVs from 5 different strategies,
so the user can pick the best one or submit them one-by-one to see which works:

  1. submission_01_user_baseline.csv     — User's existing 0.83065 baseline (sanity check)
  2. submission_02_hybrid_3plus2.csv     — User top-3 + chip-transfer top-2 (yesterday's 0.81486)
  3. submission_03_hybrid_4plus1.csv      — User top-4 + chip-transfer top-1 (more conservative)
  4. submission_04_hybrid_2plus3.csv      — User top-2 + chip-transfer top-3 (more aggressive)
  5. submission_05_passive_soft_play.csv  — Replace evidence ONLY for soft_play pairs with passive signal

Strategy:
  - Keep user's risk_score and predicted_behavior unchanged (proven signals)
  - Vary evidence hand selection across 5 strategies
  - Submit each separately to find the sweet spot

Required Kaggle inputs:
  - Competition dataset: detect-suspicious-value-transfers-in-poker
  - User's previous notebook output (the 0.83065 submission CSV) as a Kaggle dataset
"""
import os, gc, time, json, heapq
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import polars as pl

# Kaggle paths (adjust as needed for Kaggle environment)
DATA = Path("/kaggle/input/detect-suspicious-value-transfers-in-poker")
USER_SUB_PATH = Path("/kaggle/input/poker-collusion-pu-aware-evidence-ranker-output/submission.csv")
USER_SUB_FALLBACKS = [
    Path("/kaggle/input/poker-collusion-pu-aware-evidence-ranker/submission.csv"),
    Path("/kaggle/input/user-submission/submission.csv"),
    Path("/kaggle/working/user_submission.csv"),
]
OUT  = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260918
np.random.seed(SEED)

print("="*80)
print("Kaggle Notebook: Day 2 Multi-Strategy Suite")
print("="*80)
t0 = time.time()

# =========================================================================
# Step 1: Find user's previous submission
# =========================================================================
print("\n[1/6] Finding user's previous submission...")
user_sub = None
candidates = [USER_SUB_PATH] + USER_SUB_FALLBACKS
for path in candidates:
    if path.exists():
        print(f"  Found: {path}")
        user_sub = pd.read_csv(path)
        break

if user_sub is None:
    print("  ERROR: User's previous submission not found!")
    print("  Please add it as a Kaggle dataset input.")
    raise FileNotFoundError("User submission CSV not found")

print(f"  User submission loaded: {user_sub.shape}")

# =========================================================================
# Step 2: Load eval pairs and build pair lookup
# =========================================================================
print("\n[2/6] Loading eval pairs...")
eval_pairs = pd.read_csv(DATA / "evaluation_pairs.csv")
print(f"  Eval pairs: {eval_pairs.shape}")

pair_lookup = {}
for r in eval_pairs.itertuples():
    p1, p2 = sorted([r.player_1, r.player_2])
    pair_lookup[(p1, p2)] = r.pair_id

needed_players = set()
for k in pair_lookup.keys():
    needed_players.add(k[0]); needed_players.add(k[1])
print(f"  Needed players: {len(needed_players):,}")

# =========================================================================
# Step 3: Load hands.parquet
# =========================================================================
print("\n[3/6] Loading hands.parquet...")
hands_pl = pl.read_parquet(DATA / "hands.parquet", columns=["hand_id", "big_blind", "final_pot"])
hands_dict = {row["hand_id"]: (int(row["big_blind"]), int(row["final_pot"]))
              for row in hands_pl.iter_rows(named=True)}
del hands_pl
gc.collect()
print(f"  hands_dict: {len(hands_dict):,}")

# =========================================================================
# Step 4: Stream seats.parquet and build evidence candidates per pair
# =========================================================================
print("\n[4/6] Streaming seats.parquet to build evidence candidates...")

def new_evidence_heap():
    return {"transfer": [], "passive": [], "aggressive": []}

pair_evidence = defaultdict(new_evidence_heap)

pf = pq.ParquetFile(DATA / "seats.parquet")
print(f"  seats.parquet: {pf.metadata.num_rows:,} rows, {pf.num_row_groups} row groups")
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

                # Transfer signal
                asymmetry = abs(net1 - net2) / bb
                one_wins = (net1 > 0) != (net2 > 0)
                transfer_score = asymmetry * (2.0 if one_wins else 0.3)

                # Passive signal (soft play)
                passive_score = 0.0
                if showdown1 and showdown2:
                    if final_pot < bb * 4:
                        passive_score = 3.0 - (final_pot / bb / 4.0)
                    elif final_pot < bb * 10:
                        passive_score = 1.0 - (final_pot / bb / 10.0)
                if contrib1 < bb * 2 and contrib2 < bb * 2:
                    passive_score += 0.5

                # Aggressive signal (coordinated isolation)
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
        print(f"    rg {rg_idx+1}/{pf.num_row_groups}: hands={hands_scored:,}, elapsed={elapsed:.1f}s", flush=True)

print(f"  Done in {time.time()-t_stream:.1f}s. Pairs with evidence: {len(pair_evidence):,}")
del hands_dict
gc.collect()

# =========================================================================
# Step 5: Build top-K hands per pair per signal
# =========================================================================
print("\n[5/6] Building top-K hands per pair per signal...")
chip_top10 = {}
passive_top10 = {}
aggressive_top10 = {}

for pid, ev in pair_evidence.items():
    chip_top10[pid] = [h for s, h in sorted(ev["transfer"], key=lambda x: -x[0])[:10]]
    passive_top10[pid] = [h for s, h in sorted(ev["passive"], key=lambda x: -x[0])[:10]]
    aggressive_top10[pid] = [h for s, h in sorted(ev["aggressive"], key=lambda x: -x[0])[:10]]

print(f"  Built top-10 for {len(chip_top10):,} pairs")
del pair_evidence
gc.collect()

# =========================================================================
# Step 6: Build 5 submission CSVs
# =========================================================================
print("\n[6/6] Building 5 submission CSVs...")

def build_hybrid(user_sub, n_user_hands, chip_hands_map):
    """Keep first n_user_hands from user, replace rest with chip hands (excluding dups)."""
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
    """Replace evidence ONLY for soft_play pairs with passive signal."""
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

# Strategy 1: User's baseline (just save as is)
sub1 = user_sub.copy()
sub1.to_csv(OUT / "submission_01_user_baseline.csv", index=False)
print(f"  [1/5] submission_01_user_baseline.csv: {sub1.shape} (sanity check)")

# Strategy 2: Hybrid 3+2 (yesterday's attempt, scored 0.81486)
sub2 = build_hybrid(user_sub, n_user_hands=3, chip_hands_map=chip_top10)
sub2.to_csv(OUT / "submission_02_hybrid_3plus2.csv", index=False)
print(f"  [2/5] submission_02_hybrid_3plus2.csv: {sub2.shape} (yesterday's 0.81486)")

# Strategy 3: Hybrid 4+1 (more conservative)
sub3 = build_hybrid(user_sub, n_user_hands=4, chip_hands_map=chip_top10)
sub3.to_csv(OUT / "submission_03_hybrid_4plus1.csv", index=False)
print(f"  [3/5] submission_03_hybrid_4plus1.csv: {sub3.shape} (more conservative)")

# Strategy 4: Hybrid 2+3 (more aggressive)
sub4 = build_hybrid(user_sub, n_user_hands=2, chip_hands_map=chip_top10)
sub4.to_csv(OUT / "submission_04_hybrid_2plus3.csv", index=False)
print(f"  [4/5] submission_04_hybrid_2plus3.csv: {sub4.shape} (more aggressive)")

# Strategy 5: Passive evidence for soft_play pairs only
sub5 = build_passive_for_soft_play(user_sub, passive_top10)
sub5.to_csv(OUT / "submission_05_passive_soft_play.csv", index=False)
print(f"  [5/5] submission_05_passive_soft_play.csv: {sub5.shape} (passive for soft_play only)")

# =========================================================================
# Summary
# =========================================================================
print("\n" + "="*80)
print("DONE — 5 submission CSVs ready")
print("="*80)

print(f"\nFiles saved to {OUT}:")
for f in sorted(OUT.glob("submission_*.csv")):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name} ({size_mb:.1f} MB)")

print(f"\nTotal runtime: {time.time()-t0:.1f}s")
print(f"\nRecommended submission order for tomorrow (2026-09-18, 5 submissions):")
print(f"  1. submission_01_user_baseline.csv (sanity check — should be 0.83065)")
print(f"  2. submission_03_hybrid_4plus1.csv (most conservative hybrid)")
print(f"  3. submission_05_passive_soft_play.csv (targeted fix for soft_play evidence)")
print(f"  4. submission_02_hybrid_3plus2.csv (yesterday's 0.81486)")
print(f"  5. Save 1 submission for day 3 (don't use all 5 today)")
print(f"\nPick the best from this batch and iterate on day 3.")
