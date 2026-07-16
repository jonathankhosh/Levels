"""H4 HTF extremes — final algorithm (causal, Pine-v6-portable).

Walk forward over 15m bars, aggregating 1h HTF bars on the fly.

1) LEVEL CREATION (HTF pivots):
   When a 1h pivot high/low of strength S=8 confirms (8 closed 1h bars on each
   side), it becomes a level event:
   - "strong" if the pivot price is also a fresh K=400-hour extreme
     (higher than every 1h high / lower than every 1h low of the prior 400 bars).
   - Merge into the nearest existing level if within MERGE_TOL=0.20% of it
     (center updated by alpha=0.25 toward the pivot; npiv incremented), EXCEPT
     a strong pivot farther than PROMOTE_TOL=0.08% from any level always starts
     its own level (fresh multi-week extremes deserve their own line).
   - Otherwise create a new level at the pivot price (an exact candle wick).

2) LEVEL LIFE (15m touches + wick snap):
   Every 15m bar whose range extended by BAND=0.09% contains a level center
   counts as zone contact; contacts separated by >= GAP=24 bars (6h) outside
   the zone are distinct touch events. The displayed price ("snap") follows
   the most recent 15m wick within SNAP_TOL=0.30% of the center — mimicking
   the trader re-anchoring the line to the latest reaction wick (16/17 target
   levels are exact 15m wicks). Levels are never hard-deleted on break:
   the targets are role-flip levels; selection happens at emission.

3) EMISSION (active lines at data end):
   age <= 35 days:            touches >= 11
   age  > 35 days:            touches >= 85
                              or (touches >= 24 and npiv <= 9)   # strong & crisp
                              or (touches >= 14 and npiv <= 2)   # crisp shelf
   plus, any age <= 45 days:  strong level with touches >= 3     # fresh extreme
   ("crisp" = few distinct merged pivots: the zone respects one price instead
   of spraying new pivots; chop zones have npiv >= 10.)

Pine port notes: keep only the last 2S+1 completed 1h H/L for pivot checks and
the last K for the extreme check (rolling arrays); levels are a bounded array
(~300 over 22 months; prune far-away weakest if needed).
"""
import json
import os
import sys

import pandas as pd

PARENT = ".."
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PARENT)
from score_levels import score  # noqa: E402

P = dict(
    tf_hours=1, s=8, K=400,
    merge_tol_pct=0.20, promote_tol_pct=0.08, alpha=0.25,
    band_pct=0.09, snap_tol_pct=0.30, gap_bars=24,
    recent_days=35, tr=11,
    t1=85, t2=24, p2=9, t3=14, p3=2,
    ts=3, rd2=45,
)

df = pd.read_csv(f"{PARENT}/mnq_m15.csv")
H = df.high.values
L = df.low.values
E = df.epoch.values
N = len(df)
BARS_PER_DAY = 96.0


def run(p):
    tf_sec = int(p["tf_hours"] * 3600)
    s, K = p["s"], p["K"]
    merge_tol = p["merge_tol_pct"] / 100.0
    ptol = p["promote_tol_pct"] / 100.0
    band = p["band_pct"] / 100.0
    snap_tol = p["snap_tol_pct"] / 100.0
    gap, alpha = p["gap_bars"], p["alpha"]

    htf_h, htf_l = [], []
    cur_key = cur_h = cur_l = None
    levels = []

    def add_pivot(px, i, strong):
        best, bd = None, 1e18
        for lv in levels:
            d = abs(px - lv["px"])
            if d <= lv["px"] * merge_tol and d < bd:
                best, bd = lv, d
        if strong and (best is None or bd > px * ptol):
            levels.append(dict(px=px, snap=px, touches=1, born=i,
                               last_in=i, npiv=1, strong=True))
            return
        if best is not None:
            best["px"] += alpha * (px - best["px"])
            best["npiv"] += 1
            if strong:
                best["strong"] = True
        else:
            levels.append(dict(px=px, snap=px, touches=1, born=i,
                               last_in=i, npiv=1, strong=strong))

    for i in range(N):
        key = E[i] // tf_sec
        if cur_key is None:
            cur_key, cur_h, cur_l = key, H[i], L[i]
        elif key != cur_key:
            htf_h.append(cur_h)
            htf_l.append(cur_l)
            cur_key, cur_h, cur_l = key, H[i], L[i]
            m = len(htf_h) - 1 - s
            if m >= s:
                wh = htf_h[m - s:m + s + 1]
                if htf_h[m] >= max(wh) and wh.count(htf_h[m]) == 1:
                    strong = htf_h[m] > max(htf_h[max(0, m - K):m])
                    add_pivot(htf_h[m], i, strong)
                wl = htf_l[m - s:m + s + 1]
                if htf_l[m] <= min(wl) and wl.count(htf_l[m]) == 1:
                    strong = htf_l[m] < min(htf_l[max(0, m - K):m])
                    add_pivot(htf_l[m], i, strong)
        else:
            if H[i] > cur_h:
                cur_h = H[i]
            if L[i] < cur_l:
                cur_l = L[i]

        hi, lo = H[i], L[i]
        for lv in levels:
            d = lv["px"] * band
            if lo - d <= lv["px"] <= hi + d:
                if i - lv["last_in"] >= gap:
                    lv["touches"] += 1
                lv["last_in"] = i
                w = hi if abs(hi - lv["px"]) < abs(lo - lv["px"]) else lo
                if abs(w - lv["px"]) <= lv["px"] * snap_tol:
                    lv["snap"] = w

    out = []
    for lv in levels:
        age_d = (N - 1 - lv["born"]) / BARS_PER_DAY
        if age_d <= p["recent_days"]:
            ok = lv["touches"] >= p["tr"]
        else:
            ok = (lv["touches"] >= p["t1"]
                  or (lv["touches"] >= p["t2"] and lv["npiv"] <= p["p2"])
                  or (lv["touches"] >= p["t3"] and lv["npiv"] <= p["p3"]))
        if not ok and lv.get("strong") and age_d <= p["rd2"] \
                and lv["touches"] >= p["ts"]:
            ok = True
        if ok:
            out.append(lv["snap"])
    return sorted(set(out)), levels


if __name__ == "__main__":
    detected, levels = run(P)
    print(f"total levels tracked: {len(levels)}, emitted: {len(detected)}")
    payload = {"name": "H4 HTF extremes", "levels": detected, "params": P}
    with open(f"{HERE}/detected.json", "w") as f:
        json.dump(payload, f, indent=1)
    r = score(detected)
    for k in ["n_detected_in_range", "recall_strict", "recall_loose",
              "precision", "f1_loose", "true_pos", "false_pos", "neutral"]:
        print(f"{k}: {r[k]}")
