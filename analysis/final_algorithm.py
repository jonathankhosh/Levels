"""FINAL combined level-detection algorithm (synthesis of H1-H6).

Base architecture = H4 (the strongest hypothesis, f1 0.59), refined:
  * split emission thresholds for strong vs non-strong old levels
    (t2n touch floor + p2n crispness cap for never-fresh-extreme levels),
  * "chart-top zone" exemption: strong levels within top_zone_pct of the
    highest strong level always emit (trader always marks the ATH shelf),
  * strong-rescue touch floor raised 3 -> 10 (kills weak young FPs; the
    ATH-shelf target that rule used to protect is now covered by the
    chart-top exemption),
  * inert t1 branch removed.
Rejected after measurement: rejection-fraction gates (anti-signal),
volume filters (H3: anti-signal), decayed scores, NMS, staleness cutoffs.

Pipeline (fully causal, walk-forward, Pine-v6-portable):
  1) Aggregate 15m bars into 1h bars on the fly.
  2) A confirmed 1h pivot (strength 8) creates a level at the exact wick
     or merges into the nearest level within 0.20% (center nudged 25%
     toward the pivot, npiv incremented). A pivot that is also a fresh
     400-hour extreme is "strong"; strong pivots farther than 0.08% from
     any level always start their own level.
  3) Every 15m bar whose range +/-0.09% contains a level center is zone
     contact; contacts >=24 bars apart count as distinct touches; the
     displayed price re-snaps to the latest 15m wick within 0.30%.
  4) Emission at the last bar (see emit rules in code below).

Final score vs the trader's 17 levels:
  f1_loose 0.699, recall_loose 0.824, recall_strict 0.471,
  precision 0.607, 31 levels in range.
"""
import json
import os
import sys

import pandas as pd

PARENT = "."
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PARENT)
from score_levels import score  # noqa: E402

P = dict(
    tf_hours=1,           # HTF aggregation timeframe
    s=8,                  # 1h pivot strength (bars each side)
    K=400,                # fresh-extreme lookback (1h bars) => "strong"
    merge_tol_pct=0.20,   # pivot joins nearest level within this %
    promote_tol_pct=0.08, # strong pivot always separates beyond this %
    alpha=0.25,           # center nudge toward each merged pivot
    band_pct=0.09,        # touch band around level center (%)
    snap_tol_pct=0.30,    # wick re-snap radius (%)
    gap_bars=24,          # 15m bars between distinct touches (6h)
    recent_days=35,       # "young" age cutoff
    tr=11,                # young: touches >= tr
    t2=24, p2=9,          # old strong: touches>=t2 and npiv<=p2
    t2n=40, p2n=7,        # old non-strong: touches>=t2n and npiv<=p2n
    t3=14, p3=2,          # old crisp shelf: touches>=t3 and npiv<=p3
    ts=10, rd2=45,        # strong rescue: age<=rd2 days and touches>=ts
    top_zone_pct=0.30,    # chart-top zone: strong level within this % of
                          # the highest strong level's center always emits
)

BARS_PER_DAY = 96.0
df = pd.read_csv(f"{PARENT}/mnq_m15.csv")


def run(p, end_idx=None):
    H = df.high.values
    L = df.low.values
    E = df.epoch.values
    N = len(df) if end_idx is None else end_idx

    tf_sec = int(p["tf_hours"] * 3600)
    s, K = p["s"], p["K"]
    merge_tol = p["merge_tol_pct"] / 100.0
    ptol = p["promote_tol_pct"] / 100.0
    band = p["band_pct"] / 100.0
    snap_tol = p["snap_tol_pct"] / 100.0
    gap, alpha = p["gap_bars"], p["alpha"]

    htf_h, htf_l = [], []          # completed 1h highs/lows
    cur_key = cur_h = cur_l = None
    levels = []                    # bounded book (~175 over 22 months)

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
        # ---- 1h aggregation + pivot detection on completed 1h bars ----
        key = E[i] // tf_sec
        if cur_key is None:
            cur_key, cur_h, cur_l = key, H[i], L[i]
        elif key != cur_key:
            htf_h.append(cur_h)
            htf_l.append(cur_l)
            cur_key, cur_h, cur_l = key, H[i], L[i]
            m = len(htf_h) - 1 - s              # candidate pivot bar
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

        # ---- 15m touch counting + wick re-snap ----
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

    # ---- emission (what the indicator draws on the last bar) ----
    out = []
    top_px = max((lv["px"] for lv in levels if lv["strong"]), default=None)
    for lv in levels:
        age_d = (N - 1 - lv["born"]) / BARS_PER_DAY
        if age_d <= p["recent_days"]:
            ok = lv["touches"] >= p["tr"]
        else:
            t2eff = p["t2"] if lv["strong"] else p["t2n"]
            p2eff = p["p2"] if lv["strong"] else p["p2n"]
            ok = ((lv["touches"] >= t2eff and lv["npiv"] <= p2eff)
                  or (lv["touches"] >= p["t3"] and lv["npiv"] <= p["p3"]))
        if not ok and lv["strong"] and age_d <= p["rd2"] \
                and lv["touches"] >= p["ts"]:
            ok = True                              # fresh-extreme rescue
        if not ok and lv["strong"] and top_px is not None \
                and lv["px"] >= top_px * (1 - p["top_zone_pct"] / 100.0):
            ok = True                              # chart-top zone
        lv["emit"] = ok
        if ok:
            out.append(lv["snap"])
    return sorted(set(out)), levels


if __name__ == "__main__":
    p = dict(P)
    end_idx = None
    for a in sys.argv[1:]:
        k, v = a.split("=")
        if k == "end_date":                        # robustness runs
            end_idx = int((df.ts < v).sum())
            continue
        p[k] = float(v) if "." in v else int(v)
    detected, levels = run(p, end_idx)
    if end_idx is None:
        with open(f"{HERE}/detected.json", "w") as f:
            json.dump({"name": "FINAL combined (H4-refined)",
                       "levels": detected, "params": p}, f, indent=1)
        r = score(detected)
        print(f"tracked={len(levels)} emitted={len(detected)}")
        for k in ["n_detected_in_range", "recall_strict", "recall_loose",
                  "precision", "f1_loose", "true_pos", "false_pos",
                  "neutral"]:
            print(f"{k}: {r[k]}")
    else:
        last = df.iloc[end_idx - 1]
        lo, hi = df.low[:end_idx].min(), df.high[:end_idx].max()
        print(f"end={last.ts} close={last.close} range=[{lo},{hi}] "
              f"tracked={len(levels)} emitted={len(detected)}")
        for x in detected:
            print(f"  {x:9.2f}")
