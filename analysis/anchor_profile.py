"""Profile the anchor swing behind each user level + retest lifecycle.

For each level:
  - find the first bar whose exact high/low == level (the anchor candidate)
  - measure its pivot strength: how many bars before/after stayed below (for a high)
    or above (for a low) that price
  - lifecycle: distinct retest events, from which side, and whether the level
    role-flipped (rejected price both as support and as resistance)
"""
import pandas as pd
import numpy as np
import json

import os
OUT = os.path.dirname(os.path.abspath(__file__))
LEVELS = [30886.75, 30539.50, 30297.25, 30007.00, 29753.75, 29406.00, 28798.00,
          27661.00, 27501.75, 26638.75, 25891.75, 25123.50, 25052.50, 24953.00,
          24417.50, 22961.50, 22731.75]

m15 = pd.read_csv(f"{OUT}/mnq_m15.csv", parse_dates=["ts"])
H, L_, C = m15["high"].values, m15["low"].values, m15["close"].values
TS = m15["ts"]
n = len(m15)

def pivot_strength(i, price, kind):
    """bars on each side that did not exceed price (capped at 2000)."""
    left = 0
    j = i - 1
    while j >= 0 and left < 2000:
        if (kind == "high" and H[j] > price) or (kind == "low" and L_[j] < price):
            break
        left += 1; j -= 1
    right = 0
    j = i + 1
    while j < n and right < 2000:
        if (kind == "high" and H[j] > price) or (kind == "low" and L_[j] < price):
            break
        right += 1; j += 1
    return left, right

profiles = []
for lvl in LEVELS:
    cands = []
    for kind, arr in [("high", H), ("low", L_)]:
        for i in np.where(np.isclose(arr, lvl))[0]:
            lft, rgt = pivot_strength(i, lvl, kind)
            cands.append({"i": int(i), "ts": str(TS.iloc[i])[:16], "kind": kind,
                          "left": lft, "right": rgt, "score": min(lft, rgt)})
    # no exact match: nearest extreme within 0.05%
    if not cands:
        tol = lvl * 0.0005
        for kind, arr in [("high", H), ("low", L_)]:
            close_idx = np.where(np.abs(arr - lvl) <= tol)[0]
            for i in close_idx:
                lft, rgt = pivot_strength(i, float(arr[i]), kind)
                if min(lft, rgt) >= 12:
                    cands.append({"i": int(i), "ts": str(TS.iloc[i])[:16], "kind": kind,
                                  "px": float(arr[i]), "left": lft, "right": rgt,
                                  "score": min(lft, rgt)})
    cands.sort(key=lambda d: -d["score"])
    best = cands[0] if cands else None

    # retest lifecycle at ±0.05%
    tol = lvl * 0.0005
    in_band = (H >= lvl - tol) & (L_ <= lvl + tol)
    idx = np.where(in_band)[0]
    events = []
    if len(idx):
        start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i - prev > 16:  # >4h gap = new event
                events.append((start, prev))
                start = i
            prev = i
        events.append((start, prev))
    # classify each event by approach side (close 8 bars before event start)
    ev_sides = []
    for s, e in events:
        ref = C[max(0, s - 8)]
        side = "from_below" if ref < lvl else "from_above"
        # outcome: side of close 8 bars after event end
        out_ref = C[min(n - 1, e + 8)]
        outcome = "rejected" if (side == "from_below" and out_ref < lvl) or \
                               (side == "from_above" and out_ref > lvl) else "broke"
        ev_sides.append((side, outcome))
    n_below = sum(1 for s, o in ev_sides if s == "from_below")
    n_above = sum(1 for s, o in ev_sides if s == "from_above")
    n_rej = sum(1 for s, o in ev_sides if o == "rejected")

    profiles.append({
        "level": lvl,
        "anchor": best,
        "n_events": len(events),
        "approach_below": n_below,
        "approach_above": n_above,
        "rejections": n_rej,
        "reject_rate": round(n_rej / len(events), 2) if events else None,
        "n_anchor_candidates_20plus": sum(1 for cd in cands if cd["score"] >= 20),
    })

for p in profiles:
    a = p["anchor"]
    astr = (f"{a['kind']:>4} @ {a['ts']}  strength L{a['left']}/R{a['right']}"
            + (f" px={a.get('px')}" if a and a.get("px") else "")) if a else "NONE"
    print(f"{p['level']:>9}: anchor {astr:<58} | events={p['n_events']:>3} "
          f"below={p['approach_below']:>3} above={p['approach_above']:>3} "
          f"rejected={p['rejections']:>3} ({p['reject_rate']})")

with open(f"{OUT}/anchor_profiles.json", "w") as f:
    json.dump(profiles, f, indent=1, default=str)
print("\nsaved anchor_profiles.json")
