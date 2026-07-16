"""H2 FINAL - wick-rejection zone scoring with pivot-seeded zones.

Causal single pass over 15m bars; bounded memory; Pine v6-portable.

Rule
----
1. ATR = 96-bar SMA of true range.
2. Zone birth: only at confirmed swing pivots (extreme of piv_left=20 bars back,
   confirmed piv_right=5 bars later). A pivot joins the nearest existing zone
   within tol=0.12% of price, else creates a new zone (max 400 zones; weakest
   evicted). Each zone keeps a small array of its pivots (price, strength).
3. Rejection events per zone (deduped: >= gap=16 bars apart):
     a) pivot confirmation itself,
     b) wick rejection: upper/lower wick >= 0.6*ATR poking a 12-bar extreme,
        landing within tol of the zone,
     c) close-back crossing: bar range crosses the zone anchor but open and
        close stay on the approach side.
   Every event opens a 16-bar reaction window; its weight = max excursion away
   from the level in event-time ATR units, capped at 8. Events with weight
   < 0.3 are discarded.
4. Zone score decays with a 240-day half-life (applied per bar incrementally);
   each event with weight >= 1.0 adds +1 at event time. So score ~ decayed
   count of distinct, confirmed rejections.
5. Emit at data end: zones with score >= 14, greedy non-max suppression by
   score with 0.4% min separation. Level price = event anchor: the touch price
   of the zone's strongest reaction event, magnet-snapped to the nearest zone
   pivot price if one lies within 0.08%.
"""
import bisect
import json
import sys

import numpy as np
import pandas as pd

DATA = "../mnq_m15.csv"
OUT = "./detected.json"

P = dict(
    atr_len=96,
    piv_left=20, piv_right=5,
    wick_atr=0.6, loc_k=12,
    tol_frac=0.0012,
    gap_bars=16, react_bars=16,
    w_keep=0.3,          # record event (eligible as anchor) if weight >= this
    w_count=1.0,         # event adds +1 to score if weight >= this
    w_cap=8.0,
    half_life_days=240.0,
    s_min=14.0,
    sep_frac=0.004,
    snap_frac=0.0008,    # magnet-snap ee anchor to pivot within this
    max_zones=400, max_pivots=40,
)
BPD = 96
DECAY = 0.5 ** (1.0 / (P["half_life_days"] * BPD))   # per-bar decay factor


class Zone:
    __slots__ = ("anchor", "score", "count", "below", "above", "last_event_i",
                 "created_i", "best_strength", "pivots",
                 "best_w", "best_i", "best_price",
                 "pend_i", "pend_side", "pend_exc", "pend_atr", "pend_price")

    def __init__(self, price, i, strength):
        self.anchor = price          # strongest-pivot price (zone key)
        self.score = 0.0             # decayed rejection score
        self.count = 0
        self.below = 0
        self.above = 0
        self.last_event_i = i
        self.created_i = i
        self.best_strength = strength
        self.pivots = [(price, strength)]
        self.best_w = 0.0            # strongest reaction event so far
        self.best_i = -1
        self.best_price = price
        self.pend_i = -1
        self.pend_side = 0
        self.pend_exc = 0.0
        self.pend_atr = 1.0
        self.pend_price = price


def detect():
    df = pd.read_csv(DATA)
    o = df.open.values
    h = df.high.values
    lo = df.low.values
    c = df.close.values
    n = len(df)

    tr = np.maximum(h - lo, np.maximum(abs(h - np.roll(c, 1)),
                                       abs(lo - np.roll(c, 1))))
    tr[0] = h[0] - lo[0]
    atr = pd.Series(tr).rolling(P["atr_len"], min_periods=1).mean().values

    pl, pr = P["piv_left"], P["piv_right"]
    win = pl + pr + 1
    hs, ls = pd.Series(h), pd.Series(lo)
    roll_max = hs.rolling(win, min_periods=1).max().shift(-pr).values
    roll_min = ls.rolling(win, min_periods=1).min().shift(-pr).values
    piv_hi = np.zeros(n, dtype=bool)
    piv_lo = np.zeros(n, dtype=bool)
    piv_hi[:n - pr] = h[:n - pr] == roll_max[:n - pr]
    piv_lo[:n - pr] = lo[:n - pr] == roll_min[:n - pr]

    k = P["loc_k"]
    hh = hs.rolling(k, min_periods=1).max().values
    ll = ls.rolling(k, min_periods=1).min().values
    body_hi = np.maximum(o, c)
    body_lo = np.minimum(o, c)
    up_ev = (h - body_hi >= P["wick_atr"] * atr) & (h >= hh)
    dn_ev = (body_lo - lo >= P["wick_atr"] * atr) & (lo <= ll)

    zones, keys, zlist, pending = [], [], [], []

    def find_zone(price, tol):
        j = bisect.bisect_left(keys, price)
        best, bd = None, tol
        for jj in (j - 1, j):
            if 0 <= jj < len(keys):
                d = abs(keys[jj] - price)
                if d <= bd:
                    best, bd = zlist[jj], d
        return best

    def remove_sorted(z, a):
        j = bisect.bisect_left(keys, a)
        while j < len(keys) and keys[j] == a:
            if zlist[j] is z:
                keys.pop(j)
                zlist.pop(j)
                return
            j += 1

    def insert_sorted(z):
        j = bisect.bisect_left(keys, z.anchor)
        keys.insert(j, z.anchor)
        zlist.insert(j, z)

    def open_pending(z, side, i, price):
        if z.pend_i >= 0:
            return
        z.pend_i, z.pend_side = i, side
        z.pend_exc, z.pend_atr = 0.0, max(atr[i], 1e-9)
        z.pend_price = price
        pending.append(z)

    def start_event(z, side_label, i, price):
        if i - z.last_event_i < P["gap_bars"]:
            return
        z.last_event_i = i
        if side_label == "below":
            z.below += 1
            open_pending(z, +1, i, price)
        else:
            z.above += 1
            open_pending(z, -1, i, price)

    def seed_pivot(price, side_label, i, strength):
        tol = P["tol_frac"] * price
        z = find_zone(price, tol)
        if z is None:
            if len(zones) >= P["max_zones"]:
                w = min(zones, key=lambda q: (q.score, q.last_event_i))
                zones.remove(w)
                remove_sorted(w, w.anchor)
                if w in pending:
                    pending.remove(w)
            z = Zone(price, i, strength)
            zones.append(z)
            insert_sorted(z)
            z.last_event_i = i
            if side_label == "below":
                z.below += 1
                open_pending(z, +1, i, price)
            else:
                z.above += 1
                open_pending(z, -1, i, price)
        else:
            if len(z.pivots) >= P["max_pivots"]:
                z.pivots.pop(0)
            z.pivots.append((price, strength))
            if strength >= z.best_strength:
                old = z.anchor
                z.anchor = price
                z.best_strength = strength
                if z.anchor != old:
                    remove_sorted(z, old)
                    insert_sorted(z)
            start_event(z, side_label, i, price)

    for i in range(n):
        # incremental per-bar score decay (skipped for exact-zero scores)
        for z in zones:
            if z.score:
                z.score *= DECAY

        # close/extend pending reaction windows
        if pending:
            done = []
            for z in pending:
                exc = (z.anchor - lo[i]) if z.pend_side > 0 else (h[i] - z.anchor)
                if exc > z.pend_exc:
                    z.pend_exc = exc
                if i - z.pend_i >= P["react_bars"]:
                    w = min(z.pend_exc / z.pend_atr, P["w_cap"])
                    if w >= P["w_keep"]:
                        if (w, z.pend_i) >= (z.best_w, z.best_i):
                            z.best_w, z.best_i = w, z.pend_i
                            z.best_price = z.pend_price
                        if w >= P["w_count"]:
                            z.score += 1.0
                            z.count += 1
                    z.pend_i = -1
                    done.append(z)
            for z in done:
                pending.remove(z)

        # pivot confirmations
        j = i - pr
        if j >= 0:
            if piv_hi[j]:
                s = 0
                jj = j - 1
                while jj >= 0 and s < 400 and h[jj] <= h[j]:
                    s += 1
                    jj -= 1
                seed_pivot(h[j], "below", i, s)
            if piv_lo[j]:
                s = 0
                jj = j - 1
                while jj >= 0 and s < 400 and lo[jj] >= lo[j]:
                    s += 1
                    jj -= 1
                seed_pivot(lo[j], "above", i, s)

        # wick rejections
        if up_ev[i]:
            z = find_zone(h[i], P["tol_frac"] * h[i])
            if z is not None:
                start_event(z, "below", i, h[i])
        if dn_ev[i]:
            z = find_zone(lo[i], P["tol_frac"] * lo[i])
            if z is not None:
                start_event(z, "above", i, lo[i])

        # close-back crossings
        if keys:
            jl = bisect.bisect_left(keys, lo[i])
            jr = bisect.bisect_right(keys, h[i])
            for jj in range(jl, min(jr, jl + 8)):
                a, z = keys[jj], zlist[jj]
                if c[i] < a and o[i] < a and h[i] >= a:
                    start_event(z, "below", i, h[i])
                elif c[i] > a and o[i] > a and lo[i] <= a:
                    start_event(z, "above", i, lo[i])

    # ---- emit active levels ----
    def level_price(z):
        ee = z.best_price if z.best_i >= 0 else z.anchor
        snap = P["snap_frac"] * ee
        best = None
        for p, s in z.pivots:
            d = abs(p - ee)
            if d <= snap and (best is None or d < best[0]):
                best = (d, p)
        return best[1] if best else ee

    cand = [(z.score, z) for z in zones if z.score >= P["s_min"]]
    cand.sort(key=lambda t: -t[0])
    kept = []
    for s, z in cand:
        a = level_price(z)
        if all(abs(a - a2) > P["sep_frac"] * a for a2, _ in kept):
            kept.append((a, z))
    kept.sort()
    return kept


if __name__ == "__main__":
    kept = detect()
    levels = [a for a, _ in kept]
    json.dump({"name": "H2 wick-rejection zones", "levels": levels},
              open(OUT, "w"), indent=1)
    print(f"{len(levels)} active levels at data end:")
    for a, z in kept:
        print(f"  {a:>9.2f} score={z.score:>6.2f} events={z.count:>3} "
              f"below={z.below:>3} above={z.above:>3}")
