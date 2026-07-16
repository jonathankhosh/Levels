"""H1 FRACTAL PIVOT CLUSTERING — walk-forward S/R level detector.

Causal, Pine-portable design:
  - Detect swing pivot highs/lows on 15m bars at one or more (L,R) strengths.
    A pivot at bar p is CONFIRMED at bar p+R (only closed bars used).
  - Maintain a bounded array of candidate levels. A new confirmed pivot within
    merge_tol (%) of an existing level joins that cluster; otherwise it seeds a
    new level. Each cluster keeps a small bounded list of member prices
    (exact pivot/extreme prices) with per-member touch tallies; the cluster's
    anchor = the member price hit most often (ties -> more recent).
  - Every bar, each level within touch_tol of the bar's range logs an
    interaction; interactions separated by >= gap_bars form distinct touch
    EVENTS (that is the level's strength score).
  - Levels retire when untouched for stale_bars, or decisively broken
    max_breaks times (close flips side by > break_tol).
  - At data end, emit levels with enough evidence: touch_events >= min_events
    (with a lower bar for recently-formed levels), ranked score cap.
"""
import json
import sys
import numpy as np
import pandas as pd

BASE = ".."
sys.path.insert(0, BASE)
from score_levels import score as harness_score  # noqa: E402

TICK = 0.25


def load_m15():
    df = pd.read_csv(f"{BASE}/mnq_m15.csv")
    return df


def pivot_events(high, low, strengths):
    """Return list of (confirm_idx, price, kind, weight) sorted by confirm idx.
    kind: +1 pivot high, -1 pivot low. Causal: confirmed R bars after pivot."""
    n = len(high)
    ev = []
    hs = pd.Series(high)
    ls = pd.Series(low)
    for (L, R, w) in strengths:
        win = L + R + 1
        # rolling max over window [i-L, i+R] evaluated at center i
        rmaxs = hs.rolling(win, min_periods=win).max().shift(-R).values
        rmins = ls.rolling(win, min_periods=win).min().shift(-R).values
        # strict on the right side: high[i] must exceed max(high[i+1..i+R])
        rt_max = hs[::-1].rolling(R, min_periods=1).max()[::-1].shift(-1).values
        rt_min = ls[::-1].rolling(R, min_periods=1).min()[::-1].shift(-1).values
        for i in range(L, n - R):
            if high[i] == rmaxs[i] and high[i] > rt_max[i]:
                ev.append((i + R, high[i], 1, w))
            if low[i] == rmins[i] and low[i] < rt_min[i]:
                ev.append((i + R, low[i], -1, w))
    ev.sort(key=lambda e: e[0])
    return ev


class Level:
    __slots__ = ("prices", "counts", "lastm", "anchor", "n_pivots", "pivot_w",
                 "touch_events", "last_int", "created", "breaks", "side_above",
                 "side_below", "last_break", "retired", "ev_side", "ev_open",
                 "prev_appr", "rejections", "flips", "ev_decay", "rej_decay",
                 "dec_upd")

    def bump_decay(self, bar, hl_bars, ev=0.0, rej=0.0):
        f = 0.5 ** ((bar - self.dec_upd) / hl_bars)
        self.ev_decay = self.ev_decay * f + ev
        self.rej_decay = self.rej_decay * f + rej
        self.dec_upd = bar

    def __init__(self, price, bar, weight):
        self.prices = [price]
        self.counts = [1]
        self.lastm = [bar]
        self.anchor = price
        self.n_pivots = 1
        self.pivot_w = weight
        self.touch_events = 0
        self.last_int = bar
        self.created = bar
        self.breaks = 0
        self.last_break = -10**9
        self.side_above = 0
        self.side_below = 0
        self.retired = False
        self.ev_side = 0      # approach side of currently open touch event
        self.ev_open = False
        self.prev_appr = 0
        self.rejections = 0
        self.flips = 0
        self.ev_decay = 0.0   # exponentially-decayed touch-event count
        self.rej_decay = 0.0
        self.dec_upd = bar    # last bar the decay counters were updated

    def add_member(self, price, bar, max_members, eps):
        for j, p in enumerate(self.prices):
            if abs(p - price) <= eps:
                self.counts[j] += 1
                self.lastm[j] = bar
                self._reanchor()
                return
        if len(self.prices) >= max_members:
            j = int(np.argmin(self.counts))
            self.prices[j] = price
            self.counts[j] = 1
            self.lastm[j] = bar
        else:
            self.prices.append(price)
            self.counts.append(1)
            self.lastm.append(bar)
        self._reanchor()

    def _reanchor(self):
        best = 0
        for j in range(1, len(self.prices)):
            if (self.counts[j], self.lastm[j]) > (self.counts[best],
                                                  self.lastm[best]):
                best = j
        self.anchor = self.prices[best]


def run(df, P, end_idx=None):
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df) if end_idx is None else end_idx
    ev = pivot_events(high[:n], low[:n], P["strengths"])
    ei = 0
    levels = []
    arr = np.empty(0)          # anchor array mirror
    dirty = True

    def rebuild():
        nonlocal arr, dirty
        arr = np.array([lv.anchor for lv in levels]) if levels else np.empty(0)
        dirty = False

    open_events = []
    drawn = []                 # persistent chart lines: [price, draw_bar]
    for i in range(n):
        if dirty:
            rebuild()
        h, l, c = high[i], low[i], close[i]
        cprev = close[i - 1] if i else close[0]
        # resolve touch events whose quiet gap just completed
        if open_events:
            still = []
            for lv in open_events:
                if lv.retired:
                    continue
                if i - lv.last_int >= P["gap_bars"]:
                    res = 1 if c > lv.anchor else -1
                    if res == lv.ev_side:
                        lv.rejections += 1
                        lv.bump_decay(i, P["hl_bars"], rej=1.0)
                    lv.ev_open = False
                else:
                    still.append(lv)
            open_events = still
        if len(arr):
            tol = arr * P["touch_tol"]
            hit = (arr >= l - tol) & (arr <= h + tol)
            idxs = np.nonzero(hit)[0]
            for j in idxs:
                lv = levels[j]
                if i - lv.last_int >= P["gap_bars"]:
                    lv.touch_events += 1
                    lv.bump_decay(i, P["hl_bars"], ev=1.0)
                    appr = 1 if cprev > lv.anchor else -1
                    if lv.prev_appr and appr != lv.prev_appr:
                        lv.flips += 1
                    lv.prev_appr = appr
                    lv.ev_side = appr
                    if not lv.ev_open:
                        lv.ev_open = True
                        open_events.append(lv)
                    # credit nearest member with the exact extreme touched
                    ext = h if abs(h - lv.anchor) <= abs(l - lv.anchor) else l
                    lv.add_member(ext, i, P["max_members"], P["member_eps"]) \
                        if abs(ext - lv.anchor) <= arr[j] * P["merge_tol"] \
                        else None
                    if c < lv.anchor:
                        lv.side_below += 1
                    else:
                        lv.side_above += 1
                    dirty = True  # anchor may have moved
                lv.last_int = i
            # decisive breaks: close beyond level by > break_tol
            btol = arr * P["break_tol"]
            broke = ((c > arr + btol) | (c < arr - btol))
            # only count a break if level was recently in play (interacted
            # within gap) i.e. price actually crossed it around now
            for j in np.nonzero(broke)[0]:
                lv = levels[j]
                if i - lv.last_int <= P["gap_bars"] and \
                        i - lv.last_break >= P["gap_bars"]:
                    if (c > lv.anchor) != (close[max(i - 1, 0)] > lv.anchor) \
                            or i - lv.last_int <= 2:
                        lv.breaks += 1
                        lv.last_break = i
        # chart-drawing simulation: draw/replace hot levels
        if P.get("draw_th") and i % 8 == 0:
            th = P["draw_th"]
            hl = P["hl_bars"]
            rtol = P["replace_tol"]
            for lv in levels:
                if lv.rej_decay < th:      # upper bound before decay
                    continue
                sc = lv.rej_decay * 0.5 ** ((i - lv.dec_upd) / hl)
                if sc < th:
                    continue
                a = lv.anchor
                repl = None
                for d in drawn:
                    if abs(a - d[0]) <= d[0] * rtol:
                        repl = d
                        break
                if repl is None:
                    drawn.append([a, i])
                elif repl[0] != a:
                    repl[0] = a
                    repl[1] = i
        # retirement sweep (cheap, every 96 bars ~ daily)
        if i % 96 == 0 and levels:
            alive = []
            for lv in levels:
                stale = (i - lv.last_int > P["stale_bars"]
                         and lv.touch_events < P["keep_events"])
                dead = lv.breaks >= P["max_breaks"]
                if not stale and not dead:
                    alive.append(lv)
                else:
                    lv.retired = True
            if len(alive) != len(levels):
                levels = alive
                dirty = True
        # new confirmed pivots
        while ei < len(ev) and ev[ei][0] == i:
            if dirty:
                rebuild()
            _, price, kind, w = ev[ei]
            ei += 1
            merged = False
            if len(arr):
                d = np.abs(arr - price)
                j = int(np.argmin(d))
                if d[j] <= arr[j] * P["merge_tol"]:
                    lv = levels[j]
                    lv.n_pivots += 1
                    lv.pivot_w += w
                    lv.add_member(price, i, P["max_members"], P["member_eps"])
                    lv.last_int = i
                    merged = True
                    dirty = True
            if not merged:
                levels.append(Level(price, i, w))
                dirty = True
        if dirty and (ei >= len(ev) or (ei < len(ev) and ev[ei][0] != i)):
            pass  # rebuilt lazily next bar

    # ---- emission ----
    if P.get("draw_th"):
        return [d[0] for d in drawn], [(0, d[0], None) for d in drawn]
    cand = []
    for lv in levels:
        age = n - 1 - lv.created
        recent = age < P["young_bars"]
        need = P["min_events_young"] if recent else P["min_events"]
        if lv.touch_events >= need and lv.n_pivots >= P["min_pivots"]:
            sc = lv.touch_events + P["pivot_bonus"] * lv.pivot_w
            if n - 1 - lv.last_int <= P.get("recency_bars", 10**9):
                sc *= P.get("recency_mult", 1.0)
            cand.append((sc, lv.anchor, lv))
    cand.sort(key=lambda x: (x[0], x[1]), reverse=True)
    # non-max suppression: strongest level wins its neighborhood
    out = []
    for sc, a, lv in cand:
        if all(abs(a - b) > b * P["suppress_tol"] for _, b, _ in out):
            out.append((sc, a, lv))
    if P.get("max_out"):
        out = out[:P["max_out"]]
    return [a for _, a, _ in out], out


DEFAULT = dict(
    strengths=[(20, 20, 2), (5, 5, 1)],
    merge_tol=0.0012,
    touch_tol=0.0005,
    member_eps=TICK * 1.01,
    max_members=8,
    gap_bars=16,          # 4h between distinct touch events
    hl_bars=2898,         # 45-day half-life for decayed activity scores
    stale_bars=64 * 120,  # weak levels retire after ~120 untouched days
    keep_events=8,        # levels with >= this many events never go stale
    max_breaks=999,
    break_tol=0.003,
    min_events=6,
    min_events_young=3,
    young_bars=96 * 30,
    min_pivots=1,
    pivot_bonus=2,
    suppress_tol=0.0022,
    recency_bars=96 * 30,
    recency_mult=1.0,
    max_out=None,
    # final emission rule (rej_decay threshold + NMS + latest2 vote)
    emit_th=3.2,
    emit_sup=0.0021,
    vote_tol=0.0012,
    vote_minc=2,
)


def final_emit(detail, n, P):
    """Final rule: keep clusters whose recency-decayed rejection score is
    >= emit_th; strongest-first NMS at emit_sup; line price = the cluster
    member (exact candle extreme) most recently re-validated with >=
    vote_minc hits within vote_tol of the anchor."""
    CL = []
    for sc, a, l in detail:
        f = 0.5 ** ((n - 1 - l.dec_upd) / P["hl_bars"])
        CL.append((l.rej_decay * f, a, l))
    CL.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out = []
    for rd, a, l in CL:
        if rd < P["emit_th"]:
            break
        if all(abs(a - b) > b * P["emit_sup"] for _, b, _ in out):
            out.append((rd, a, l))
    prices = []
    for rd, a, l in out:
        mem = [(p, c, b) for p, c, b in zip(l.prices, l.counts, l.lastm)
               if abs(p - a) <= a * P["vote_tol"] and c >= P["vote_minc"]]
        prices.append(max(mem, key=lambda m: m[2])[0] if mem else a)
    return prices


def main():
    df = load_m15()
    P = dict(DEFAULT)
    P.update(min_events=0, min_events_young=0, min_pivots=1,
             suppress_tol=1e-9)
    for kv in sys.argv[1:]:
        k, v = kv.split("=")
        P[k] = eval(v)
    lv, detail = run(df, P)
    if not P.get("draw_th"):
        lv = final_emit(detail, len(df), P)
    res = harness_score(lv)
    print(json.dumps({k: res[k] for k in
                      ["n_detected_in_range", "recall_strict", "recall_loose",
                       "precision", "f1_loose", "true_pos", "false_pos"]}))
    for t, nearest, d in res["per_target"]:
        flag = "STRICT" if d <= 10 else ("loose" if d <= 30 else "MISS")
        print(f"  {t:>9.2f} -> {nearest if nearest else '-':>9}  "
              f"d={d if d < 1e9 else -1:>8.2f}  {flag}")
    json.dump({"name": "H1 pivot cluster", "levels": lv},
              open(f"{BASE}/h1_pivot_cluster/detected.json", "w"))
    if P.get("dumpcsv"):
        import csv
        wtr = csv.writer(open(f"{BASE}/h1_pivot_cluster/clusters.csv", "w"))
        wtr.writerow(["anchor", "score", "events", "pivots", "pivw",
                      "created", "lastint", "breaks", "above", "below",
                      "nmember", "maxmembercount"])
        for sc, a, l in detail:
            wtr.writerow([a, sc, l.touch_events, l.n_pivots, l.pivot_w,
                          l.created, l.last_int, l.breaks, l.side_above,
                          l.side_below, len(l.prices), max(l.counts)])
        json.dump([{"anchor": a, "sc": sc, "events": l.touch_events,
                    "pivots": l.n_pivots, "pivw": l.pivot_w,
                    "created": int(l.created), "lastint": int(l.last_int),
                    "breaks": l.breaks, "above": l.side_above,
                    "below": l.side_below, "rejections": l.rejections,
                    "flips": l.flips,
                    "ev_decay": l.ev_decay * 0.5 ** ((len(df) - 1 - l.dec_upd)
                                                     / P["hl_bars"]),
                    "rej_decay": l.rej_decay * 0.5 ** ((len(df) - 1 - l.dec_upd)
                                                       / P["hl_bars"]),
                    "members": [[float(p), int(c), int(b)] for p, c, b in
                                zip(l.prices, l.counts, l.lastm)]}
                   for sc, a, l in detail],
                  open(f"{BASE}/h1_pivot_cluster/clusters.json", "w"))
    if P.get("debug"):
        targets = [30886.75, 30539.50, 30297.25, 30007.00, 29753.75, 29406.00,
                   28798.00, 27661.00, 27501.75, 26638.75, 25891.75, 25123.50,
                   25052.50, 24953.00, 24417.50, 22961.50, 22731.75]
        print("--- emitted levels near targets (within 120 pts) ---")
        for sc, a, l in sorted(detail, key=lambda x: -x[1]):
            near = min(abs(a - t) for t in targets)
            tag = f" <== d={near:.2f}" if near <= 120 else ""
            if 22500 <= a <= 31100:
                print(f"  {a:9.2f} sc={sc:6.1f} ev={l.touch_events:3d} "
                      f"piv={l.n_pivots:2d} w={l.pivot_w:3d} "
                      f"created={l.created} lastint={l.last_int} "
                      f"members={[(round(p, 2), c) for p, c in zip(l.prices, l.counts)]}{tag}")


if __name__ == "__main__":
    main()
