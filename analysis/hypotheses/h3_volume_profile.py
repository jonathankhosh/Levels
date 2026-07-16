"""H3 VOLUME PROFILE NODES — final rule.

Walk-forward, causal, Pine-portable:

  1. Partition the 15m history into consecutive fixed 20-trading-day periods
     (1920 bars of 15m), starting at the first bar of the dataset.
  2. At each period CLOSE, build that completed period's volume-at-price
     histogram: each bar's volume is spread across its high-low range in
     5-point bins, proportional to the bin overlap.
  3. From the histogram take three prices:
        POC  = bin with maximum volume
        VAL/VAH = low/high edges of the 70% value area grown greedily
                  around the POC (classic value-area expansion)
  4. Snap each of the three prices to the nearest CONFIRMED pivot candle
     extreme (pivot high/low with 2 bars each side on 15m) within 0.15%.
     Unsnappable prices are dropped (keeps levels on exact candle H/L).
  5. Register the snapped price into a persistent level book; if it lands
     within 0.20% of an existing level the old level absorbs it (original
     price kept — trader keeps the original line), else a new level is born.
     Levels are never deleted (old nodes stay relevant for years).
  6. Output = the whole book at the end of data.

Pine port notes: histogram only needs the period's price range (~400-600
bins at 5 pts; use 10-pt bins to halve). Pivots via ta.pivothigh/low(2, 2).
Book = two arrays (price, hits) with linear merge scan.
"""
import sys, os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.dirname(HERE)

# ---- parameters (generic, no target values anywhere) ----
PERIOD_BARS   = 20 * 96   # 20 trading days of 15m bars
BIN_SIZE      = 5.0       # points
VA_FRAC       = 0.70      # value-area volume fraction
PIVOT_LR      = 2         # pivot strength (bars left/right)
SNAP_TOL_PCT  = 0.15      # max distance node -> pivot extreme, %
MERGE_TOL_PCT = 0.20      # book merge tolerance, %

def main():
    df = pd.read_csv(os.path.join(SCRATCH, 'mnq_m15.csv'))
    h = df['high'].values; l = df['low'].values
    v = df['volume'].values.astype(float)
    n = len(df)

    # confirmed pivot extremes (price-sorted for fast nearest lookup)
    piv = []
    L = PIVOT_LR
    for i in range(L, n - L):
        seg = h[i-L:i+L+1]
        if h[i] == seg.max() and (seg == h[i]).sum() == 1:
            piv.append((i, h[i]))
        seg = l[i-L:i+L+1]
        if l[i] == seg.min() and (seg == l[i]).sum() == 1:
            piv.append((i, l[i]))
    piv.sort(key=lambda x: x[1])
    piv_idx = np.array([p[0] for p in piv])
    piv_px = np.array([p[1] for p in piv])

    book = []   # dicts: px, hits, first_t, last_t

    def register(px, t):
        tol = px * MERGE_TOL_PCT / 100.0
        for lev in book:
            if abs(lev['px'] - px) <= tol:
                if t > lev['last_t']:
                    lev['hits'] += 1
                    lev['last_t'] = t
                return
        book.append({'px': px, 'hits': 1, 'first_t': t, 'last_t': t})

    def snap(px, t):
        tol = px * SNAP_TOL_PCT / 100.0
        m = (piv_idx + L <= t) & (np.abs(piv_px - px) <= tol)
        if not m.any():
            return None
        c = piv_px[m]
        return float(c[np.argmin(np.abs(c - px))])

    for s in range(0, n - PERIOD_BARS + 1, PERIOD_BARS):
        e = s + PERIOD_BARS                      # period close (causal)
        lo0 = np.floor(l[s:e].min() / BIN_SIZE) * BIN_SIZE
        nb = int(np.ceil((h[s:e].max() - lo0) / BIN_SIZE)) + 1
        hist = np.zeros(nb)
        for i in range(s, e):
            lo, hi, vol = l[i], h[i], v[i]
            b0 = int((lo - lo0) // BIN_SIZE)
            if hi <= lo:
                hist[b0] += vol
                continue
            b1 = int((hi - lo0) // BIN_SIZE)
            if b1 == b0:
                hist[b0] += vol
                continue
            edges = lo0 + BIN_SIZE * np.arange(b0, b1 + 2)
            w = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None)
            hist[b0:b1+1] += w / w.sum() * vol
        poc = int(np.argmax(hist))
        total = hist.sum()
        lo_i = hi_i = poc
        acc = hist[poc]
        while acc < VA_FRAC * total:
            up = hist[hi_i+1] if hi_i + 1 < nb else -1.0
            dn = hist[lo_i-1] if lo_i - 1 >= 0 else -1.0
            if up >= dn:
                hi_i += 1; acc += max(up, 0.0)
            else:
                lo_i -= 1; acc += max(dn, 0.0)
        for px in (lo0 + (poc + .5) * BIN_SIZE,
                   lo0 + (lo_i + .5) * BIN_SIZE,
                   lo0 + (hi_i + .5) * BIN_SIZE):
            sp = snap(px, e)
            if sp is not None:
                register(sp, e)

    levels = sorted(lev['px'] for lev in book)
    out = {"name": "H3 volume profile nodes (20d period POC+VA edges, pivot-snapped)",
           "levels": levels}
    path = os.path.join(HERE, 'detected.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"wrote {len(levels)} levels -> {path}")

if __name__ == '__main__':
    main()
