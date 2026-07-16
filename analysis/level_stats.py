"""First-pass interaction stats: how MNQ 15m candles behaved around each blue level."""
import pandas as pd
import numpy as np

import os
OUT = os.path.dirname(os.path.abspath(__file__))
LEVELS = [30886.75, 30539.50, 30297.25, 30007.00, 29753.75, 29406.00, 28798.00,
          27661.00, 27501.75, 26638.75, 25891.75, 25123.50, 25052.50, 24953.00,
          24417.50, 22961.50, 22731.75]

m15 = pd.read_csv(f"{OUT}/mnq_m15.csv", parse_dates=["ts"])
h, l, o, c = (m15[k].values for k in ["high", "low", "open", "close"])
ts = m15["ts"].values

rows = []
for L in LEVELS:
    tol = L * 0.0005  # ±0.05% ≈ ±12-15 pts "spray" band
    band_lo, band_hi = L - tol, L + tol

    in_band = (h >= band_lo) & (l <= band_hi)          # bar range overlaps band
    crosses = (h >= L) & (l <= L)                       # bar straddles exact line

    # wick rejection: bar pokes into band but body closes fully on one side
    rej_from_below = (h >= band_lo) & (np.maximum(o, c) < band_lo)   # resistance touch
    rej_from_above = (l <= band_hi) & (np.minimum(o, c) > band_hi)   # support touch

    # regime: closes above vs below the line; count flips (true breaks)
    side = np.sign(c - L)
    nz = side[side != 0]
    flips = int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0

    idx = np.where(in_band)[0]
    first = pd.Timestamp(ts[idx[0]]).date() if len(idx) else None
    last = pd.Timestamp(ts[idx[-1]]).date() if len(idx) else None

    # count distinct "visits": consecutive in-band bars grouped, gap > 8 bars = new visit
    visits = 0
    if len(idx):
        visits = 1 + int(np.sum(np.diff(idx) > 8))

    rows.append({
        "level": L,
        "bars_in_band": int(in_band.sum()),
        "bars_straddle": int(crosses.sum()),
        "visits": visits,
        "rej_below(res)": int(rej_from_below.sum()),
        "rej_above(sup)": int(rej_from_above.sum()),
        "close_flips": flips,
        "first_touch": first,
        "last_touch": last,
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False))

# Round-number / tick structure check
print("\nlevel mod 100 / mod 50 / mod 25:")
for L in LEVELS:
    print(f"  {L:>9}: mod100={L % 100:>6.2f}  mod50={L % 50:>6.2f}  mod25={L % 25:>6.2f}")
