"""Merge TrendFollower MNQ CSVs into clean OHLCV datasets (UTC epoch)."""
import pandas as pd
import numpy as np

import os
SRC = os.environ.get("TRENDFOLLOWER_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "TrendFollower"))
OUT = os.path.dirname(os.path.abspath(__file__))

def load(path, tcol="time"):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    t = df[cols.get("time", cols.get("datetime"))]
    if pd.api.types.is_numeric_dtype(t):
        ts = pd.to_datetime(t, unit="s", utc=True)
    else:
        ts = pd.to_datetime(t, utc=True, format="ISO8601")
    out = pd.DataFrame({
        "ts": ts,
        "open": df[cols["open"]],
        "high": df[cols["high"]],
        "low": df[cols["low"]],
        "close": df[cols["close"]],
        "volume": df[cols["volume"]] if "volume" in cols else np.nan,
    })
    return out.dropna(subset=["open", "high", "low", "close"])

# 15m: older export (Sept 2024 →) + freshest export (→ July 2026)
m15_a = load(f"{SRC}/CME_MINI_DL_MNQ1!, 15 (4).csv")   # 2024-09 → 2026-06-05, has Volume
m15_b = load(f"{SRC}/CME_MINI_DL_MNQ1!, 15 (1).csv")   # 2025-09 → 2026-07-09
m15 = pd.concat([m15_a, m15_b]).sort_values("ts")
m15 = m15.drop_duplicates(subset="ts", keep="last").reset_index(drop=True)

# 1h: 2022-06 → 2026-06-05
h1 = load(f"{SRC}/CME_MINI_DL_MNQ1!, 60 (1).csv").sort_values("ts")
h1 = h1.drop_duplicates(subset="ts", keep="last").reset_index(drop=True)

for name, df in [("m15", m15), ("h1", h1)]:
    df["epoch"] = (df["ts"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
    df.to_csv(f"{OUT}/mnq_{name}.csv", index=False)
    print(name, len(df), df["ts"].min(), "->", df["ts"].max(),
          "| price range:", df["low"].min(), "-", df["high"].max())

# sanity: gap check on 15m
d = m15["epoch"].diff().dropna()
print("15m gap histogram (top):")
print(d.value_counts().head(5))
