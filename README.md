# Levels — auto-detected support/resistance for MNQ

A Pine Script v6 strategy that auto-detects the horizontal support/resistance
levels a discretionary trader would hand-draw, derived by reverse-engineering
17 hand-drawn levels on an MNQ1! chart against 22 months of 15-minute data.

## The idea

Hand-drawn S/R lines turn out to have a precise signature (see
`analysis/README.md`):

- they sit on **exact candle wicks** of significant higher-timeframe swing
  extremes (TradingView magnet snap — 16 of 17 matched to the tick);
- what makes an extreme worth drawing is its **retest lifecycle**: 10–85
  distinct reaction events, approached from both sides (support ↔ resistance
  role flips), persisting for weeks to months;
- round numbers, volume nodes and rejection-quality stats turned out to be
  irrelevant or actively misleading (measured, not assumed).

`AutoLevels.pine` implements the detector that best reproduced the hand-drawn
set: **1h swing pivots (strength 8) feed a persistent level book; levels earn
their line by accumulating distinct touches; drawn prices re-snap to the
latest reaction wick.** Fully causal — closed bars only, no `request.security`,
no repainting of confirmed state.

Validated vs the trader's 17 levels (MNQ 15m, 2024-09 → 2026-07):
**14/17 found within 30 pts, 8/17 within 10 pts, precision 0.61,
~31 lines drawn in the visible range.**

## Usage

1. Open a **15-minute MNQ1!** chart on TradingView (the tuning timeframe).
2. Pine Editor → paste `AutoLevels.pine` → Add to chart.
3. Teal lines = below price (support), red = above (resistance);
   `★` labels mark fresh-extreme ("strong") levels; `Nt` = touch count.
4. The strategy layer (toggleable) demonstrates the levels by trading band
   bounces: dip into a level's band and close back across it → entry at the
   signal bar's close, fixed %-stop beyond the level, R-multiple target. It
   is a level-quality harness, not a finished system — size, sessions and
   news filters are deliberately out of scope.
5. For alerts, create ONE alert on the script with condition
   "Any alert() function call" — bounce/rejection messages include the level
   price.

Load more history (TradingView "deep backtesting" or a paid data window) for
a fuller level book: the detector needs ~2 weeks of bars before its first
pivot confirms, and old levels only get better with more history.

## Repo layout

```
AutoLevels.pine              the Pine v6 strategy (detector + demo trading layer)
analysis/
  README.md                  methodology + findings (start here)
  PINE_SPEC.md               self-contained algorithm spec the Pine port implements
  final_algorithm.py         reference implementation (Python)
  score_levels.py            scoring harness vs the 17 targets
  targets.json               the ground-truth levels from the chart screenshots
  prep_data.py               rebuilds the merged datasets from ../../TrendFollower
  level_stats.py             per-level interaction stats
  anchor_profile.py          anchor-candle + lifecycle profiling
  hypotheses/                the 6 competing detectors, archived as-run
  results/                   tournament scores + final detections
```

Data comes from the TrendFollower repo's TradingView CSV exports
(`CME_MINI_DL_MNQ1!` files); nothing here fetches live data.

## Tuning to another symbol/timeframe

The sharp knobs (retune these first, see `analysis/PINE_SPEC.md` §6):
`pivStrength` (8), `mergeTolPct` (0.20), `gapBars` (24 = 6h on 15m),
`freshLookback` (400 HTF bars), `bandPct` (0.09). The emission thresholds
plateau — move them only after the sharp knobs are placed.
