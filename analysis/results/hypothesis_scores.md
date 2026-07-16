# Hypothesis tournament — scores vs the 17 hand-drawn levels

Six detection hypotheses were implemented and tuned independently (each by a
separate agent against the shared scoring harness), then the best ideas were
synthesized into the final algorithm. Scoring: `strict` = detected within
10 pts of a target, `loose` = within 30 pts; precision counts detected levels
in range 22500–31100 that match no target (detections near known non-blue
chart objects are neutral). Optimization metric: F1 (loose).

| id | approach | recall strict | recall loose | precision | F1 | lines in range |
|----|----------|--------------:|-------------:|----------:|----:|---------------:|
| h1 | fractal pivot clustering (two strengths, cluster anchors) | 0.29 | 0.53 | 0.47 | 0.50 | 20 |
| h2 | wick-rejection zone scoring (ATR wicks + close-back events) | 0.35 | 0.53 | 0.33 | 0.41 | 31 |
| h3 | volume profile (period POC/VA edges, pivot-snapped) | 0.24 | 0.47 | 0.27 | 0.34 | 34 |
| **h4** | **1h pivots (strength 8) + fresh-extreme flag + touch-count emission** | **0.47** | **0.82** | 0.46 | **0.59** | 40 |
| h5 | role-flip lifecycle promotion (both-side reaction events) | 0.29 | 0.76 | 0.37 | 0.50 | 38 |
| h6 | impulse origins / breakout-retest nomination | 0.18 | 0.59 | 0.37 | 0.46 | 30 |
| — | **FINAL: h4 refined at emission** (see below) | **0.47** | **0.82** | **0.61** | **0.70** | **31** |

## What the synthesis kept, and why

- **Base = h4 unchanged**: 1h swing pivots of strength 8, merged within 0.20%,
  centers nudged 25% per merge, touch counting in a ±0.09% band with 6h event
  separation, price re-snapped to the latest 15m wick. Every core parameter
  re-verified as a sharp local optimum.
- **Added** (each verified to raise F1): split emission thresholds for strong
  (fresh 400h extreme) vs never-strong levels; fresh-extreme rescue touch floor
  raised 3 → 10; chart-top-zone exemption (strong levels within 0.3% of the
  highest strong level always draw).
- **Rejected after measurement**: rejection-fraction gates (anti-signal —
  rejection stats do NOT separate real levels from noise), volume filters
  (anti-signal, h3), recency-decayed scores, non-max suppression, staleness
  retirement (kills year-old valid levels like 22731.75).

## Final per-target result (data end 2026-07-08)

| target | nearest detection | distance | verdict |
|-------:|------------------:|---------:|---------|
| 30886.75 | 30911.75 | 25.00 | loose |
| 30539.50 | 30530.25 |  9.25 | STRICT |
| 30297.25 | 30293.00 |  4.25 | STRICT |
| 30007.00 | 30036.25 | 29.25 | loose |
| 29753.75 | 29744.00 |  9.75 | STRICT |
| 29406.00 | 29395.00 | 11.00 | loose |
| 28798.00 | 28810.50 | 12.50 | loose |
| 27661.00 | — | — | MISS (young zone, 3-4 touches, no 1h pivot at the line) |
| 27501.75 | — | — | MISS (young double-top, too few touches before breakout) |
| 26638.75 | — | — | MISS (no 1h pivot within 40 pts) |
| 25891.75 | 25873.25 | 18.50 | loose |
| 25123.50 | 25120.50 |  3.00 | STRICT |
| 25052.50 | 25044.00 |  8.50 | STRICT |
| 24953.00 | 24949.25 |  3.75 | STRICT |
| 24417.50 | 24423.50 |  6.00 | STRICT |
| 22961.50 | 22991.25 | 29.75 | loose |
| 22731.75 | 22741.50 |  9.75 | STRICT |

The three misses are structural: every attempt to recover them (looser pivots,
lower young thresholds, zone widening) cost more in false positives than it
gained, across all six hypothesis studies.

## Robustness (no-peeking check)

Re-running with data truncated at 2026-02-01 / 2026-05-01 / 2026-07-08 yields
19 / 26 / 38 levels — density grows sensibly with history, no flooding, no
empty chart — and spot-checked levels at every date are exact prior 15m candle
extremes, matching the trader's magnet-snap placement style.
