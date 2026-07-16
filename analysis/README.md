# Level-detection analysis

How the AutoLevels algorithm was derived from the trader's 17 hand-drawn MNQ
levels. Everything here is reproducible from the TrendFollower repo's CSV
exports.

## The question

The trader drew 17 blue horizontal lines on an MNQ1! chart at prices where
"price had issue breaking or going past" — placed by eye at the easiest-to-see
line inside a reaction "spray" zone:

```
30886.75  30539.50  30297.25  30007.00  29753.75  29406.00  28798.00
27661.00  27501.75  26638.75  25891.75  25123.50  25052.50  24953.00
24417.50  22961.50  22731.75
```

Can these be auto-detected from raw OHLCV?

## What the levels turned out to be (characterization)

Using 43,579 fifteen-minute bars (2024-09-01 → 2026-07-08, merged from the
TrendFollower exports):

1. **Exact wick snaps.** 16 of 17 levels exactly equal a specific 15m candle
   high or low — TradingView's magnet snapped the drawing to a real extreme.
   E.g. 30886.75 is the high of 2026-06-16 08:00 UTC; 27501.75 the double-top
   high of 2026-04-27/28. They are NOT round numbers (checked mod 25/50/100).
2. **Role-flip retest lifecycles.** Each level produced 10–85 distinct
   reaction events (contacts separated by >4h) at a ±0.05% band, approached
   from BOTH sides, with 47–100% rejection rates. Support became resistance
   and back — which is why the lines stayed relevant for weeks to months.
3. **The one exception proves the "spray" model.** 27661.00 matches no candle
   exactly: it is an eyeballed center between a May-1 consolidation cap
   (highs 27640–27674) and the May-4 retest lows (27651–27652).

## Method

1. `prep_data.py` merges the TrendFollower CSV exports into `mnq_m15.csv`
   (+ `mnq_h1.csv` for 2022+ hourly context).
2. `level_stats.py`, `anchor_profile.py` produce the characterization above.
3. Six detection hypotheses were implemented and tuned independently against
   the shared scorer (`score_levels.py`, targets in `targets.json`):
   pivot clustering, wick-rejection zones, volume profile, HTF pivots,
   role-flip lifecycle, impulse origins. Scripts + their detections are in
   `hypotheses/` (archived as-run; execute each from inside `hypotheses/`).
4. The best (h4, HTF pivots, F1 0.59) was refined at the emission stage into
   `final_algorithm.py` (F1 0.70). Scores and decisions:
   `results/hypothesis_scores.md`.
5. `PINE_SPEC.md` is the self-contained port spec that `../AutoLevels.pine`
   implements.

## Reproducing

```bash
pip install pandas numpy
cd analysis
python3 prep_data.py            # needs ../../TrendFollower (or TRENDFOLLOWER_DIR env)
python3 level_stats.py
python3 anchor_profile.py
python3 final_algorithm.py      # writes detected.json next to the script, prints scores
python3 score_levels.py results/final_detected.json
python3 final_algorithm.py end_date=2026-05-01   # robustness: truncate history
```

## Honest limitations

- Tuned on ONE instrument, ONE 22-month window, against ONE trader's 17
  levels. The parameter table in `PINE_SPEC.md` marks which knobs are sharp
  (pivot strength, merge %, touch gap) vs flat. Expect to retune per symbol.
- Three targets are structurally missed (27661, 27501.75, 26638.75): young
  zones with 3–4 touches or no 1h pivot at the line. Recovering them cost
  more in false positives than it gained in every hypothesis study.
- Precision 0.61 means ~4 of every 10 drawn lines match nothing the trader
  drew — though some of those are arguably-valid levels the trader skipped.
