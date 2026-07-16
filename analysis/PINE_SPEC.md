# PINE SPEC — Auto S/R Level Detector (final combined algorithm)

Port target: Pine Script v6 indicator, run on the **15-minute chart** of the
instrument (built and tuned on MNQ 15m). Everything below is causal and uses
only closed bars. This spec is self-contained: implement it top-to-bottom
without reading the Python reference.

Validated performance vs a trader's 17 hand-drawn MNQ levels
(2024-09 to 2026-07): recall_loose 14/17, recall_strict 8/17,
precision 0.607, f1 0.699, 31 lines in the visible range.

---

## 1. Inputs (name = final value, suggested UI range)

| input            | value | range      | meaning |
|------------------|-------|------------|---------|
| `tfHours`        | 1     | 1–4        | HTF aggregation timeframe in hours |
| `pivStrength` s  | 8     | 5–12       | HTF pivot strength (closed HTF bars each side) |
| `freshLookback` K| 400   | 200–600    | HTF bars back for the fresh-extreme ("strong") test |
| `mergeTolPct`    | 0.20  | 0.10–0.30  | % distance for a pivot to join an existing level |
| `promoteTolPct`  | 0.08  | 0.05–0.15  | strong pivot farther than this % from every level starts its own |
| `alpha`          | 0.25  | 0.10–0.50  | center nudge toward each merged pivot |
| `bandPct`        | 0.09  | 0.06–0.12  | half-width % of the touch band around a level center |
| `snapTolPct`     | 0.30  | 0.15–0.40  | % radius for re-snapping the drawn price to the latest wick |
| `gapBars`        | 24    | 16–32      | chart (15m) bars separating distinct touch events (24 = 6h) |
| `recentDays`     | 35    | 25–45      | age (days) below which a level is "young" |
| `tYoung`         | 11    | 8–14       | young level: min touches to draw |
| `tOldStrong`     | 24    | 18–32      | old strong level: min touches (with npiv cap below) |
| `pOldStrong`     | 9     | 6–12       | old strong level: max distinct merged pivots (npiv) |
| `tOldWeak`       | 40    | 36–48      | old NON-strong level: min touches |
| `pOldWeak`       | 7     | 5–9        | old NON-strong level: max npiv |
| `tShelf`         | 14    | 12–18      | old crisp shelf: min touches |
| `pShelf`         | 2     | 1–3        | old crisp shelf: max npiv |
| `tRescue`        | 10    | 6–12       | fresh-extreme rescue: min touches |
| `rescueDays`     | 45    | 35–55      | fresh-extreme rescue: max age (days) |
| `topZonePct`     | 0.30  | 0.20–0.50  | chart-top zone half-depth % (flat plateau 0.2–0.5) |
| `maxLevels`      | 500   | fixed      | safety cap on the level book |

Constant: `BARS_PER_DAY = 96` (15m bars; adjust if chart TF differs).

All `*Pct` tolerances are used as `pct/100 * price` (relative distances).

## 2. State (parallel arrays — the "level book")

One entry per level. ~175 entries accumulate over 22 months on MNQ; cap at
`maxLevels` (see 5.4).

| array       | type  | meaning |
|-------------|-------|---------|
| `lvPx`      | float | level CENTER (drifts via alpha-nudge on merges) |
| `lvSnap`    | float | DISPLAYED price (exact candle wick, re-snapped; init = pivot price) |
| `lvTouches` | int   | distinct touch events (init 1) |
| `lvBorn`    | int   | `bar_index` at creation |
| `lvLastIn`  | int   | `bar_index` of the last bar inside the touch band (init = born) |
| `lvNpiv`    | int   | count of HTF pivots merged into this level (init 1) |
| `lvStrong`  | bool  | true once any merged pivot was a fresh K-bar extreme |

HTF state:

| var/array           | meaning |
|---------------------|---------|
| `htfH[]`, `htfL[]`  | highs/lows of COMPLETED HTF bars. Only the last `max(2s+1, K+s+1)` entries are ever read — keep as bounded rolling arrays (push to end, drop from front beyond that size). Track a running count `htfCount` of total completed HTF bars if you drop old ones (needed only for the `m >= s` guard). |
| `curKey`,`curH`,`curL` | current (incomplete) HTF bucket key and its running high/low |

## 3. Per-bar procedure (every closed 15m bar, in this order)

### 3.1 HTF aggregation
`key = floor(time / (tfHours * 3600 * 1000))`  (Pine `time` is ms).

- First bar: `curKey := key, curH := high, curL := low`.
- `key == curKey`: `curH := max(curH, high)`, `curL := min(curL, low)`.
- `key != curKey` (an HTF bar just completed):
  1. push `curH` to `htfH`, `curL` to `htfL`;
  2. `curKey := key, curH := high, curL := low`;
  3. run the pivot check of 3.2.

Do NOT use `request.security` — this bucket aggregation is deterministic,
non-repainting, and matches the validated behaviour exactly.

### 3.2 HTF pivot check (only when an HTF bar completes)
Let the completed-HTF-bar list be indexed 0..last. Candidate pivot index
`m = last - s`. Require `m >= s` (i.e. at least `2s+1` completed bars).
With window `W = htfH[m-s .. m+s]` (2s+1 values):

- **Pivot high** if `htfH[m] >= max(W)` AND `htfH[m]` occurs exactly ONCE
  in `W` (strict uniqueness — equal-high twins are rejected).
  - `strong := htfH[m] > max(htfH[m-K .. m-1])` (clamp start at 0; with a
    rolling array keep exactly the last `K+s+1` highs so this window exists).
  - call `addPivot(htfH[m], strong)`.
- **Pivot low** (mirror): `htfL[m] <= min(W)`, unique in `W`;
  `strong := htfL[m] < min(htfL[m-K .. m-1])`; `addPivot(htfL[m], strong)`.

Both a pivot high and a pivot low may fire on the same completed HTF bar.
Confirmation lag is `s` HTF bars — that is expected and allowed.

### 3.3 `addPivot(px, strong)` — create or merge
1. Find the NEAREST existing level with `abs(px - lvPx[j]) <= lvPx[j] * mergeTol`
   (scan all; keep min distance; note tolerance uses the LEVEL's price).
2. If `strong` AND (no such level OR nearest distance `> px * promoteTol`):
   append a NEW level `(px, px, 1, bar_index, bar_index, 1, true)`.
   (A fresh multi-week extreme deserves its own line even near a neighbor.)
3. Else if a nearest level `j` was found:
   `lvPx[j] += alpha * (px - lvPx[j])`; `lvNpiv[j] += 1`;
   `if strong: lvStrong[j] := true`. (`lvSnap`, `lvTouches`, `lvBorn`
   unchanged; do NOT update `lvLastIn`.)
4. Else: append a NEW level `(px, px, 1, bar_index, bar_index, 1, strong)`.

### 3.4 Touch counting + wick re-snap (every 15m bar, ALL levels)
For each level `j`, let `d = lvPx[j] * band`:
if `low - d <= lvPx[j] <= high + d` (bar range, band-extended, contains the
center):
- if `bar_index - lvLastIn[j] >= gapBars`: `lvTouches[j] += 1`
  (a distinct touch event);
- `lvLastIn[j] := bar_index` (ALWAYS, touch event or not — this is what
  makes consecutive in-zone bars one event);
- wick snap: `w = (abs(high - lvPx[j]) < abs(low - lvPx[j])) ? high : low`;
  if `abs(w - lvPx[j]) <= lvPx[j] * snapTol`: `lvSnap[j] := w`.
  (The drawn line follows the LATEST reaction wick near the center; the
  center itself does not move here.)

Note 3.3 runs before 3.4 on the same bar (a just-created level can be
touched by the current bar; with init `lvLastIn = bar_index` it will not
score a new touch until `gapBars` later — matches reference).

## 4. Emission — which levels are drawn (recompute on `barstate.islast`)

`ageDays(j) = (bar_index - lvBorn[j]) / BARS_PER_DAY`
`topPx = max(lvPx[j] over all levels with lvStrong[j])` (n/a if none).

Level `j` is drawn if ANY of:

1. **Young**: `ageDays <= recentDays` AND `lvTouches >= tYoung`.
2. **Old strong**: `ageDays > recentDays` AND `lvStrong` AND
   `lvTouches >= tOldStrong` AND `lvNpiv <= pOldStrong`.
3. **Old non-strong**: `ageDays > recentDays` AND NOT `lvStrong` AND
   `lvTouches >= tOldWeak` AND `lvNpiv <= pOldWeak`.
   (Never-fresh-extreme levels must be both busier and crisper.)
4. **Crisp shelf**: `ageDays > recentDays` AND
   `lvTouches >= tShelf` AND `lvNpiv <= pShelf`
   (one clean price respected repeatedly; chop zones have npiv >= 10).
5. **Fresh-extreme rescue**: `lvStrong` AND `ageDays <= rescueDays` AND
   `lvTouches >= tRescue`.
6. **Chart-top zone**: `lvStrong` AND `lvPx >= topPx * (1 - topZone)`.
   (The shelf just under the all-time/dataset high is always marked.)

Draw price = `lvSnap[j]` (an exact candle wick -> magnet-snappable).
De-duplicate exact-equal `lvSnap` values before drawing. Expect ~35-40
lines over a 22-month book; only ~55% lie in any one screen's range.
Levels are NEVER retired on break (targets are role-flip levels) and never
deleted by staleness — selection happens purely at draw time, so the drawn
set self-updates as touches accumulate on any bar you treat as "last".

## 5. Pine implementation notes

1. **Arrays**: use `array.new_float/int/bool` for the seven level arrays,
   push in lockstep. 3.4 is an O(levels) loop per bar — fine for ~200.
2. **HTF rolling windows**: after pushing to `htfH/htfL`, if
   `array.size > max(2s+1, K+s+1)` shift from the front. Index arithmetic
   in 3.2 then uses positions relative to the array end:
   pivot value = `array.get(htfH, size-1-s)`, window = last `2s+1` entries,
   fresh-extreme window = the `K` entries preceding those last `s+1`... —
   simplest correct mapping: keep exactly `K + s + 1` entries; pivot index
   is `size - 1 - s`; pivot window is `[size-1-2s, size-1]`; fresh window
   is `[0, size-2-s]` truncated to at most K entries. Guard: require
   `htfCount >= 2s+1` for the pivot test (fresh window may be shorter than
   K early on — use whatever exists, min 1 entry, else strong = false...
   reference: with fewer than 1 prior bar, `max()` of empty -> treat as
   strong = false).
3. **Book cap**: if level count would exceed `maxLevels`, evict the
   non-strong level with the fewest touches (oldest `lvLastIn` as
   tie-break). Never evict strong or currently-drawn levels. (The
   reference never hit any cap: 173 levels in 22 months.)
4. **Drawing**: `line.new(bar_index, snap, bar_index+1, snap, extend =
   extend.both)`; delete and redraw all lines on each `barstate.islast`
   update, or maintain a `line[]` array keyed to the book.
5. **No volume, no ATR, no `request.security`, no retirement timers** —
   all were measured and rejected (volume gating and rejection-rate gates
   actively hurt; staleness cutoffs kill year-old valid levels).
6. Ticks: prices are raw `high`/`low` values — no rounding. Works on any
   symbol; percent tolerances make it scale-free.

## 6. Parameter sensitivity (from the tuning grids)

- SHARP (re-tune per instrument with care): `pivStrength` (8 >> 7 or 9),
  `mergeTolPct` (0.20; 0.22+ merges distinct shelves, 0.16 splits),
  `gapBars` (24 >> 16 or 28), `freshLookback` (400 >> 300/500),
  `bandPct` (0.09 best; 0.07-0.09 acceptable), `snapTolPct` (0.30).
- MODERATE: `tYoung` 10-11, `tRescue` exactly 10 on MNQ (8 costs f1 ~0.02),
  `tOldWeak` plateau 39-46, `recentDays` 30-35.
- FLAT (safe defaults): `topZonePct` 0.2-0.5, `rescueDays` 45-50,
  `pShelf` 1-2, `tShelf` 14-16, `alpha`, `promoteTolPct`.

## 7. Reference results (MNQ 15m, data end 2026-07-08)

38 drawn (31 in scoring range 22500-31100). In-range list:
22741.50, 22991.25, 23819.00, 24224.00, 24423.50, 24766.25, 24924.50,
24949.25, 25010.75, 25044.00, 25073.25, 25120.50, 25147.00, 25160.00,
25245.00, 25282.25, 25710.50, 25873.25, 28810.50, 29395.00, 29457.00,
29686.25, 29744.00, 30036.25, 30226.00, 30293.00, 30357.75, 30530.25,
30594.75, 30911.75, 30967.75
Below range: 20458.25, 21177.00, 21606.50, 21613.00, 21968.00, 22157.50,
22225.50.
Scores: 14/17 loose, 8/17 strict, precision 0.607, f1 0.699. Structural
misses: 27661.00 / 27501.75 (young levels, 3-4 touches, zones misplaced
30-45 pts) and 26638.75 (no HTF pivot within 40 pts) — no generic rule
recovered them without net f1 loss in any of the six hypothesis studies.
Robustness: ending the data at 2026-05-01 / 2026-02-01 yields 26 / 19
levels, all exact prior 15m wicks, similar density across the traded range.
