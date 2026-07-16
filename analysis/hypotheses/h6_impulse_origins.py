"""H6 - Impulse origins & breakout retests. Causal walk-forward level detector. v2

Single pass over 15m bars (Pine-portable):
  NOMINATION EVENTS (price always an exact candle high/low):
   1. Impulse LAUNCH: N-bar ROC crosses +/-X -> base extreme of prior N+M bars.
   2. Impulse STALL: leg momentum dies (|ROC| < X*decay) -> leg's extreme price.
   3. BREAKOUT-RETEST: confirmed pivot (left=right=L) broken by a close, then
      revisited from the other side within D bars -> pivot price.
  LEVEL BOOK (bounded array, per level: price, born, ev_score, touches below/
  above, rejections, last-contact bar):
   - nominate(p,w): merge into nearest book level within merge_pct else append.
   - every bar: levels whose band the bar enters (cooldown >= gap bars) get a
     touch; side = prior close vs level; rejection = close back on entry side.
     Optional re-snap: level price moves to the rejecting bar's wick extreme
     when that extreme lies inside the merge band (keeps exact-extreme prop).
   - retirement: book capped; drop lowest rank = (ev_score + touches).
  EMISSION at last bar:
   - candidate filter: touches >= min_t, ev_score >= min_ev_sc,
     (both_sides required only if level older than young_bars).
   - spatial NMS: sort by rank desc, keep unless within sep_pct of a kept one.
"""
import json
import sys
import numpy as np
import pandas as pd

BASE = ".."
OUT = f"{BASE}/h6_impulse_origins/detected.json"

DEF = dict(
    N=16, X=0.006, M=8, decay=0.25,          # impulse params (15m bars)
    L=12, D=480, rtol_pct=0.0008,            # breakout-retest params
    merge_pct=0.0012, band_pct=0.0005, gap=16,
    max_levels=400,
    w_launch=3.0, w_stall=3.0, w_retest=4.0,
    resnap=1,                                 # 1: move price to rejecting wick
    min_t=12, min_ev_sc=3.0,
    both_sides=0, young_bars=4000,            # role-flip filter w/ young grace
    sep_pct=0.0055,                           # NMS separation
    w_touch=1.0, w_rej=1.0,                   # rank weights
    hl=8000,                                  # half-life-ish decay (bars) for SC
    use_decay=1,                              # rank by decayed SC instead of raw
    pair_frac=0.8,                            # allow 1 close neighbor if rank
                                              # >= pair_frac * neighbor rank
    min_t_young=2,                            # touch floor for young levels
    rep_mode="rj", rep_ratio=1.0,             # window rep = max-rejection price
    keep_top=0,
)


def run(p):
    df = pd.read_csv(f"{BASE}/mnq_m15.csv")
    close = df.close.values; high = df.high.values; low = df.low.values
    n = len(df)
    N, X, M, L, D = p["N"], p["X"], p["M"], p["L"], p["D"]
    roc = np.zeros(n)
    roc[N:] = close[N:] / close[:-N] - 1

    # book columns (P0 = original nominated price; resnap bounded around it)
    # DAYS/WKS: distinct day/week buckets with any contact; LDAY/LWK last bucket
    P = []; P0 = []; BORN = []; EV = []; TB = []; TA = []; RJ = []; LAST = []
    DAYS = []; WKS = []; LDAY = []; LWK = []
    EVD = []; TD = []; RJD = []; DLAST = []   # decayed accumulators
    MAXLEG = []                               # largest impulse leg touching level
    DAYBARS = 96; WKBARS = 480
    import math
    HL = p["hl"]

    def bump_bucket(k, i):
        d = i // DAYBARS; w = i // WKBARS
        if d != LDAY[k]:
            DAYS[k] += 1; LDAY[k] = d
        if w != LWK[k]:
            WKS[k] += 1; LWK[k] = w

    def decay_to(k, i):
        f = math.exp(-(i - DLAST[k]) / HL)
        EVD[k] *= f; TD[k] *= f; RJD[k] *= f; DLAST[k] = i

    def rank_arr():
        return (np.asarray(EV) + p["w_touch"] * (np.asarray(TB) + np.asarray(TA))
                + p["w_rej"] * np.asarray(RJ))

    def nominate(i, price, w, leg=0.0):
        price = float(price)
        if P:
            arr = np.asarray(P)
            k = int(np.argmin(np.abs(arr - price)))
            if abs(arr[k] - price) <= p["merge_pct"] * price:
                EV[k] += w; LAST[k] = i; bump_bucket(k, i)
                decay_to(k, i); EVD[k] += w
                if leg > MAXLEG[k]: MAXLEG[k] = leg
                return
        P.append(price); P0.append(price); BORN.append(i); EV.append(w)
        TB.append(0); TA.append(0); RJ.append(0); LAST.append(i)
        DAYS.append(1); WKS.append(1)
        LDAY.append(i // DAYBARS); LWK.append(i // WKBARS)
        EVD.append(w); TD.append(0.0); RJD.append(0.0); DLAST.append(i)
        MAXLEG.append(leg)
        if len(P) > p["max_levels"]:
            j = int(np.argmin(rank_arr()))
            for lst in (P, P0, BORN, EV, TB, TA, RJ, LAST,
                        DAYS, WKS, LDAY, LWK, EVD, TD, RJD, DLAST, MAXLEG):
                lst.pop(j)

    state = 0; stall_ext = 0.0; leg_base = 0.0
    piv = []  # [price, kind(+1 hi/-1 lo), broken, break_i, done]
    MAXPIV = 60

    start = max(N + M, 2 * L) + 1
    for i in range(start, n):
        # impulse machine
        if state == 1:
            if high[i] > stall_ext: stall_ext = high[i]
            if roc[i] < X * p["decay"]:
                leg = abs(stall_ext - leg_base) / leg_base
                nominate(i, stall_ext, p["w_stall"], leg)
                nominate(i, leg_base, 0.0, leg)
                state = 0
        elif state == -1:
            if low[i] < stall_ext: stall_ext = low[i]
            if roc[i] > -X * p["decay"]:
                leg = abs(stall_ext - leg_base) / leg_base
                nominate(i, stall_ext, p["w_stall"], leg)
                nominate(i, leg_base, 0.0, leg)
                state = 0
        if state == 0:
            if roc[i] >= X and roc[i-1] < X:
                j0 = i - N - M
                leg_base = float(low[j0:i+1].min())
                nominate(i, leg_base, p["w_launch"])
                state = 1; stall_ext = high[i]
            elif roc[i] <= -X and roc[i-1] > -X:
                j0 = i - N - M
                leg_base = float(high[j0:i+1].max())
                nominate(i, leg_base, p["w_launch"])
                state = -1; stall_ext = low[i]

        # pivot confirmation with lag L
        c = i - L
        if high[c] == high[c-L:i+1].max() and high[c] > high[c-L:c].max(initial=-1e18):
            piv.append([float(high[c]), 1, 0, -1, 0])
        if low[c] == low[c-L:i+1].min() and low[c] < low[c-L:c].min(initial=1e18):
            piv.append([float(low[c]), -1, 0, -1, 0])
        if len(piv) > MAXPIV:
            piv = piv[-MAXPIV:]

        # breakout & retest
        for pv in piv:
            price, kind, broken, bi, done = pv
            if done: continue
            if not broken:
                if kind == 1 and close[i] > price: pv[2] = 1; pv[3] = i
                elif kind == -1 and close[i] < price: pv[2] = 1; pv[3] = i
            else:
                if i - pv[3] > D:
                    pv[4] = 1; continue
                tol = p["rtol_pct"] * price
                if kind == 1 and low[i] <= price + tol and close[i] > price - tol:
                    nominate(i, price, p["w_retest"]); pv[4] = 1
                elif kind == -1 and high[i] >= price - tol and close[i] < price + tol:
                    nominate(i, price, p["w_retest"]); pv[4] = 1

        # touches
        if P:
            arr = np.asarray(P)
            band = p["band_pct"] * arr
            hit = (low[i] <= arr + band) & (high[i] >= arr - band)
            for k in np.flatnonzero(hit):
                if i - LAST[k] < p["gap"]:
                    continue
                LAST[k] = i; bump_bucket(k, i); decay_to(k, i)
                TD[k] += 1.0
                from_below = close[i-1] < P[k]
                if from_below:
                    TB[k] += 1
                    if close[i] < P[k]:
                        RJ[k] += 1; RJD[k] += 1.0
                        if p["resnap"] and abs(high[i] - P0[k]) <= p["band_pct"] * P0[k]:
                            P[k] = float(high[i])
                else:
                    TA[k] += 1
                    if close[i] > P[k]:
                        RJ[k] += 1; RJD[k] += 1.0
                        if p["resnap"] and abs(low[i] - P0[k]) <= p["band_pct"] * P0[k]:
                            P[k] = float(low[i])

    for k in range(len(P)):
        decay_to(k, n - 1)
    book = [dict(price=P[k], born=BORN[k], ev=EV[k], tb=TB[k], ta=TA[k],
                 rj=RJ[k], last=LAST[k], days=DAYS[k], wks=WKS[k],
                 evd=EVD[k], td=TD[k], rjd=RJD[k], maxleg=MAXLEG[k])
            for k in range(len(P))]
    return book, n


def emit(book, n_bars, p):
    cands = []
    big_leg = p.get("big_leg", 0.0)
    min_ev_young = p.get("min_ev_young", p["min_ev_sc"])
    for b in book:
        t = b["tb"] + b["ta"]
        young = (n_bars - b["born"]) <= p["young_bars"]
        if young:
            ok = t >= p["min_t_young"] and b["ev"] >= min_ev_young
        else:
            ok = t >= p["min_t"] and b["ev"] >= p["min_ev_sc"]
        bigleg_pass = big_leg > 0 and b["maxleg"] >= big_leg
        if not ok and not bigleg_pass:
            continue
        if p["both_sides"] and not young and not bigleg_pass:
            if b["tb"] == 0 or b["ta"] == 0:
                continue
        mode = p.get("rank_mode", "decayed" if p["use_decay"] else "raw")
        if mode == "decayed":
            rank = b["evd"] + p["w_touch"] * b["td"] + p["w_rej"] * b["rjd"]
        elif mode == "days":
            rank = b["days"]
        elif mode == "wks":
            rank = b["wks"]
        elif mode == "dayswks":
            rank = b["days"] + 3 * b["wks"]
        else:
            rank = b["ev"] + p["w_touch"] * t + p["w_rej"] * b["rj"]
        cands.append((b["price"], rank, b))
    cands.sort(key=lambda t: -t[1])
    kept = []
    for c in cands:
        nbrs = [kc for kc in kept if abs(c[0] - kc[0]) <= p["sep_pct"] * c[0]]
        if not nbrs:
            kept.append(c)
        elif (p["pair_frac"] > 0 and len(nbrs) == 1
              and c[1] >= p["pair_frac"] * nbrs[0][1]):
            kept.append(c)
    if p["keep_top"]:
        kept = kept[:int(p["keep_top"])]
    rep = p.get("rep_mode", "")
    if rep:
        fn = {"rj": lambda b: b["rj"], "days": lambda b: b["days"],
              "t": lambda b: b["tb"] + b["ta"]}[rep]
        ratio = p.get("rep_ratio", 1.0)
        out = []
        for price, rank, b in kept:
            win = [c for c in cands
                   if abs(c[0] - price) <= p["sep_pct"] * price]
            bb = max(win, key=lambda c: fn(c[2]))
            if fn(bb[2]) < ratio * fn(b):
                bb = (price, rank, b)
            out.append((bb[0], rank, bb[2]))
        # dedupe identical reps
        seen = set(); kept = []
        for c in out:
            if c[0] not in seen:
                seen.add(c[0]); kept.append(c)
    return kept


if __name__ == "__main__":
    params = dict(DEF)
    for a in sys.argv[1:]:
        k, v = a.split("=")
        params[k] = type(DEF[k])(v)
    book, n_bars = run(params)
    res = emit(book, n_bars, params)
    levels = [round(c[0], 2) for c in res]
    json.dump({"name": "H6 impulse origins", "levels": levels}, open(OUT, "w"))
    inr = [x for x in levels if 22500 <= x <= 31100]
    print(f"emitted {len(levels)} levels, {len(inr)} in scoring range")
