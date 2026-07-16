"""H5 ROLE-FLIP LIFECYCLE level detector (walk-forward, causal, Pine-portable).

Rule sketch:
  - Candidates seeded at confirmed swing pivots (pivot high/low, L/R bars,
    confirmed `piv` bars after the extreme). Seed price = exact pivot extreme.
  - Each candidate tracks reaction events against a band +/- tol*price:
      * touch starts when a bar's range intersects the band
      * resolves when a close appears beyond the band on either side
        (or after M bars, by side of close vs level)
      * same side as approach -> REJECTION; opposite side -> BREAK (side flips)
      * events counted only if >= gap bars (4h=16 m15 bars) since last event
  - PROMOTE candidate -> level once it has >= K counted events satisfying the
    promo rule ('both_touch': >=1 approach from each side; 'flip': >=1 rejection
    as support AND >=1 as resistance; 'any': just K events).
  - RETIRE on decay: `retire_consec` consecutive counted breaks with no
    rejection in between.
  - Bounded memory: at most `max_levels` tracked; unpromoted candidates with
    the oldest last touch are evicted first.
Output: promoted, alive levels at the final bar.
"""
import json
import sys
import time

import numpy as np
import pandas as pd

BASE = ".."


def load():
    df = pd.read_csv(f"{BASE}/mnq_m15.csv")
    return (df["high"].to_numpy(), df["low"].to_numpy(),
            df["close"].to_numpy(), df["ts"].to_numpy())


def pivots(H, L, piv):
    """pivot high/low flags at the PIVOT bar j (confirmed at j+piv)."""
    h = pd.Series(H)
    l = pd.Series(L)
    lmaxH = h.rolling(piv).max().shift(1)          # max of piv bars left
    rmaxH = h.shift(-1)[::-1].rolling(piv).max()[::-1]  # max of piv bars right
    lminL = l.rolling(piv).min().shift(1)
    rminL = l.shift(-1)[::-1].rolling(piv).min()[::-1]
    ph = (h >= lmaxH) & (h > rmaxH)
    pl = (l <= lminL) & (l < rminL)
    ph &= lmaxH.notna() & rmaxH.notna()
    pl &= lminL.notna() & rminL.notna()
    return ph.to_numpy(), pl.to_numpy()


def spans(H, L, capn=2000):
    """bars since a strictly higher high / lower low (causal, monotonic stack)."""
    n = len(H)
    sh = np.zeros(n, np.int64)
    sl = np.zeros(n, np.int64)
    stack = []  # (idx, high) decreasing highs
    for i in range(n):
        while stack and stack[-1][1] <= H[i]:
            stack.pop()
        sh[i] = min(i - stack[-1][0], capn) if stack else capn
        stack.append((i, H[i]))
    stack = []
    for i in range(n):
        while stack and stack[-1][1] >= L[i]:
            stack.pop()
        sl[i] = min(i - stack[-1][0], capn) if stack else capn
        stack.append((i, L[i]))
    return sh, sl


def detect(H, L, C, piv=8, tol=0.0008, K=3, gap=16, M=16, promo="both_touch",
           retire_consec=3, max_levels=400, merge_mult=1.0, seed_event=True,
           min_ev_emit=0, emit_merge=0.0, leave_mult=0.0, retire_rate=0.0,
           rate_min_ev=6, min_rej_emit=0, min_rate_emit=0.0, react_mult=1.0,
           fizzle_is_event=1, emit_top_n=0, dom_dist=0.0, leave_pct=0.0,
           max_loiter=0.0, edge_pct=0.0):
    n = len(H)
    ph, pl = pivots(H, L, piv)
    spanH, spanL = spans(H, L)
    runmax = np.maximum.accumulate(H)
    runmin = np.minimum.accumulate(L)
    cap = 6000
    price = np.zeros(cap)
    w = np.zeros(cap)
    armed = np.ones(cap, bool)
    side = np.zeros(cap, np.int8)
    pend = np.zeros(cap, bool)
    pside = np.zeros(cap, np.int8)
    pbar = np.zeros(cap, np.int64)
    lastev = np.full(cap, -10**9, np.int64)
    evb = np.zeros(cap, np.int32)   # approaches from below (resistance tests)
    eva = np.zeros(cap, np.int32)   # approaches from above (support tests)
    rejb = np.zeros(cap, np.int32)
    reja = np.zeros(cap, np.int32)
    brk = np.zeros(cap, np.int32)
    consec = np.zeros(cap, np.int32)
    lasttouch = np.zeros(cap, np.int64)
    promoted = np.zeros(cap, bool)
    alive = np.zeros(cap, bool)
    born = np.zeros(cap, np.int64)
    inband = np.zeros(cap, np.int64)
    span = np.zeros(cap, np.int64)
    cnt = 0
    use_arm = leave_mult > 0.0 or leave_pct > 0.0

    def leave_dist(k):
        return max(leave_mult * w[k], leave_pct * price[k])

    def check_promo(k, i=0):
        nonlocal promoted
        ev = evb[k] + eva[k]
        if ev < K:
            return
        # edge exception: near running ATH/ATL a one-sided rejection
        # record suffices (both-side approaches are impossible there)
        if edge_pct > 0.0:
            if (rejb[k] >= K and price[k] >= runmax[i] * (1 - edge_pct)) or \
               (reja[k] >= K and price[k] <= runmin[i] * (1 + edge_pct)):
                promoted[k] = True
                return
        if promo == "both_touch":
            ok = evb[k] >= 1 and eva[k] >= 1
        elif promo == "flip":
            ok = rejb[k] >= 1 and reja[k] >= 1
        elif promo.startswith("hybrid"):
            # both sides touched, OR a strong one-sided rejection record
            strong = int(promo[6:] or 4)
            ok = ((evb[k] >= 1 and eva[k] >= 1)
                  or rejb[k] >= strong or reja[k] >= strong)
        else:
            ok = True
        if ok:
            promoted[k] = True

    for i in range(n):
        hi, lo, cl = H[i], L[i], C[i]
        # --- seed candidates from pivots confirmed this bar ---
        j = i - piv
        if j >= 0:
            for isph in (True, False):
                if isph and not ph[j]:
                    continue
                if not isph and not pl[j]:
                    continue
                P = H[j] if isph else L[j]
                if cnt:
                    a = np.where(alive[:cnt])[0]
                    if a.size and np.any(np.abs(price[a] - P)
                                         <= merge_mult * tol * P):
                        continue  # near-duplicate: existing level absorbs it
                k = cnt
                cnt += 1
                price[k] = P
                w[k] = tol * P
                side[k] = -1 if isph else 1
                if seed_event:
                    if isph:
                        evb[k] = 1
                        rejb[k] = 1
                    else:
                        eva[k] = 1
                        reja[k] = 1
                    lastev[k] = j
                lasttouch[k] = j
                born[k] = j
                span[k] = spanH[j] if isph else spanL[j]
                alive[k] = True
                # start disarmed if price still hugging the zone
                armed[k] = (not use_arm) or abs(cl - P) >= leave_dist(k)
                check_promo(k, i)

        if cnt == 0:
            continue
        a = np.where(alive[:cnt])[0]
        # --- loiter accounting: bars whose range intersects the band ---
        ib = (lo <= price[a] + w[a]) & (hi >= price[a] - w[a])
        inband[a[ib]] += 1
        # --- re-arm levels once price has left the zone by leave dist ---
        if use_arm:
            ld = np.maximum(leave_mult * w[a], leave_pct * price[a])
            rm = (~armed[a]) & (~pend[a]) & (np.abs(cl - price[a]) >= ld)
            armed[a[rm]] = True
        # --- start touches ---
        tmask = (armed[a] & ~pend[a]
                 & (lo <= price[a] + w[a]) & (hi >= price[a] - w[a]))
        for k in a[tmask]:
            pend[k] = True
            pside[k] = side[k] if side[k] != 0 else (1 if cl > price[k] else -1)
            pbar[k] = i
        # --- resolve pending ---
        pmask = pend[a]
        for k in a[pmask]:
            P, wd = price[k], w[k]
            rd = react_mult * wd  # decisive-reaction distance
            fizzle = False
            if cl >= P + rd:
                cs = 1
            elif cl <= P - rd:
                cs = -1
            elif i - pbar[k] >= M:
                cs = 1 if cl > P else -1
                fizzle = True  # never moved decisively away: indecision
            else:
                continue
            rejected = (cs == pside[k]) and not fizzle
            broke = (cs != pside[k]) and not fizzle
            if i - lastev[k] >= gap and (not fizzle or fizzle_is_event):
                if pside[k] == -1:
                    evb[k] += 1
                    if rejected:
                        rejb[k] += 1
                else:
                    eva[k] += 1
                    if rejected:
                        reja[k] += 1
                if rejected:
                    consec[k] = 0
                elif broke:
                    brk[k] += 1
                    consec[k] += 1
                    if consec[k] >= retire_consec:
                        alive[k] = False
                lastev[k] = i
                # rejection-rate decay retirement
                if retire_rate > 0.0:
                    ev = evb[k] + eva[k]
                    if ev >= rate_min_ev and (rejb[k] + reja[k]) < retire_rate * ev:
                        alive[k] = False
                check_promo(k, i)
            side[k] = cs
            pend[k] = False
            armed[k] = not use_arm
            lasttouch[k] = i
        # --- memory cap: evict stale unpromoted candidates ---
        if alive[:cnt].sum() > max_levels:
            u = np.where(alive[:cnt] & ~promoted[:cnt])[0]
            if u.size:
                drop = u[np.argsort(lasttouch[u])][:u.size // 4 + 1]
                alive[drop] = False

    out = np.where(alive[:cnt] & promoted[:cnt])[0]
    if min_ev_emit:
        out = out[(evb[out] + eva[out]) >= min_ev_emit]
    if min_rej_emit:
        out = out[(rejb[out] + reja[out]) >= min_rej_emit]
    if min_rate_emit > 0.0:
        ev = evb[out] + eva[out]
        out = out[(rejb[out] + reja[out]) >= min_rate_emit * np.maximum(ev, 1)]
    if max_loiter > 0.0:
        ev = np.maximum(evb[out] + eva[out], 1)
        out = out[inband[out] / ev <= max_loiter]
    # dominance selection: greedily keep strongest levels, suppress weaker
    # neighbors within dom_dist (fraction of price); cap at emit_top_n
    if dom_dist > 0.0 or emit_top_n:
        sc = (rejb[out] + reja[out]).astype(float)
        order = out[np.argsort(-sc)]
        kept = []
        for k in order:
            if dom_dist > 0.0 and any(
                    abs(price[k] - price[k2]) <= dom_dist * price[k]
                    for k2 in kept):
                continue
            kept.append(k)
            if emit_top_n and len(kept) >= emit_top_n:
                break
        out = np.array(kept, dtype=int)
    levels = sorted(zip(price[out], (evb[out] + eva[out]),
                        rejb[out] + reja[out], lasttouch[out], inband[out],
                        span[out], born[out], evb[out], eva[out],
                        rejb[out], reja[out], brk[out]))
    # optional final merge of near-duplicate actives (keep most events)
    if emit_merge > 0 and levels:
        merged = []
        cur = [levels[0]]
        for lv in levels[1:]:
            if lv[0] - cur[-1][0] <= emit_merge * lv[0]:
                cur.append(lv)
            else:
                merged.append(max(cur, key=lambda x: x[1]))
                cur = [lv]
        merged.append(max(cur, key=lambda x: x[1]))
        levels = merged
    return [float(x[0]) for x in levels], [
        {"price": float(p), "events": int(e), "rej": int(r), "last": int(lt),
         "inband": int(ib), "span": int(sp), "born": int(bn),
         "evb": int(eb), "eva": int(ea), "rejb": int(rb), "reja": int(ra),
         "brk": int(bk)}
        for p, e, r, lt, ib, sp, bn, eb, ea, rb, ra, bk in levels]


def emit_final(detail, minev=3, dom=0.005):
    """Final chart selection (Pine-portable: bounded arrays + sorting).

    1. Keep levels with >= minev counted events.
    2. Greedy dominance: walk levels by descending rejection count; a level
       is kept only if no already-kept level lies within dom (0.5%) of it.
    3. Placement: the drawn line for each kept zone = event-weighted median
       of member level prices within dom of the winner (all members are
       exact pivot candle extremes, so the line stays magnet-snappable).
    """
    P = np.array([d["price"] for d in detail])
    EV = np.array([d["events"] for d in detail], float)
    RJ = np.array([d["rej"] for d in detail], float)
    idx = np.where(EV >= minev)[0]
    order = idx[np.argsort(-RJ[idx])]
    kept = []
    for k in order:
        if any(abs(P[k] - P[k2]) <= dom * P[k] for k2 in kept):
            continue
        kept.append(k)
    out = []
    for k in kept:
        zone = idx[np.abs(P[idx] - P[k]) <= dom * P[k]]
        o = zone[np.argsort(P[zone])]
        cum = np.cumsum(EV[o])
        out.append(float(P[o][np.searchsorted(cum, cum[-1] / 2.0)]))
    return sorted(out)


BEST = dict(piv=8, tol=0.0008, K=3, gap=16, M=16, promo="both_touch",
            retire_consec=2)
BEST_EMIT = dict(minev=3, dom=0.005)

if __name__ == "__main__":
    H, L, C, TS = load()
    t0 = time.time()
    kw = dict(BEST)
    for arg in sys.argv[1:]:
        k, v = arg.split("=")
        kw[k] = v if k == "promo" else (float(v) if "." in v else int(v))
    _, detail = detect(H, L, C, **kw)
    levels = emit_final(detail, **BEST_EMIT)
    print(f"took {time.time()-t0:.1f}s  n_levels={len(levels)}")
    json.dump({"name": "H5 role-flip lifecycle", "levels": levels},
              open(f"{BASE}/h5_role_flip/detected.json", "w"))
    sys.path.insert(0, BASE)
    from score_levels import score
    r = score(levels)
    for kk in ["n_detected_in_range", "recall_strict", "recall_loose",
               "precision", "f1_loose", "true_pos", "false_pos"]:
        print(f"  {kk}: {r[kk]}")
