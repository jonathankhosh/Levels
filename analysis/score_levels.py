"""Shared scoring harness: compare detected levels vs the user's 17 blue levels.

Usage:  python3 score_levels.py detected.json
where detected.json = {"levels": [price, ...], "name": "H1 pivot cluster"}

Scoring:
  - TARGETS: the 17 blue-box levels from the user's MNQ chart.
  - NEUTRAL: other price-scale boxes on the user's chart that are NOT blue
    (orders/positions/other drawings). Detected levels near these are excluded
    from the precision denominator (neither reward nor penalty).
  - Scoring range [22500, 31100]: detected levels outside are ignored entirely
    (chart-history levels below the visible zone are not false positives).
  - strict match: |detected - target| <= 10 pts; loose match: <= 30 pts.
Prints recall/precision/F1 and per-target nearest-detection table.
"""
import json
import sys

TARGETS = [30886.75, 30539.50, 30297.25, 30007.00, 29753.75, 29406.00, 28798.00,
           27661.00, 27501.75, 26638.75, 25891.75, 25123.50, 25052.50, 24953.00,
           24417.50, 22961.50, 22731.75]
NEUTRAL = [26379.50, 25390.25, 25274.25, 25223.00, 25206.00, 25163.75,
           24035.75, 23940.50, 23886.00, 23757.00]
LO, HI = 22500.0, 31100.0
STRICT, LOOSE = 10.0, 30.0

def score(levels):
    lv = sorted(set(float(x) for x in levels if LO <= float(x) <= HI))
    per_target = []
    hits_strict = hits_loose = 0
    for t in TARGETS:
        d, nearest = min(((abs(x - t), x) for x in lv), default=(float("inf"), None))
        per_target.append((t, nearest, d))
        hits_strict += d <= STRICT
        hits_loose += d <= LOOSE
    tp = sum(1 for x in lv if any(abs(x - t) <= LOOSE for t in TARGETS))
    neutral = sum(1 for x in lv
                  if not any(abs(x - t) <= LOOSE for t in TARGETS)
                  and any(abs(x - m) <= LOOSE for m in NEUTRAL))
    fp = len(lv) - tp - neutral
    recall_s = hits_strict / len(TARGETS)
    recall_l = hits_loose / len(TARGETS)
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (2 * precision * recall_l / (precision + recall_l)
          if precision + recall_l else 0.0)
    return {"n_detected_in_range": len(lv), "recall_strict": round(recall_s, 3),
            "recall_loose": round(recall_l, 3), "precision": round(precision, 3),
            "f1_loose": round(f1, 3), "true_pos": tp, "false_pos": fp,
            "neutral": neutral, "per_target": per_target}

if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    r = score(data["levels"])
    print(f"== {data.get('name', sys.argv[1])} ==")
    for k in ["n_detected_in_range", "recall_strict", "recall_loose",
              "precision", "f1_loose", "true_pos", "false_pos", "neutral"]:
        print(f"  {k}: {r[k]}")
    print("  per-target (target -> nearest detected, distance):")
    for t, nearest, d in r["per_target"]:
        flag = "STRICT" if d <= STRICT else ("loose" if d <= LOOSE else "MISS")
        ns = f"{nearest:.2f}" if nearest is not None else "-"
        ds = f"{d:.2f}" if d != float("inf") else "-"
        print(f"    {t:>9.2f} -> {ns:>9}  d={ds:>8}  {flag}")
