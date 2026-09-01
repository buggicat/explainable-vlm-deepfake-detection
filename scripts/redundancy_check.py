#!/usr/bin/env python3
"""Cross-family redundancy spot-check for the fifteen-metric framework.

Not a full 15x15 correlation matrix (105 pairs — too much for a focused
check). Instead, computes Pearson r (with a bootstrap CI) for a handful of
cross-family pairs picked to test whether metrics that COULD plausibly
overlap actually do, pooling records across all three models unless noted.
Also explicitly checks Faithfulness against its own two sub-measures (Tool
presence, Numeric grounding) to confirm that correlation is a construction
artifact (Faithfulness = mean of the two), not a surprise empirical finding.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

RESULTS = Path(__file__).resolve().parents[1] / "results" / "full_run.scored.jsonl"
OUT = Path(__file__).resolve().parents[1] / "results" / "redundancy_ci.json"

N_BOOT = 2000
SEED = 20260819
ALPHA = 0.05


def load_records():
    recs = []
    with RESULTS.open() as f:
        for line in f:
            r = json.loads(line)
            s = r.get("scored", {})
            faith = s.get("faithfulness", {}) or {}
            expl = s.get("explanation", {}) or {}
            eva = s.get("evidence_verdict_alignment", {}) or {}
            conn = s.get("reasoning_connectivity", {}) or {}
            recs.append({
                "provider": r["provider"],
                "condition": r["condition"],
                "tool_presence": faith.get("tool_presence_score"),
                "numeric_grounding": faith.get("numeric_grounding_score"),
                "faithfulness": faith.get("score"),
                "alignment": eva.get("score"),
                "connectivity": conn.get("connectivity_per_sentence"),
                "forensic_specificity": expl.get("forensic_specificity_ratio"),
                "category_coverage": expl.get("unique_forensic_categories"),
                "evidence_completeness": expl.get("evidence_completeness_score"),
            })
    return recs


def percentile_ci(vals, alpha=ALPHA):
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def bootstrap_pearson_pair(x, y, rng, n_boot=N_BOOT):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if np.std(x) == 0 or np.std(y) == 0:
        return {"point": None, "ci_lo": None, "ci_hi": None, "n": n,
                "excludes_zero": False,
                "note": "undefined — near-constant input (little/no variance to correlate)"}
    point, _ = pearsonr(x, y)
    idx = rng.integers(0, n, size=(n_boot, n))
    rs = []
    for row in idx:
        xr, yr = x[row], y[row]
        if np.std(xr) == 0 or np.std(yr) == 0:
            continue
        r, _ = pearsonr(xr, yr)
        rs.append(r)
    if len(rs) < n_boot * 0.5:
        return {"point": float(point), "ci_lo": None, "ci_hi": None, "n": n,
                "excludes_zero": False, "note": "insufficient valid resamples"}
    lo, hi = percentile_ci(np.array(rs))
    return {"point": float(point), "ci_lo": lo, "ci_hi": hi, "n": n,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def paired_valid(records, field_x, field_y, condition=None):
    xs, ys = [], []
    for r in records:
        if condition is not None and r["condition"] != condition:
            continue
        vx, vy = r[field_x], r[field_y]
        if vx is None or vy is None:
            continue
        xs.append(vx)
        ys.append(vy)
    return xs, ys


def main():
    rng = np.random.default_rng(SEED)
    records = load_records()

    pairs = [
        # (label, field_x, field_y, condition, family_x, family_y)
        ("tool_presence__x__numeric_grounding", "tool_presence", "numeric_grounding", "B",
         "Faithfulness/Grounding (construction check)", "Faithfulness/Grounding (construction check)"),
        ("numeric_grounding__x__alignment", "numeric_grounding", "alignment", "B",
         "Faithfulness/Grounding", "Reasoning Coherence"),
        ("faithfulness__x__connectivity", "faithfulness", "connectivity", "B",
         "Faithfulness/Grounding", "Reasoning Coherence"),
        ("forensic_specificity__x__connectivity", "forensic_specificity", "connectivity", None,
         "Explanation Content", "Reasoning Coherence"),
        ("category_coverage__x__tool_presence", "category_coverage", "tool_presence", "B",
         "Explanation Content", "Faithfulness/Grounding"),
        ("evidence_completeness__x__faithfulness", "evidence_completeness", "faithfulness", "B",
         "Explanation Content (control)", "Faithfulness/Grounding"),
    ]

    out = {"n_boot": N_BOOT, "alpha": ALPHA, "seed": SEED, "pairs": {}}
    for label, fx, fy, cond, fam_x, fam_y in pairs:
        xs, ys = paired_valid(records, fx, fy, cond)
        if len(xs) < 20:
            out["pairs"][label] = {"error": "insufficient paired data", "n": len(xs)}
            continue
        res = bootstrap_pearson_pair(xs, ys, rng)
        res["family_x"] = fam_x
        res["family_y"] = fam_y
        res["condition_filter"] = cond
        out["pairs"][label] = res

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}\n")
    for label, v in out["pairs"].items():
        if "error" in v:
            print(f"  {label:45s} SKIPPED ({v['error']}, n={v['n']})")
            continue
        if v.get("point") is None or v.get("ci_lo") is None:
            print(f"  {label:45s} {v.get('note', 'undefined')} (n={v['n']})")
            print(f"      {v['family_x']}  <-->  {v['family_y']}")
            continue
        flag = "  <-- excludes zero (real correlation)" if v["excludes_zero"] else "  (CI includes zero)"
        print(f"  {label:45s} r={v['point']:+.3f}  CI=[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]  n={v['n']}{flag}")
        print(f"      {v['family_x']}  <-->  {v['family_y']}")


if __name__ == "__main__":
    main()
