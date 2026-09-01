#!/usr/bin/env python3
"""Bootstrap confidence intervals for detection accuracy, bias gap, agentic
uplift, and explanation-quality metrics.

Computes 95% percentile bootstrap CIs from the already-scored per-image logs
(full_run.scored.jsonl) — no new API calls, reuses the label/prediction/self-
calibration fields score.py already computed per record.

Reports, per (provider, scenario, condition) cell unless noted:
  - accuracy CI (2000 resamples, resample images with replacement)
  - bias-gap CI per model per condition (S1 acc - mean(S2_self, S2_local) acc);
    S1 and S2 are different image sets, so this is an UNPAIRED bootstrap
    (resample each cell independently, recombine per iteration)
  - uplift CI per model per scenario (Acc(B) - Acc(A)); the same 1200/200
    image_ids are used in both conditions (confirmed paired design), so this
    is a PAIRED bootstrap (resample image indices once, apply to both A and B)
  - self-calibration Pearson r CI per model per condition, pooled across the
    three scenarios (one r per model/condition rather than per cell)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import expected_calibration_error, semantic_cosine, bleu1, _s2_base  # reuse exact logic

RESULTS = Path(__file__).resolve().parents[1] / "results" / "full_run.scored.jsonl"
JUDGE_RESULTS = Path(__file__).resolve().parents[1] / "results" / "llm_judge_final.jsonl"
OUT = Path(__file__).resolve().parents[1] / "results" / "bootstrap_ci.json"

N_BOOT = 2000
SEED = 20260819  # fixed seed for reproducibility of the CI computation itself
ALPHA = 0.05  # 95% CI

MODELS = ["openai", "gemini", "anthropic"]
S2_VARIANTS = ["scenario2_self_cond", "scenario2_local_inpaint"]


def load_records():
    recs = []
    with RESULTS.open() as f:
        for line in f:
            r = json.loads(line)
            s = r.get("scored", {})
            if s.get("correct") is None:
                continue
            recs.append({
                "provider": r["provider"],
                "scenario": r["scenario"],
                "condition": r["condition"],
                "image_id": r["image_id"],
                "correct": int(s["correct"]),
                "self_conf_mean": s.get("self_conf_mean"),
                "label": 1 if r["label"] == "fake" else 0,
                "pred": 1 if r["classification"] == "fake" else 0,
                "p_fake": float(s["p_fake"]) if s.get("p_fake") is not None else None,
                "compactness": (s.get("compactness") or {}).get("score"),
                "forensic_specificity_ratio": (s.get("explanation") or {}).get(
                    "forensic_specificity_ratio"),
                "manipulation_term_count": (s.get("explanation") or {}).get(
                    "manipulation_term_count"),
                "content_term_count": (s.get("explanation") or {}).get(
                    "content_term_count"),
                "faithfulness_score": (s.get("faithfulness") or {}).get("score"),
                "tool_presence_score": (s.get("faithfulness") or {}).get("tool_presence_score"),
                "numeric_grounding_score": (s.get("faithfulness") or {}).get(
                    "numeric_grounding_score"),
                "alignment_score": (s.get("evidence_verdict_alignment") or {}).get("score"),
                "connectivity": (s.get("reasoning_connectivity") or {}).get(
                    "connectivity_per_sentence"),
                "raw_text": r.get("raw_text") or "",
                "raw_label": r["label"],
            })
    return recs


def load_judge_records():
    recs = []
    with JUDGE_RESULTS.open() as f:
        for line in f:
            r = json.loads(line)
            jp = r.get("judge_parsed") or {}
            g = jp.get("overall_groundedness")
            h = jp.get("hallucination_detected")
            recs.append({
                "provider": r["provider"],
                "groundedness": float(g) if isinstance(g, (int, float)) else None,
                "hallucinated": 1 if h is True else (0 if h is False else None),
            })
    return recs


def bootstrap_mean(vals: list[float], rng: np.random.Generator, n_boot=N_BOOT):
    """Generic percentile-bootstrap CI for the mean of any 1-D list of floats
    (works for 0/1 rate metrics like J-Hal too)."""
    arr = np.array(vals, dtype=float)
    n = len(arr)
    point = float(arr.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = percentile_ci(means)
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n": n}


def percentile_ci(vals: np.ndarray, alpha: float = ALPHA):
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_accuracy(correct_arr: np.ndarray, rng: np.random.Generator, n_boot=N_BOOT):
    n = len(correct_arr)
    point = float(correct_arr.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    resampled_means = correct_arr[idx].mean(axis=1)
    lo, hi = percentile_ci(resampled_means)
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n": n}


def bootstrap_paired_diff(a_arr: np.ndarray, b_arr: np.ndarray, rng: np.random.Generator, n_boot=N_BOOT):
    """a_arr, b_arr are the SAME length, index-aligned (same image order)."""
    n = len(a_arr)
    point = float(b_arr.mean() - a_arr.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = b_arr[idx].mean(axis=1) - a_arr[idx].mean(axis=1)
    lo, hi = percentile_ci(diffs)
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n": n,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def bootstrap_s2_pooled_uplift(variant_pairs: list[tuple[np.ndarray, np.ndarray]],
                                rng: np.random.Generator, n_boot=N_BOOT):
    """S2 uplift CI genuinely pooled across both S2 variants (self-cond,
    local-inpaint) — resamples each variant's paired A/B indices
    independently per iteration, averages the two variants' uplift for that
    iteration, and repeats. This is the same "average across S2 variants
    per resample" logic bias_gap_ci already uses, applied to uplift instead
    of accuracy. NOT the same as taking the envelope (min/max) of the two
    variants' separately-computed CIs, which is a looser approximation."""
    point = float(np.mean([float(b.mean() - a.mean()) for a, b in variant_pairs]))
    per_iter_means = []
    for _ in range(n_boot):
        variant_uplifts = []
        for a_arr, b_arr in variant_pairs:
            n = len(a_arr)
            idx = rng.integers(0, n, size=n)
            variant_uplifts.append(float(b_arr[idx].mean() - a_arr[idx].mean()))
        per_iter_means.append(float(np.mean(variant_uplifts)))
    per_iter_means = np.array(per_iter_means)
    lo, hi = percentile_ci(per_iter_means)
    return {"point": point, "ci_lo": lo, "ci_hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def bootstrap_s2_pooled_accuracy(variant_arrs: list[np.ndarray], rng: np.random.Generator,
                                  n_boot=N_BOOT):
    """Table 1's 'S2' row is the mean of both S2 variants (self-cond,
    local-inpaint) collapsed into one row — same pooling as
    bootstrap_s2_pooled_uplift, applied to a plain accuracy array instead
    of a paired A/B difference."""
    point = float(np.mean([a.mean() for a in variant_arrs]))
    per_iter = []
    for _ in range(n_boot):
        means = []
        for arr in variant_arrs:
            n = len(arr)
            idx = rng.integers(0, n, size=n)
            means.append(float(arr[idx].mean()))
        per_iter.append(float(np.mean(means)))
    per_iter = np.array(per_iter)
    lo, hi = percentile_ci(per_iter)
    return {"point": point, "ci_lo": lo, "ci_hi": hi}


def bootstrap_s2_pooled_table1_metric(variant_data: list[tuple], metric: str,
                                       rng: np.random.Generator, n_boot=N_BOOT):
    """Same S2-row pooling as bootstrap_s2_pooled_accuracy, for F1/AUC/ECE/
    Brier instead of plain accuracy. variant_data: list of (labels, preds,
    probs) arrays, one tuple per S2 variant."""
    points = [_safe_metric(l, p, pr, metric) for l, p, pr in variant_data]
    points = [x for x in points if x is not None]
    if not points:
        return None
    point = float(np.mean(points))
    per_iter = []
    for _ in range(n_boot):
        vals = []
        for l, p, pr in variant_data:
            n = len(l)
            idx = rng.integers(0, n, size=n)
            v = _safe_metric(l[idx], p[idx], pr[idx], metric)
            if v is not None:
                vals.append(v)
        if vals:
            per_iter.append(float(np.mean(vals)))
    if len(per_iter) < n_boot * 0.5:
        return {"point": point, "ci_lo": None, "ci_hi": None}
    per_iter = np.array(per_iter)
    lo, hi = percentile_ci(per_iter)
    return {"point": point, "ci_lo": lo, "ci_hi": hi}


def bootstrap_unpaired_diff_from_group(s1_arr: np.ndarray, s2_arrs: list[np.ndarray],
                                        rng: np.random.Generator, n_boot=N_BOOT):
    """bias gap = acc(S1) - mean(acc(S2_variant) for each variant); each array
    resampled independently since the underlying image sets differ."""
    n1 = len(s1_arr)
    idx1 = rng.integers(0, n1, size=(n_boot, n1))
    s1_means = s1_arr[idx1].mean(axis=1)

    s2_means_per_variant = []
    for arr in s2_arrs:
        n2 = len(arr)
        idx2 = rng.integers(0, n2, size=(n_boot, n2))
        s2_means_per_variant.append(arr[idx2].mean(axis=1))
    s2_mean_combined = np.mean(s2_means_per_variant, axis=0)

    diffs = s1_means - s2_mean_combined
    point = float(s1_arr.mean() - np.mean([a.mean() for a in s2_arrs]))
    lo, hi = percentile_ci(diffs)
    return {"point": point, "ci_lo": lo, "ci_hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def _safe_metric(labels, preds, probs, metric: str):
    try:
        if metric == "f1":
            return float(f1_score(labels, preds, zero_division=0))
        if metric == "auc":
            if len(set(labels)) < 2:
                return None
            return float(roc_auc_score(labels, probs))
        if metric == "ece":
            return float(expected_calibration_error(labels, probs))
        if metric == "brier":
            return float(np.mean((np.array(probs) - np.array(labels, dtype=float)) ** 2))
    except (ValueError, ZeroDivisionError):
        return None
    return None


def bootstrap_table1_metric(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray,
                             metric: str, rng: np.random.Generator, n_boot=N_BOOT):
    n = len(labels)
    point = _safe_metric(labels, preds, probs, metric)
    if point is None:
        return None
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = []
    for row in idx:
        v = _safe_metric(labels[row], preds[row], probs[row], metric)
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot * 0.5:  # too many degenerate resamples (e.g. single-class draw)
        return {"point": point, "ci_lo": None, "ci_hi": None, "n": n,
                "note": "insufficient valid resamples"}
    lo, hi = percentile_ci(np.array(vals))
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n": n, "n_valid_resamples": len(vals)}


def bootstrap_pearson(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot=N_BOOT):
    n = len(x)
    point, _ = pearsonr(x, y)
    idx = rng.integers(0, n, size=(n_boot, n))
    rs = []
    for row in idx:
        xr, yr = x[row], y[row]
        if np.std(xr) == 0 or np.std(yr) == 0:
            continue
        r, _ = pearsonr(xr, yr)
        rs.append(r)
    rs = np.array(rs)
    lo, hi = percentile_ci(rs)
    return {"point": float(point), "ci_lo": lo, "ci_hi": hi, "n": n,
            "excludes_zero": bool(lo > 0 or hi < 0), "n_valid_resamples": len(rs)}


def bootstrap_pearson_mean_of_scenarios(xy_by_scenario: dict[str, tuple[np.ndarray, np.ndarray]],
                                         rng: np.random.Generator, n_boot=N_BOOT):
    """Computes one Pearson r PER SCENARIO, then averages across the three
    scenarios (matching xai_metrics.csv / score.py's convention), rather
    than pooling all records into a single r. Resamples each
    scenario's records independently (with replacement) per iteration, so
    the block structure (n per scenario) matches how the point estimate was
    built. Scenarios whose resample has zero variance in x or y are dropped
    from that iteration's average, same convention as bootstrap_pearson."""
    scen_names = list(xy_by_scenario.keys())
    point_rs = []
    for scen in scen_names:
        x, y = xy_by_scenario[scen]
        r, _ = pearsonr(x, y)
        point_rs.append(r)
    point = float(np.mean(point_rs))

    ns = {scen: len(xy_by_scenario[scen][0]) for scen in scen_names}
    means = []
    for _ in range(n_boot):
        rs = []
        for scen in scen_names:
            x, y = xy_by_scenario[scen]
            row = rng.integers(0, ns[scen], size=ns[scen])
            xr, yr = x[row], y[row]
            if np.std(xr) == 0 or np.std(yr) == 0:
                continue
            r, _ = pearsonr(xr, yr)
            rs.append(r)
        if rs:
            means.append(float(np.mean(rs)))
    means = np.array(means)
    lo, hi = percentile_ci(means)
    return {"point": point, "ci_lo": lo, "ci_hi": hi,
            "n_per_scenario": ns, "excludes_zero": bool(lo > 0 or hi < 0),
            "n_valid_resamples": len(means)}


def main():
    rng = np.random.default_rng(SEED)
    records = load_records()

    # index: (provider, scenario, condition) -> ordered list of (image_id, correct)
    cells = defaultdict(list)
    for r in records:
        cells[(r["provider"], r["scenario"], r["condition"])].append(r)

    out = {"n_boot": N_BOOT, "alpha": ALPHA, "seed": SEED,
           "accuracy_ci": {}, "uplift_ci": {}, "bias_gap_ci": {}, "self_calibration_ci": {},
           "self_calibration_meanofscenarios_ci": {}, "table1_metric_ci": {}}

    # 1. Per-cell accuracy CI
    for key, recs in cells.items():
        arr = np.array([r["correct"] for r in recs])
        out["accuracy_ci"]["|".join(key)] = bootstrap_accuracy(arr, rng)

    # 1b. Per-cell F1 / AUC / ECE / Brier CI (Table 1's other bolded/ranked columns)
    for key, recs in cells.items():
        labels = np.array([r["label"] for r in recs])
        preds = np.array([r["pred"] for r in recs])
        probs = np.array([r["p_fake"] if r["p_fake"] is not None else np.nan for r in recs])
        cell_out = {}
        for metric in ("f1", "auc", "ece", "brier"):
            if metric in ("auc", "ece", "brier") and np.isnan(probs).any():
                # drop records with missing p_fake for the probability-based metrics
                mask = ~np.isnan(probs)
                res = bootstrap_table1_metric(labels[mask], preds[mask], probs[mask], metric, rng)
            else:
                res = bootstrap_table1_metric(labels, preds, probs, metric, rng)
            if res is not None:
                cell_out[metric] = res
        out["table1_metric_ci"]["|".join(key)] = cell_out

    # 2. Uplift CI (paired, B - A) per model per scenario
    scenarios = ["scenario1"] + S2_VARIANTS
    for model in MODELS:
        for scen in scenarios:
            key_a = (model, scen, "A")
            key_b = (model, scen, "B")
            if key_a not in cells or key_b not in cells:
                continue
            recs_a = {r["image_id"]: r["correct"] for r in cells[key_a]}
            recs_b = {r["image_id"]: r["correct"] for r in cells[key_b]}
            shared_ids = sorted(set(recs_a) & set(recs_b))
            a_arr = np.array([recs_a[i] for i in shared_ids])
            b_arr = np.array([recs_b[i] for i in shared_ids])
            out["uplift_ci"][f"{model}|{scen}"] = bootstrap_paired_diff(a_arr, b_arr, rng)

    # 2b. S2 pooled uplift CI (genuinely pooled across both S2 variants, not
    #     the envelope of two separate CIs) — this is the number reported in
    #     the table's S2 Uplift cell, matching how bias_gap already treats S2.
    out["s2_pooled_uplift_ci"] = {}
    for model in MODELS:
        variant_pairs = []
        for scen in S2_VARIANTS:
            key_a, key_b = (model, scen, "A"), (model, scen, "B")
            if key_a not in cells or key_b not in cells:
                continue
            recs_a = {r["image_id"]: r["correct"] for r in cells[key_a]}
            recs_b = {r["image_id"]: r["correct"] for r in cells[key_b]}
            shared_ids = sorted(set(recs_a) & set(recs_b))
            variant_pairs.append((np.array([recs_a[i] for i in shared_ids]),
                                   np.array([recs_b[i] for i in shared_ids])))
        if len(variant_pairs) == len(S2_VARIANTS):
            out["s2_pooled_uplift_ci"][model] = bootstrap_s2_pooled_uplift(variant_pairs, rng)

    # 2c. Table 1's "S2" row CI (Acc/F1/AUC/ECE/Brier), pooled across both S2
    #     variants the same way — Table 1 shows one S2 row per model per
    #     condition, not two separate variant rows.
    out["s2_pooled_accuracy_ci"] = {}
    out["s2_pooled_table1_metric_ci"] = {}
    for model in MODELS:
        for cond in ["A", "B"]:
            variant_arrs, variant_data = [], []
            ok = True
            for scen in S2_VARIANTS:
                key = (model, scen, cond)
                if key not in cells:
                    ok = False
                    break
                recs = cells[key]
                variant_arrs.append(np.array([r["correct"] for r in recs]))
                labels = np.array([r["label"] for r in recs])
                preds = np.array([r["pred"] for r in recs])
                probs = np.array([r["p_fake"] if r["p_fake"] is not None else np.nan for r in recs])
                variant_data.append((labels, preds, probs))
            if not ok:
                continue
            out["s2_pooled_accuracy_ci"][f"{model}|{cond}"] = \
                bootstrap_s2_pooled_accuracy(variant_arrs, rng)
            cell_out = {}
            for metric in ("f1", "auc", "ece", "brier"):
                vd = variant_data
                if metric in ("auc", "ece", "brier"):
                    vd = [(l[~np.isnan(pr)], p[~np.isnan(pr)], pr[~np.isnan(pr)])
                          for l, p, pr in variant_data]
                res = bootstrap_s2_pooled_table1_metric(vd, metric, rng)
                if res is not None:
                    cell_out[metric] = res
            out["s2_pooled_table1_metric_ci"][f"{model}|{cond}"] = cell_out

    # 3. Bias-gap CI (unpaired, S1 - mean(S2 variants)) per model per condition
    for model in MODELS:
        for cond in ["A", "B"]:
            key_s1 = (model, "scenario1", cond)
            if key_s1 not in cells:
                continue
            s1_arr = np.array([r["correct"] for r in cells[key_s1]])
            s2_arrs = []
            ok = True
            for variant in S2_VARIANTS:
                key_s2 = (model, variant, cond)
                if key_s2 not in cells:
                    ok = False
                    break
                s2_arrs.append(np.array([r["correct"] for r in cells[key_s2]]))
            if not ok:
                continue
            out["bias_gap_ci"][f"{model}|{cond}"] = bootstrap_unpaired_diff_from_group(
                s1_arr, s2_arrs, rng)

    # 4. Self-calibration Pearson r CI, pooled across the three scenarios, per model/condition
    for model in MODELS:
        for cond in ["A", "B"]:
            xs, ys = [], []
            for scen in scenarios:
                key = (model, scen, cond)
                if key not in cells:
                    continue
                for r in cells[key]:
                    if r["self_conf_mean"] is not None:
                        xs.append(r["self_conf_mean"])
                        ys.append(r["correct"])
            if len(xs) < 10:
                continue
            x_arr = np.array(xs, dtype=float)
            y_arr = np.array(ys, dtype=float)
            out["self_calibration_ci"][f"{model}|{cond}"] = bootstrap_pearson(x_arr, y_arr, rng)

    # 4b. Self-calibration Pearson r CI as the mean of three per-scenario
    #     correlations (see xai_metrics.csv / score.py), rather than one
    #     correlation pooled over all records.
    out["self_calibration_meanofscenarios_ci"] = {}
    for model in MODELS:
        for cond in ["A", "B"]:
            xy_by_scenario = {}
            for scen in scenarios:
                key = (model, scen, cond)
                if key not in cells:
                    continue
                xs, ys = [], []
                for r in cells[key]:
                    if r["self_conf_mean"] is not None:
                        xs.append(r["self_conf_mean"])
                        ys.append(r["correct"])
                if len(xs) >= 10:
                    xy_by_scenario[scen] = (np.array(xs, dtype=float), np.array(ys, dtype=float))
            if len(xy_by_scenario) < 2:
                continue
            out["self_calibration_meanofscenarios_ci"][f"{model}|{cond}"] = \
                bootstrap_pearson_mean_of_scenarios(xy_by_scenario, rng)

    # 5. Compactness CI per model per condition, pooled over scenarios.
    out["compactness_ci"] = {}
    for model in MODELS:
        for cond in ["A", "B"]:
            vals = [r["compactness"] for r in records
                    if r["provider"] == model and r["condition"] == cond
                    and r["compactness"] is not None]
            if len(vals) < 10:
                continue
            arr = np.array(vals)
            n = len(arr)
            idx = rng.integers(0, n, size=(N_BOOT, n))
            means = arr[idx].mean(axis=1)
            lo, hi = percentile_ci(means)
            out["compactness_ci"][f"{model}|{cond}"] = {
                "point": float(arr.mean()), "ci_lo": lo, "ci_hi": hi, "n": n}

    # 6. Forensic-specificity-ratio "shift" CI (B - A, paired), pooled over
    #    scenarios.
    out["spec_shift_ci"] = {}
    for model in MODELS:
        a_map, b_map = {}, {}
        for r in records:
            if r["provider"] != model or r["forensic_specificity_ratio"] is None:
                continue
            key = (r["scenario"], r["image_id"])
            if r["condition"] == "A":
                a_map[key] = r["forensic_specificity_ratio"]
            elif r["condition"] == "B":
                b_map[key] = r["forensic_specificity_ratio"]
        shared = sorted(set(a_map) & set(b_map))
        if len(shared) < 10:
            continue
        a_arr = np.array([a_map[k] for k in shared])
        b_arr = np.array([b_map[k] for k in shared])
        out["spec_shift_ci"][model] = bootstrap_paired_diff(a_arr, b_arr, rng)

    # 7. Contrastive sensitivity (Sens-SC / Sens-LI) CI per model per S2
    #    variant, pooled over conditions A+B. Compute each pair's divergence
    #    ONCE (embedding call is expensive), then bootstrap resample over the
    #    resulting list of pair-level divergence values.
    out["sensitivity_ci"] = {}
    for model in MODELS:
        for variant, label_short in [("scenario2_self_cond", "SC"),
                                      ("scenario2_local_inpaint", "LI")]:
            pairs: dict[tuple, dict] = defaultdict(dict)
            for r in records:
                if r["provider"] != model or r["scenario"] != variant:
                    continue
                base = _s2_base(r["image_id"])
                if base is None:
                    continue
                key = (base, r["condition"])
                pairs[key][r["raw_label"]] = r["raw_text"]
            divergences = []
            for cond_map in pairs.values():
                if "real" not in cond_map or "fake" not in cond_map:
                    continue
                sim = semantic_cosine(cond_map["real"], cond_map["fake"])
                if sim is not None:
                    divergences.append(1.0 - sim)
            if len(divergences) < 10:
                continue
            arr = np.array(divergences)
            n = len(arr)
            idx = rng.integers(0, n, size=(N_BOOT, n))
            means = arr[idx].mean(axis=1)
            lo, hi = percentile_ci(means)
            out["sensitivity_ci"][f"{model}|{label_short}"] = {
                "point": float(arr.mean()), "ci_lo": lo, "ci_hi": hi, "n_pairs": n}

    # 8. Tool, Num, Faith — condition B only, pooled over scenarios per model.
    #    Simple per-record means, so pooling and mean-of-scenarios coincide
    #    here (unlike self-calibration's Pearson r) since group sizes per
    #    scenario are roughly equal.
    for out_key, field in [("tool_presence_ci", "tool_presence_score"),
                            ("numeric_grounding_ci", "numeric_grounding_score"),
                            ("faithfulness_ci", "faithfulness_score")]:
        out[out_key] = {}
        for model in MODELS:
            vals = [r[field] for r in records
                    if r["provider"] == model and r["condition"] == "B"
                    and r[field] is not None]
            if len(vals) >= 10:
                out[out_key][model] = bootstrap_mean(vals, rng)

    # 8b. Cross-condition BLEU-1 (A vs B, same image), pooled over scenarios
    #     per model — pure Python, no embedding model needed (unlike Cos,
    #     which requires sentence-transformers and is NOT computed here —
    #     this sandbox has no cached embedding model; run on the machine
    #     that has sentence-transformers installed, same as Sens-SC/Sens-LI
    #     and the Scenario-1 sensitivity baseline were).
    out["bleu_ci"] = {}
    for model in MODELS:
        by_key = {}
        for r in records:
            if r["provider"] != model:
                continue
            key = (r["image_id"], r["scenario"])
            by_key.setdefault(key, {})[r["condition"]] = r["raw_text"]
        bleu_vals = [bleu1(cm["B"], cm["A"]) for cm in by_key.values()
                     if "A" in cm and "B" in cm]
        if len(bleu_vals) >= 10:
            out["bleu_ci"][model] = bootstrap_mean(bleu_vals, rng)

    # 9. LLM-judge groundedness (J-Grnd) and hallucination rate (J-Hal)
    #    (n=198: 99 GPT + 99 Gemini, condition B only — Claude is the judge
    #    and is not itself judged).
    judge_records = load_judge_records()
    out["judge_groundedness_ci"] = {}
    out["judge_hallucination_ci"] = {}
    for model in ("openai", "gemini"):
        g_vals = [r["groundedness"] for r in judge_records
                  if r["provider"] == model and r["groundedness"] is not None]
        h_vals = [r["hallucinated"] for r in judge_records
                  if r["provider"] == model and r["hallucinated"] is not None]
        if len(g_vals) >= 10:
            out["judge_groundedness_ci"][model] = bootstrap_mean(g_vals, rng)
        if len(h_vals) >= 10:
            out["judge_hallucination_ci"][model] = bootstrap_mean(h_vals, rng)

    # 10. Align, Conn, Spec%, Manip., Content — per model per condition,
    #     pooled over scenarios (same equal-group-size reasoning as block 8
    #     above; only Pearson r needed the special mean-of-scenarios
    #     treatment).
    for out_key, field, scale in [("alignment_ci", "alignment_score", 1.0),
                                   ("connectivity_ci", "connectivity", 1.0),
                                   ("spec_pct_ci", "forensic_specificity_ratio", 100.0),
                                   ("manip_term_ci", "manipulation_term_count", 1.0),
                                   ("content_term_ci", "content_term_count", 1.0)]:
        out[out_key] = {}
        for model in MODELS:
            for cond in ["A", "B"]:
                vals = [r[field] * scale for r in records
                        if r["provider"] == model and r["condition"] == cond
                        and r[field] is not None]
                if len(vals) >= 10:
                    out[out_key][f"{model}|{cond}"] = bootstrap_mean(vals, rng)

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")

    # Readable summary
    print("\n--- Accuracy CIs (95%) ---")
    for k, v in sorted(out["accuracy_ci"].items()):
        print(f"  {k:45s} acc={v['point']:.3f}  CI=[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]  n={v['n']}")

    print("\n--- Table 1 metric CIs (F1 / AUC / ECE / Brier, 95%) ---")
    for k, cell in sorted(out["table1_metric_ci"].items()):
        for metric, v in cell.items():
            if v.get("ci_lo") is None:
                print(f"  {k:35s} {metric:6s} point={v['point']:.3f}  CI=insufficient valid resamples")
            else:
                print(f"  {k:35s} {metric:6s} point={v['point']:.3f}  CI=[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")

    print("\n--- Uplift CIs (B - A, paired, 95%) ---")
    for k, v in sorted(out["uplift_ci"].items()):
        flag = "  <-- excludes zero" if v["excludes_zero"] else ""
        print(f"  {k:35s} uplift={v['point']:+.3f}  CI=[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]{flag}")

    print("\n--- S2 pooled uplift CIs (genuinely pooled across both S2 variants, 95%) ---")
    for k, v in sorted(out["s2_pooled_uplift_ci"].items()):
        flag = "  <-- excludes zero" if v["excludes_zero"] else ""
        print(f"  {k:12s} uplift={v['point']:+.3f}  CI=[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]{flag}")

    print("\n--- Bias-gap CIs (S1 - mean(S2), unpaired, 95%) ---")
    for k, v in sorted(out["bias_gap_ci"].items()):
        flag = "  <-- excludes zero" if v["excludes_zero"] else ""
        print(f"  {k:20s} gap={v['point']:+.3f}  CI=[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]{flag}")

    print("\n--- Self-calibration Pearson r CIs (pooled over scenarios, 95%) ---")
    for k, v in sorted(out["self_calibration_ci"].items()):
        flag = "  <-- excludes zero" if v["excludes_zero"] else ""
        print(f"  {k:20s} r={v['point']:+.3f}  CI=[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]  n={v['n']}{flag}")

    print("\n--- Self-calibration Pearson r CIs (mean of per-scenario r, 95%) ---")
    for k, v in sorted(out["self_calibration_meanofscenarios_ci"].items()):
        flag = "  <-- excludes zero" if v["excludes_zero"] else ""
        print(f"  {k:20s} r={v['point']:+.3f}  CI=[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]  n/scen={v['n_per_scenario']}{flag}")

    print("\n--- Compactness CIs (pooled over scenarios, 95%) ---")
    for k, v in sorted(out["compactness_ci"].items()):
        print(f"  {k:20s} comp={v['point']:.3f}  CI=[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]  n={v['n']}")

    print("\n--- Forensic-specificity shift CIs (B - A, paired, pooled over scenarios, 95%) ---")
    for k, v in sorted(out["spec_shift_ci"].items()):
        flag = "  <-- excludes zero" if v["excludes_zero"] else ""
        print(f"  {k:12s} shift={v['point']:+.4f}  CI=[{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]{flag}")

    print("\n--- Contrastive sensitivity CIs (pooled over conditions, 95%) — 'Gemini...most, GPT...least' ---")
    for k, v in sorted(out["sensitivity_ci"].items()):
        print(f"  {k:15s} sens={v['point']:.3f}  CI=[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]  n_pairs={v['n_pairs']}")


if __name__ == "__main__":
    main()
