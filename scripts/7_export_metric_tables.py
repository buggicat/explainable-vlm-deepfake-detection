#!/usr/bin/env python3
"""Generate paper result tables directly from the scored summaries.

Reads:
  results/vlm_eval_scored.summary.json    (classification + explainability metrics)
  results/explanation_audit_summary.json  (independent groundedness audit)

Currently emits the Section 4.1 / 5.4 "Faithfulness & Grounding" tables
(markdown + LaTeX). Values are means across the scenarios present for each
model, in the agentic condition (B) where these metrics are defined.

Usage:
  python scripts/7_export_metric_tables.py
  python scripts/7_export_metric_tables.py --results-dir results --latex
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

PROVIDER_ORDER = ["openai", "gemini", "anthropic"]

# Paper-wide short labels (defined once in §3.4, used in all tables).
MODEL_ABBREV = {
    "gpt-5.4-mini": "GPT",
    "gemini-2.5-pro": "Gemini",
    "claude-sonnet-4-6": "Claude",
}


def ab(model: str) -> str:
    return MODEL_ABBREV.get(model, model)


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def faithfulness_rows(summary: dict) -> tuple[list, dict]:
    """Per-provider mean of faithfulness/tool/numeric over condition-B scenarios.

    Returns (rows, per_scenario) where per_scenario logs the exact cells used.
    """
    acc = defaultdict(lambda: defaultdict(list))   # provider -> field -> [values]
    model_of = {}
    per_scenario = defaultdict(list)
    for key, m in summary.items():
        if key.startswith("__") or key.count("|") != 2:
            continue
        prov, scen, cond = key.split("|")
        if cond != "B":
            continue
        if "faithfulness_mean" not in m:
            continue
        model_of[prov] = m.get("model", prov)
        for field in ("faithfulness_mean", "tool_presence_mean", "numeric_grounding_mean"):
            if m.get(field) is not None:
                acc[prov][field].append(m[field])
                per_scenario[prov].append((scen, field, m[field]))
    rows = []
    for prov in PROVIDER_ORDER:
        if prov not in acc:
            continue
        rows.append({
            "model": model_of[prov],
            "faith": mean(acc[prov]["faithfulness_mean"]),
            "tool": mean(acc[prov]["tool_presence_mean"]),
            "numeric": mean(acc[prov]["numeric_grounding_mean"]),
            "n_scen": len(acc[prov]["faithfulness_mean"]),
        })
    return rows, per_scenario


def groundedness_rows(judge: dict) -> list:
    """Per-model mean groundedness over scenarios (condition B)."""
    by_model = defaultdict(list)
    for key, v in judge.get("by_group", {}).items():
        model = key.split("|")[0]
        by_model[model].append(v["groundedness_mean"])
    rows = [{"model": mdl, "grounded": mean(vals), "n": len(vals)}
            for mdl, vals in by_model.items()]
    rows.sort(key=lambda r: r["model"])
    return rows


def hallucination_programmatic_rows(summary: dict) -> tuple[list, dict]:
    """Per-provider mean rule-based hallucination_rate, condition A and B."""
    acc = defaultdict(lambda: {"A": [], "B": []})
    model_of = {}
    per = defaultdict(list)
    for key, m in summary.items():
        if key.startswith("__") or key.count("|") != 2:
            continue
        prov, scen, cond = key.split("|")
        if cond not in ("A", "B") or m.get("hallucination_rate") is None:
            continue
        model_of[prov] = m.get("model", prov)
        acc[prov][cond].append(m["hallucination_rate"])
        per[prov].append((scen, cond, m["hallucination_rate"]))
    rows = []
    for prov in PROVIDER_ORDER:
        if prov not in acc:
            continue
        a, b = acc[prov]["A"], acc[prov]["B"]
        rows.append({"model": model_of[prov],
                     "a": mean(a) if a else None,
                     "b": mean(b) if b else None})
    return rows, per


def judge_hallucination_rows(jsonl_path: Path) -> tuple[list, float | None]:
    """Per-model LLM-judge hallucination_detected rate from the raw judge JSONL."""
    by_model = defaultdict(list)
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            hd = (r.get("judge_parsed") or {}).get("hallucination_detected")
            if isinstance(hd, bool):
                by_model[r.get("model") or r.get("provider")].append(1 if hd else 0)
    rows = [{"model": m, "rate": mean(v), "n": len(v)} for m, v in by_model.items()]
    rows.sort(key=lambda r: r["model"])
    allv = [x for v in by_model.values() for x in v]
    return rows, (mean(allv) if allv else None)


def md_halluc(rows) -> str:
    out = ["| Model | Cond A | Cond B |", "|---|---|---|"]
    for r in rows:
        a = f"{r['a']:.3f}" if r["a"] is not None else "—"
        b = f"{r['b']:.3f}" if r["b"] is not None else "—"
        out.append(f"| {ab(r['model'])} | {a} | {b} |")
    return "\n".join(out)


def md_judge_halluc(rows, overall) -> str:
    out = ["| Model | Halluc. (judge) | n |", "|---|---|---|"]
    for r in rows:
        out.append(f"| {ab(r['model'])} | {r['rate']:.3f} | {r['n']} |")
    if overall is not None:
        out.append(f"\noverall judge hallucination-detected rate = {overall:.3f}")
    return "\n".join(out)


REASON_FIELDS = {
    "align": "evidence_verdict_alignment_mean",
    "conn": "reasoning_connectivity_mean",
    "comp": "compactness_mean",
}


def reasoning_rows(summary: dict) -> list:
    """Per-provider means of the 3 reasoning-coherence metrics, conditions A and B."""
    acc = defaultdict(lambda: defaultdict(lambda: {"A": [], "B": []}))
    model_of = {}
    for key, m in summary.items():
        if key.startswith("__") or key.count("|") != 2:
            continue
        prov, scen, cond = key.split("|")
        if cond not in ("A", "B"):
            continue
        model_of[prov] = m.get("model", prov)
        for fk, fn in REASON_FIELDS.items():
            if m.get(fn) is not None:
                acc[prov][fk][cond].append(m[fn])
    rows = []
    for prov in PROVIDER_ORDER:
        if prov not in acc:
            continue
        r = {"model": model_of[prov]}
        for fk in REASON_FIELDS:
            for cond in ("A", "B"):
                vals = acc[prov][fk][cond]
                r[f"{fk}_{cond}"] = mean(vals) if vals else None
        rows.append(r)
    return rows


def md_reasoning(rows, cond) -> str:
    out = [f"| Model | Align ({cond}) | Conn ({cond}) | Comp ({cond}) |", "|---|---|---|---|"]
    for r in rows:
        out.append(f"| {ab(r['model'])} | {r['align_'+cond]:.2f} | {r['conn_'+cond]:.2f} | {r['comp_'+cond]:.2f} |")
    return "\n".join(out)


def md_reasoning_both(rows) -> str:
    """Both conditions in one table: Model | Cond | Align | Conn | Comp."""
    out = ["| Model | Cond | Align | Conn | Comp |", "|---|---|---|---|---|"]
    for r in rows:
        for cond in ("A", "B"):
            out.append(f"| {ab(r['model'])} | {cond} | {r['align_'+cond]:.2f} | "
                       f"{r['conn_'+cond]:.2f} | {r['comp_'+cond]:.2f} |")
    return "\n".join(out)


def selfcal_rows(summary: dict) -> list:
    """Per-provider mean self-calibration (Pearson, Spearman), conditions A and B."""
    acc = defaultdict(lambda: {"A": {"p": [], "s": []}, "B": {"p": [], "s": []}})
    model_of = {}
    for key, m in summary.items():
        if key.startswith("__") or key.count("|") != 2:
            continue
        prov, scen, cond = key.split("|")
        if cond not in ("A", "B"):
            continue
        sc = m.get("self_calibration")
        if not sc:
            continue
        model_of[prov] = m.get("model", prov)
        if sc.get("pearson_r") is not None:
            acc[prov][cond]["p"].append(sc["pearson_r"])
        if sc.get("spearman_r") is not None:
            acc[prov][cond]["s"].append(sc["spearman_r"])
    rows = []
    for prov in PROVIDER_ORDER:
        if prov not in acc:
            continue
        r = {"model": model_of[prov]}
        for cond in ("A", "B"):
            r[f"p_{cond}"] = mean(acc[prov][cond]["p"]) if acc[prov][cond]["p"] else None
            r[f"s_{cond}"] = mean(acc[prov][cond]["s"]) if acc[prov][cond]["s"] else None
        rows.append(r)
    return rows


def md_selfcal(rows) -> str:
    out = ["| Model | Cond | Pearson r | Spearman r |", "|---|---|---|---|"]
    for r in rows:
        for cond in ("A", "B"):
            out.append(f"| {ab(r['model'])} | {cond} | {r['p_'+cond]:.2f} | {r['s_'+cond]:.2f} |")
    return "\n".join(out)


def content_rows(summary: dict) -> list:
    """Per-provider Explanation Content metrics, conditions A and B."""
    fields = {
        "spec": ("explanation_mean", "forensic_specificity_ratio"),
        "manip": ("explanation_mean", "manipulation_term_count"),
        "content": ("explanation_mean", "content_term_count"),
        "cov": (None, "category_coverage_ratio_mean"),
        "compl": ("explanation_mean", "evidence_completeness_score_mean"),
    }
    acc = defaultdict(lambda: defaultdict(lambda: {"A": [], "B": []}))
    model_of = {}
    for key, m in summary.items():
        if key.startswith("__") or key.count("|") != 2:
            continue
        prov, scen, cond = key.split("|")
        if cond not in ("A", "B"):
            continue
        model_of[prov] = m.get("model", prov)
        for fk, (parent, name) in fields.items():
            src = m.get(parent, {}) if parent else m
            if isinstance(src, dict) and src.get(name) is not None:
                acc[prov][fk][cond].append(src[name])
    rows = []
    for prov in PROVIDER_ORDER:
        if prov not in acc:
            continue
        r = {"model": model_of[prov]}
        for fk in fields:
            for cond in ("A", "B"):
                vals = acc[prov][fk][cond]
                r[f"{fk}_{cond}"] = mean(vals) if vals else None
        rows.append(r)
    return rows


def md_content(rows) -> str:
    """Forensic specificity (%), manipulation- and content-term counts; both conditions.

    (Category coverage and evidence completeness are reported in prose: completeness is
    uniformly 1.00 by the enforced schema; coverage is a SUPPORT breadth descriptor.)
    """
    out = ["| Model | Cond | Spec % | Manip | Content |",
           "|---|---|---|---|---|"]
    for r in rows:
        for cond in ("A", "B"):
            out.append(
                f"| {ab(r['model'])} | {cond} | {100*r['spec_'+cond]:.1f} | "
                f"{r['manip_'+cond]:.1f} | {r['content_'+cond]:.1f} |")
    return "\n".join(out)


def parse_consistency(path: Path) -> dict:
    """Parse metric_outputs/table_4_5_consistency.txt (produced on the compute node by
    compute_consistency.py — needs sentence-transformers, so it is NOT recomputed here).

    Returns {abbrev: {bleu, cos, sens_sc, sens_li}}, BLEU-1 and cosine pooled (mean)
    over scenarios, contrastive sensitivity kept per Scenario-2 type.
    """
    if not path.exists():
        return {}
    bleu, cos = defaultdict(list), defaultdict(list)
    sens = defaultdict(dict)
    mode = None
    for line in path.read_text().splitlines():
        if "Cross-condition consistency" in line:
            mode = "cc"; continue
        if "Contrastive sensitivity" in line:
            mode = "cs"; continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        m = cells[0]
        if m not in ("GPT", "Gemini", "Claude"):
            continue
        try:
            if mode == "cc" and len(cells) >= 4:
                bleu[m].append(float(cells[2])); cos[m].append(float(cells[3]))
            elif mode == "cs" and len(cells) >= 3:
                sens[m][cells[1]] = float(cells[2])
        except ValueError:
            continue
    out = {}
    for m in ("GPT", "Gemini", "Claude"):
        if m in bleu:
            out[m] = {"bleu": mean(bleu[m]), "cos": mean(cos[m]),
                      "sens_sc": sens[m].get("S2-SC"), "sens_li": sens[m].get("S2-LI")}
    return out


def md_xai_audit(faith_rows, grounded_rows, jhal_rows, cons) -> str:
    """Combined per-model auditability table (condition B + pooled consistency):
    faithfulness/tool/numeric + LLM-judge groundedness/hallucination + consistency/sensitivity.
    Claude is the LLM judge, so its judge cells are blank."""
    grounded = {ab(r["model"]): r["grounded"] for r in grounded_rows}
    jhal = {ab(r["model"]): r["rate"] for r in jhal_rows}
    out = ["| Model | Tool | Num | Faith | J-Grnd | J-Hal | BLEU | Cos | Sens-SC | Sens-LI |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in faith_rows:
        m = ab(r["model"])
        c = cons.get(m, {})
        g = f"{grounded[m]:.2f}" if m in grounded else "--"
        jh = f"{jhal[m]:.3f}" if m in jhal else "--"
        bleu = f"{c['bleu']:.2f}" if c.get("bleu") is not None else "--"
        cosv = f"{c['cos']:.2f}" if c.get("cos") is not None else "--"
        ssc = f"{c['sens_sc']:.3f}" if c.get("sens_sc") is not None else "--"
        sli = f"{c['sens_li']:.3f}" if c.get("sens_li") is not None else "--"
        out.append(f"| {m} | {r['tool']:.2f} | {r['numeric']:.2f} | {r['faith']:.2f} | "
                   f"{g} | {jh} | {bleu} | {cosv} | {ssc} | {sli} |")
    return "\n".join(out)


def md_xai_cond(reason_rows, sc_rows, cont_rows) -> str:
    """Combined per-model x condition table: reasoning coherence + self-calibration
    (Pearson) + explanation content (specificity %, manipulation/content term counts)."""
    sc = {r["model"]: r for r in sc_rows}
    cont = {r["model"]: r for r in cont_rows}
    out = ["| Model | Cond | Align | Conn | Comp | Pearson | Spec% | Manip | Content |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in reason_rows:
        m = r["model"]
        s, ct = sc.get(m, {}), cont.get(m, {})
        for cond in ("A", "B"):
            out.append(f"| {ab(m)} | {cond} | {r['align_'+cond]:.2f} | {r['conn_'+cond]:.2f} | "
                       f"{r['comp_'+cond]:.2f} | {s.get('p_'+cond):.2f} | "
                       f"{100*ct.get('spec_'+cond):.1f} | {ct.get('manip_'+cond):.1f} | "
                       f"{ct.get('content_'+cond):.1f} |")
    return "\n".join(out)


CLS_FIELDS = ("accuracy", "f1", "auc", "ece", "brier_score")


def classification_rows(summary: dict) -> list:
    """Per (provider, scenario) classification + calibration, both conditions."""
    acc = defaultdict(lambda: defaultdict(dict))  # (prov,scen) -> cond -> {field:val}
    model_of = {}
    for key, m in summary.items():
        if key.startswith("__") or key.count("|") != 2:
            continue
        prov, scen, cond = key.split("|")
        if cond not in ("A", "B"):
            continue
        model_of[prov] = m.get("model", prov)
        acc[(prov, scen)][cond] = {f: m.get(f) for f in CLS_FIELDS}
    rows = []
    for prov in PROVIDER_ORDER:
        for scen in SCEN_ORDER:
            k = (prov, scen)
            if k not in acc:
                continue
            rows.append({"model": model_of[prov], "scen": scen,
                         "A": acc[k].get("A", {}), "B": acc[k].get("B", {})})
    return rows


def md_classification(rows) -> str:
    out = ["| Model | Scen | Acc A | Acc B | AUC A | AUC B |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {ab(r['model'])} | {SCEN_ABBR[r['scen']]} | "
                   f"{r['A'].get('accuracy', float('nan')):.2f} | "
                   f"{r['B'].get('accuracy', float('nan')):.2f} | "
                   f"{r['A'].get('auc', float('nan')):.2f} | "
                   f"{r['B'].get('auc', float('nan')):.2f} |")
    return "\n".join(out)


def _f(d, k):
    v = d.get(k)
    return f"{v:.2f}" if isinstance(v, (int, float)) else "--"


def md_classification_full(rows) -> str:
    """Full per-cell detection table (page-wide): acc/F1/AUC/ECE/Brier + agentic uplift.

    Uplift = Acc(B) - Acc(A) is a single per-(model,scenario) value (equivalent to the
    standalone uplift table), so it is shown once, on the B row of each scenario pair.
    """
    out = ["| Model | Scen | Cond | Acc | F1 | AUC | ECE | Brier | Uplift |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        a_acc, b_acc = r["A"].get("accuracy"), r["B"].get("accuracy")
        upl = (b_acc - a_acc) if isinstance(a_acc, (int, float)) and isinstance(b_acc, (int, float)) else None
        for cond in ("A", "B"):
            d = r[cond]
            u = f"{upl:+.3f}" if (cond == "B" and upl is not None) else ""
            out.append(f"| {ab(r['model'])} | {SCEN_ABBR[r['scen']]} | {cond} | "
                       f"{_f(d,'accuracy')} | {_f(d,'f1')} | {_f(d,'auc')} | "
                       f"{_f(d,'ece')} | {_f(d,'brier_score')} | {u} |")
    return "\n".join(out)


SCEN_ORDER = ["scenario1", "scenario2_self_cond", "scenario2_local_inpaint"]
SCEN_ABBR = {"scenario1": "S1", "scenario2_self_cond": "S2-SC", "scenario2_local_inpaint": "S2-LI"}

PROVIDER_ABBR = {"openai": "GPT", "gemini": "Gemini", "anthropic": "Claude"}


def bias_gap_rows(summary: dict) -> list:
    """§5.2 bias gap = scenario1_acc - mean(scenario2_acc), per provider per condition.
    Read straight from the summary's __bias_gap__ block."""
    bg = summary["__bias_gap__"]
    rows = []
    for prov in PROVIDER_ORDER:
        for cond in ("A", "B"):
            k = f"{prov}|cond{cond}"
            if k in bg:
                d = bg[k]
                rows.append({"model": PROVIDER_ABBR[prov], "cond": cond,
                             "s1": d["scenario1_acc"], "s2": d["scenario2_mean_acc"],
                             "gap": d["bias_gap"]})
    return rows


def md_bias_gap(rows) -> str:
    out = ["| Model | Cond | S1 Acc | S2 Acc | Bias gap |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['model']} | {r['cond']} | {r['s1']:.2f} | "
                   f"{r['s2']:.2f} | {r['gap']:+.3f} |")
    return "\n".join(out)


def uplift_rows(summary: dict) -> list:
    """§5.3 agentic uplift = condition_B_acc - condition_A_acc, per provider/scenario,
    from __agentic_uplift__; tool-presence rate read from each B cell."""
    up = summary["__agentic_uplift__"]
    rows = []
    for prov in PROVIDER_ORDER:
        for scen in SCEN_ORDER:
            k = f"{prov}|{scen}"
            if k in up:
                d = up[k]
                bcell = summary.get(f"{prov}|{scen}|B", {})
                rows.append({"model": PROVIDER_ABBR[prov], "scen": scen,
                             "a": d["condition_A_acc"], "b": d["condition_B_acc"],
                             "uplift": d["uplift"],
                             "tool": bcell.get("tool_presence_mean")})
    return rows


def md_uplift(rows) -> str:
    out = ["| Model | Scen | Acc A | Acc B | Uplift | Tool use |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        tool = f"{r['tool']:.2f}" if isinstance(r['tool'], (int, float)) else "--"
        out.append(f"| {r['model']} | {SCEN_ABBR[r['scen']]} | {r['a']:.2f} | "
                   f"{r['b']:.2f} | {r['uplift']:+.3f} | {tool} |")
    return "\n".join(out)


def md_faith(rows) -> str:
    out = ["| Model | Faith. | Tool | Num. |", "|---|---|---|---|"]
    for r in rows:
        out.append(f"| {ab(r['model'])} | {r['faith']:.2f} | {r['tool']:.2f} | {r['numeric']:.2f} |")
    return "\n".join(out)


def md_grounded(rows) -> str:
    out = ["| Model | Grounded. |", "|---|---|"]
    for r in rows:
        out.append(f"| {ab(r['model'])} | {r['grounded']:.2f} |")
    return "\n".join(out)


def latex_faith(rows) -> str:
    lines = [r"\begin{tabular}{lccc}", r"\hline",
             r"Model & Faith. & Tool & Num. \\", r"\hline"]
    for r in rows:
        lines.append(f"{r['model']} & {r['faith']:.2f} & {r['tool']:.2f} & {r['numeric']:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--figures-dir", type=Path, default=Path("metric_outputs"),
                    help="where to save the table text output")
    ap.add_argument("--latex", action="store_true", help="also emit LaTeX")
    ap.add_argument("--no-save", action="store_true", help="print only, do not write file")
    args = ap.parse_args()

    summary = load(args.results_dir / "vlm_eval_scored.summary.json")
    judge = load(args.results_dir / "explanation_audit_summary.json")

    rows, per_scenario = faithfulness_rows(summary)
    grows = groundedness_rows(judge)

    lines: list[str] = []
    lines.append("Section 4.1 / 5.4 — Faithfulness & Grounding tables")
    lines.append("generated by scripts/7_export_metric_tables.py from the scored summaries\n")

    lines.append("=== PROVENANCE: per-scenario condition-B cells used ===")
    for prov, items in per_scenario.items():
        lines.append(f"  {prov}:")
        for scen, field, val in items:
            lines.append(f"    {scen:24s} {field:22s} {val}")

    lines.append("\n=== Table: Faithfulness & grounding (cond. B, mean over scenarios) ===")
    lines.append(md_faith(rows))
    lines.append(f"\noverall judge groundedness_mean = {judge.get('groundedness_mean')}, "
                 f"hallucination_detected_rate = {judge.get('hallucination_detected_rate')}")

    lines.append("\n=== Table: LLM-judge groundedness (cond. B, 1-5) ===")
    lines.append(md_grounded(grows))

    if args.latex:
        lines.append("\n=== LaTeX: faithfulness ===")
        lines.append(latex_faith(rows))

    text = "\n".join(lines)
    print(text)

    # ---- §5.1 Detection performance ----
    clsrows = classification_rows(summary)
    cls_lines = ["Section 5.1 — Detection performance",
                 "generated from results/vlm_eval_scored.summary.json\n",
                 "=== PROVENANCE: acc / f1 / auc / ece / brier per cell ==="]
    for r in clsrows:
        for cond in ("A", "B"):
            d = r[cond]
            cls_lines.append(
                f"  {ab(r['model']):7s} {SCEN_ABBR[r['scen']]:6s} {cond}  "
                f"acc={d.get('accuracy', float('nan')):.4f} f1={d.get('f1', float('nan')):.4f} "
                f"auc={d.get('auc', float('nan')):.4f} ece={d.get('ece', float('nan')):.4f} "
                f"brier={d.get('brier_score', float('nan')):.4f}")
    cls_lines.append("\n=== Table: Detection performance (FULL, page-wide) ===")
    cls_lines.append(md_classification_full(clsrows))
    cls_lines.append("\n=== Table: Detection performance (compact, acc+AUC) ===")
    cls_lines.append(md_classification(clsrows))
    cls_text = "\n".join(cls_lines)
    print("\n" + cls_text)

    # ---- §5.2 Bias gap ----
    bgrows = bias_gap_rows(summary)
    bglines = ["Section 5.2 — The Bias Gap",
               "bias_gap = scenario1_acc - mean(scenario2_acc); from __bias_gap__ block",
               "positive = relies on dataset bias (inflated standard-benchmark score)\n",
               "=== Table: Bias gap (per model, per condition) ==="]
    bglines.append(md_bias_gap(bgrows))
    bg_text = "\n".join(bglines)
    print("\n" + bg_text)

    # ---- §5.3 The Effect of Tool Use ----
    uprows = uplift_rows(summary)
    uplines = ["Section 5.3 — The Effect of Tool Use",
               "uplift = condition_B_acc - condition_A_acc; from __agentic_uplift__",
               "Tool use = tool_presence_mean of the B cell (share of responses showing tool use)\n",
               "=== Table: Agentic uplift (per model, per scenario) ==="]
    uplines.append(md_uplift(uprows))
    up_text = "\n".join(uplines)
    print("\n" + up_text)

    # ---- Family 2: Hallucination (§4.2 / §5.4) ----
    hrows, hper = hallucination_programmatic_rows(summary)
    jrows, joverall = judge_hallucination_rows(args.results_dir / "explanation_audit.jsonl")
    hlines: list[str] = []
    hlines.append("Section 4.2 / 5.4 — Hallucination tables")
    hlines.append("generated by scripts/7_export_metric_tables.py from the scored summary + judge JSONL\n")
    hlines.append("=== PROVENANCE: per-scenario hallucination_rate cells ===")
    for prov, items in hper.items():
        hlines.append(f"  {prov}:")
        for scen, cond, val in items:
            hlines.append(f"    {scen:24s} cond={cond}  {val}")
    hlines.append("\n=== Table: Rule-based hallucination rate (mean over scenarios) ===")
    hlines.append(md_halluc(hrows))
    hlines.append("\n=== Table: LLM-judge hallucination-detected rate (cond. B) ===")
    hlines.append(md_judge_halluc(jrows, joverall))
    htext = "\n".join(hlines)
    print("\n" + htext)

    # ---- Family 2: Reasoning Coherence (§4.2 / §5.4) ----
    rrows = reasoning_rows(summary)
    rlines: list[str] = []
    rlines.append("Section 4.2 / 5.4 — Reasoning Coherence tables")
    rlines.append("generated by scripts/7_export_metric_tables.py from the scored summary\n")
    rlines.append("Align = evidence-verdict alignment, Conn = reasoning connectivity, Comp = compactness")
    rlines.append("\n=== PROVENANCE: per-provider means, both conditions ===")
    for r in rrows:
        rlines.append(f"  {r['model']}:")
        for fk in ("align", "conn", "comp"):
            rlines.append(f"    {fk:6s} A={r[fk+'_A']:.4f}  B={r[fk+'_B']:.4f}")
    rlines.append("\n=== Table: Reasoning coherence (both conditions) ===")
    rlines.append(md_reasoning_both(rrows))
    rlines.append("\n=== Table: Reasoning coherence (agentic condition B only) ===")
    rlines.append(md_reasoning(rrows, "B"))
    rtext = "\n".join(rlines)
    print("\n" + rtext)

    # ---- Family 3: Confidence Reliability / self-calibration (§4.3 / §5.4) ----
    screws = selfcal_rows(summary)
    slines = ["Section 4.3 / 5.4 — Self-calibration table",
              "generated by scripts/7_export_metric_tables.py from the scored summary\n",
              "=== PROVENANCE: per-provider Pearson/Spearman means, both conditions ==="]
    for r in screws:
        slines.append(f"  {r['model']}: "
                      f"A p={r['p_A']:.4f} s={r['s_A']:.4f} | B p={r['p_B']:.4f} s={r['s_B']:.4f}")
    slines.append("\n=== Table: Self-calibration (both conditions) ===")
    slines.append(md_selfcal(screws))
    stext = "\n".join(slines)
    print("\n" + stext)

    # ---- Family 4: Explanation Content (§4.4 / §5.4) ----
    crows = content_rows(summary)
    clines = ["Section 4.4 / 5.4 — Explanation Content table",
              "generated by scripts/7_export_metric_tables.py from the scored summary\n",
              "Spec % = forensic-term share of words; Manip/Content = mean term counts;",
              "Compl = evidence completeness\n",
              "Category coverage (Option A): reported as the raw count of DISTINCT forensic",
              "categories used, out of the set the prompt offered in that condition",
              "(A = 5, B = 8; promptv6_{baseline,agentic}.txt). Derived from the internal",
              "category_coverage_ratio_mean (= unique/9): count = ratio_mean * 9.\n",
              "=== PROVENANCE: per-provider means, both conditions ==="]
    OFFERED = {"A": 5, "B": 8}  # forensic_category enum size per condition (promptv6)
    for r in crows:
        clines.append(f"  {r['model']}:")
        for fk in ("spec", "manip", "content", "compl"):
            clines.append(f"    {fk:8s} A={r[fk+'_A']:.4f}  B={r[fk+'_B']:.4f}")
        cnt_a = r["cov_A"] * 9 if r["cov_A"] is not None else float("nan")
        cnt_b = r["cov_B"] * 9 if r["cov_B"] is not None else float("nan")
        clines.append(f"    cov_cnt  A={cnt_a:.2f} of {OFFERED['A']}  "
                      f"B={cnt_b:.2f} of {OFFERED['B']}  (raw distinct-category count)")
    clines.append("\n=== Table: Explanation content (both conditions) ===")
    clines.append(md_content(crows))
    ctext = "\n".join(clines)
    print("\n" + ctext)

    # ---- §5.4 COLLAPSED page-wide tables (2 combined) ----
    cons = parse_consistency(args.figures_dir / "table_4_5_consistency.txt")
    xlines = ["Section 5.4 — Explanation Quality (collapsed page-wide tables)",
              "Table A combines faithfulness + LLM-judge + consistency/sensitivity (per model);",
              "Table B combines reasoning + self-calibration + content (per model x condition).\n"]
    if not cons:
        xlines.append("[WARN] table_4_5_consistency.txt not found — BLEU/Cos/Sens columns blank.")
    xlines.append("=== Combined Table A: per-model auditability (cond. B + pooled consistency) ===")
    xlines.append(md_xai_audit(rows, grows, jrows, cons))
    xlines.append("\n=== Combined Table B: per-model x condition (reasoning + self-cal + content) ===")
    xlines.append(md_xai_cond(rrows, screws, crows))
    xtext = "\n".join(xlines)
    print("\n" + xtext)

    if not args.no_save:
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        (args.figures_dir / "table_5_1_detection.txt").write_text(cls_text + "\n")
        (args.figures_dir / "table_5_2_biasgap.txt").write_text(bg_text + "\n")
        (args.figures_dir / "table_5_3_uplift.txt").write_text(up_text + "\n")
        (args.figures_dir / "table_5_4_xai_collapsed.txt").write_text(xtext + "\n")
        (args.figures_dir / "table_4_4_content.txt").write_text(ctext + "\n")
        (args.figures_dir / "table_4_3_selfcalibration.txt").write_text(stext + "\n")
        f1 = args.figures_dir / "table_4_1_faithfulness.txt"
        f1.write_text(text + "\n")
        f2 = args.figures_dir / "table_4_1_hallucination.txt"
        f2.write_text(htext + "\n")
        f3 = args.figures_dir / "table_4_2_reasoning.txt"
        f3.write_text(rtext + "\n")
        print(f"\n[saved] {f1}\n[saved] {f2}\n[saved] {f3}")


if __name__ == "__main__":
    main()
