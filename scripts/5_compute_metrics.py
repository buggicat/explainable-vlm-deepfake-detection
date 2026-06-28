#!/usr/bin/env python3
"""Score evaluation JSONL: classification, calibration, explainability, faithfulness,
hallucination, tool-use, and cross-condition consistency metrics.

Writes a per-record scored JSONL and a summary JSON aggregating by
(provider, scenario, condition).

New metrics vs v1:
  - MCC (Matthews Correlation Coefficient)
  - Cohen's Kappa
  - BLEU-1 (unigram precision on raw_text; cross-condition A vs B for same image)
  - Semantic cosine similarity (sentence-transformers; graceful fallback if not installed)
  - Explanation word count, forensic term count, evidence count
  - Decision pathway steps (from new decision_pathway field)
  - Alternative hypothesis presence
  - Plain-language summary presence (XAI completeness flag)
  - Numeric grounding faithfulness (claimed evidence.result floats vs tool_trace outputs)
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score,
    average_precision_score,
)
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------------
# Forensic vocabulary — terms from the 54-technique taxonomy that indicate
# the model is actually referencing forensic knowledge
# ---------------------------------------------------------------------------
FORENSIC_TERMS = re.compile(
    r"\b("
    r"ela|error.level.analysis|"
    r"fft|fourier|discrete.fourier|"
    r"dct|discrete.cosine|"
    r"wavelet|pywavelets|"
    r"noise.residual|noise.map|prnu|photo.response.non.uniformity|"
    r"exif|metadata|"
    r"power.spectrum|spectral|radial.energy|diagonal.frequen|"
    r"quantization|quantisation|jpeg.artifact|compression.artifact|block.artifact|"
    r"gan.fingerprint|diffusion.fingerprint|upsampling.artifact|"
    r"inpaint|splicing|copy.move|face.swap|"
    r"illumination|gradient.consistency|perspective|vanishing.point|"
    r"chromatic.aberration|lens.distortion|depth.of.field|"
    r"histogram|entropy|kurtosis|skewness|"
    r"laplacian|sobel|canny|"
    r"skin.texture|hair.artifact|symmetry.artifact|"
    r"scikit.image|skimage|opencv|cv2|scipy|numpy|pillow|matplotlib"
    r")\b",
    re.IGNORECASE,
)

# Manipulation-specific forensic signals (model is doing real forensics)
MANIPULATION_TERMS = re.compile(
    r"\b("
    r"inpaint|splicing|copy.move|face.swap|clone.stamp|"
    r"gan.fingerprint|diffusion.fingerprint|generative.artifact|"
    r"fft|fourier|dct|wavelet|frequency.domain|spectral|power.spectrum|"
    r"noise.residual|prnu|photo.response|compression.artifact|quantization.artifact|"
    r"ela|error.level|block.artifact|double.jpeg|"
    r"upsampling.artifact|checkerboard|aliasing|ringing"
    r")\b",
    re.IGNORECASE,
)

# Content/appearance terms — may reflect dataset bias rather than forensics
CONTENT_TERMS = re.compile(
    r"\b("
    r"lighting|illuminat|shadow|background|pose|expression|"
    r"hair.color|eye.color|skin.tone|complexion|"
    r"makeup|glasses|jewelry|clothing|"
    r"smile|neutral|happy|sad|angry|"
    r"young|old|age|gender|race|ethnicity"
    r")\b",
    re.IGNORECASE,
)

# Logical connectives — distinguish connected reasoning from observation lists
CONNECTIVE_TERMS = re.compile(
    r"\b(therefore|because|which indicates|which suggests|thus|hence|consequently|"
    r"as a result|this suggests|this means|implying|due to|given that|since|"
    r"however|although|nevertheless|despite|in contrast|on the other hand|"
    r"this supports|this argues|this points to|consistent with|inconsistent with)\b",
    re.IGNORECASE,
)

HALLUC_PATTERNS = [
    (re.compile(r"\b(frame rate|fps|temporal|frames per second)\b", re.I),
     "video_claim_on_image"),
    (re.compile(r"\bI cannot (determine|analy|identify|tell)\b", re.I),
     "refusal_pattern"),
]


# ---------------------------------------------------------------------------
# Sentence-transformers (optional — skip gracefully if not installed)
# ---------------------------------------------------------------------------
_SBERT_MODEL = None
_SBERT_AVAILABLE = None


def _get_sbert():
    global _SBERT_MODEL, _SBERT_AVAILABLE
    if _SBERT_AVAILABLE is False:
        return None
    if _SBERT_MODEL is not None:
        return _SBERT_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _SBERT_AVAILABLE = True
        return _SBERT_MODEL
    except Exception:
        _SBERT_AVAILABLE = False
        return None


def semantic_cosine(text_a: str, text_b: str) -> float | None:
    model = _get_sbert()
    if model is None or not text_a or not text_b:
        return None
    embs = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


# ---------------------------------------------------------------------------
# BLEU-1 (unigram precision, no external dependency)
# ---------------------------------------------------------------------------
def bleu1(hypothesis: str, reference: str) -> float:
    """Unigram precision: fraction of hypothesis words that appear in reference."""
    hyp = hypothesis.lower().split()
    if not hyp:
        return 0.0
    ref_counts: dict[str, int] = {}
    for w in reference.lower().split():
        ref_counts[w] = ref_counts.get(w, 0) + 1
    matched = sum(min(hyp.count(w), ref_counts.get(w, 0)) for w in set(hyp))
    return matched / len(hyp)


# ---------------------------------------------------------------------------
# Per-record helpers
# ---------------------------------------------------------------------------
def load_records(path: Path):
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


_NUM_RE = re.compile(
    r"[-+]?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def _is_computational_evidence(tool: str) -> bool:
    """True when evidence claims a code/library tool (not pure visual observation)."""
    t = (tool or "").lower()
    return bool(t) and "visual" not in t


def _trace_output_corpus(trace: list[dict]) -> str:
    """Concatenate executable outputs from a tool trace (not model prose).

    Provider formats differ:
      - Gemini: code_result.payload.output
      - OpenAI: code.payload.outputs (often empty in current capture)
      - Anthropic: code_result.payload.content (stdout tuples)
    """
    parts: list[str] = []
    for ev in trace or []:
        payload = ev.get("payload", {})
        if not isinstance(payload, dict):
            parts.append(str(payload))
            continue
        if payload.get("output"):
            parts.append(str(payload["output"]))
        for out in payload.get("outputs") or []:
            parts.append(json.dumps(out) if isinstance(out, (dict, list)) else str(out))
        for item in payload.get("content") or []:
            parts.append(str(item))
    return "\n".join(parts)


def _floats_in_text(text: str) -> list[float]:
    vals: list[float] = []
    for m in _NUM_RE.finditer(text or ""):
        try:
            vals.append(float(m.group()))
        except ValueError:
            pass
    return vals


def _num_variants(value: float) -> set[str]:
    """String forms a model might copy from code output."""
    variants = {
        str(value),
        f"{value:.12g}",
        f"{value:.6f}".rstrip("0").rstrip("."),
        f"{value:.4f}".rstrip("0").rstrip("."),
        f"{value:.2f}".rstrip("0").rstrip("."),
        f"{value:e}",
        f"{value:.4e}",
        f"{value:.6e}",
    }
    return {v for v in variants if v}


def _float_grounded(value: float, corpus: str, corpus_floats: list[float] | None = None) -> bool:
    """Return True if value appears in corpus (exact string or numeric tolerance)."""
    if any(v in corpus for v in _num_variants(value)):
        return True
    pool = corpus_floats if corpus_floats is not None else _floats_in_text(corpus)
    return any(
        abs(value) > 1 and math.isclose(value, v, rel_tol=1e-3, abs_tol=1e-6)
        or abs(value) <= 1 and math.isclose(value, v, rel_tol=1e-2, abs_tol=1e-4)
        for v in pool
    )


def _numeric_grounding_for_item(result: str, corpus: str, corpus_floats: list[float]) -> dict:
    """Per evidence item: fraction of claimed floats found in trace output."""
    claimed = _floats_in_text(result)
    if not claimed:
        return {"score": None, "n_claimed": 0, "n_grounded": 0}
    grounded = sum(1 for v in claimed if _float_grounded(v, corpus, corpus_floats))
    return {
        "score": grounded / len(claimed),
        "n_claimed": len(claimed),
        "n_grounded": grounded,
    }


def faithfulness_score(parsed: dict | None, trace: list[dict]) -> dict:
    """Audit claimed evidence against the execution trace.

    Two complementary checks:
      1. tool_presence — declared library/tool name tokens appear in trace
      2. numeric_grounding — numeric values in evidence.result appear in trace outputs

    Combined score = mean of available sub-scores (both when numeric claims exist).
    """
    empty = {
        "score": None,
        "tool_presence_score": None,
        "numeric_grounding_score": None,
        "n_evidence": 0,
        "n_matched": 0,
        "n_numeric_evidence": 0,
        "n_numeric_matched": 0,
        "trace_has_output": False,
    }
    if not parsed or not isinstance(parsed.get("evidence"), list):
        return empty
    evidence = parsed["evidence"]
    comp_items = [
        ev for ev in evidence
        if isinstance(ev, dict) and _is_computational_evidence(ev.get("tool", ""))
    ]
    if not comp_items:
        return empty

    trace_text = json.dumps(trace).lower() if trace else ""
    output_corpus = _trace_output_corpus(trace)
    corpus_floats = _floats_in_text(output_corpus)
    trace_has_output = bool(output_corpus.strip())

    tool_matched = 0
    numeric_item_scores: list[float] = []
    numeric_items_with_claims = 0
    numeric_items_fully_grounded = 0

    for ev in comp_items:
        tool = (ev.get("tool") or "").lower()
        tokens = re.findall(r"[a-z_][a-z0-9_]+", tool)
        if trace and any(tok in trace_text for tok in tokens if len(tok) >= 3):
            tool_matched += 1

        ng = _numeric_grounding_for_item(str(ev.get("result", "")), output_corpus, corpus_floats)
        if ng["n_claimed"] > 0:
            numeric_items_with_claims += 1
            if ng["score"] == 1.0:
                numeric_items_fully_grounded += 1
            numeric_item_scores.append(ng["score"])

    n_comp = len(comp_items)
    tool_presence = (tool_matched / n_comp) if trace else (0.0 if comp_items else None)
    numeric_grounding = (
        float(mean(numeric_item_scores)) if numeric_item_scores else None
    )

    sub_scores = [s for s in (tool_presence, numeric_grounding) if s is not None]
    combined = float(mean(sub_scores)) if sub_scores else None

    return {
        "score": combined,
        "tool_presence_score": tool_presence,
        "numeric_grounding_score": numeric_grounding,
        "n_evidence": n_comp,
        "n_matched": tool_matched,
        "n_numeric_evidence": numeric_items_with_claims,
        "n_numeric_matched": numeric_items_fully_grounded,
        "trace_has_output": trace_has_output,
    }


def hallucination_score(parsed: dict | None, raw_text: str,
                        trace: list[dict]) -> dict:
    hits = []
    blob = raw_text or ""
    if parsed:
        blob += "\n" + json.dumps(parsed)
    for pat, name in HALLUC_PATTERNS:
        if pat.search(blob):
            hits.append(name)
    has_code = any(e.get("kind") in ("code", "code_result") for e in (trace or []))
    if parsed and not has_code:
        for ev in (parsed.get("evidence") or []):
            res = str((ev or {}).get("result", ""))
            if re.search(r"\d+\.\d+", res):
                hits.append("numeric_claim_no_code")
                break
    return {"hits": hits, "n_hits": len(hits)}


_EVIDENCE_FIELDS = ("tool", "result", "forensic_category", "interpretation", "why", "how")

def _evidence_completeness(evidence: list) -> float | None:
    """Mean fraction of required schema fields populated across evidence items."""
    if not evidence:
        return None
    scores = []
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        filled = sum(1 for f in _EVIDENCE_FIELDS if ev.get(f) and str(ev[f]).strip())
        scores.append(filled / len(_EVIDENCE_FIELDS))
    return float(mean(scores)) if scores else None


_ERROR_PAT = re.compile(r"\b(error|exception|traceback|stderr)\b", re.I)


def tool_stats(trace: list[dict]) -> dict:
    if not trace:
        return {"n_code": 0, "n_web": 0, "tool_success_rate": None}
    n_code = sum(1 for e in trace if e.get("kind") == "code")
    n_web = sum(1 for e in trace if e.get("kind") == "web_search")
    code_results = [e for e in trace if e.get("kind") == "code_result"]
    if code_results:
        successes = sum(
            1 for e in code_results
            if not _ERROR_PAT.search(str(e.get("output", "")))
        )
        tool_success_rate = successes / len(code_results)
    else:
        tool_success_rate = None
    return {"n_code": n_code, "n_web": n_web, "tool_success_rate": tool_success_rate}


_INFLUENCE_PAT = re.compile(
    r"([+-]?\s*\d+\.?\d*)\s+toward\s+(fake|real)", re.IGNORECASE
)


def evidence_verdict_alignment(parsed: dict | None, classification: str) -> dict:
    """Net signed influence sum vs final verdict.

    Parses each evidence item's 'influence' field (e.g. '+0.08 toward fake',
    '-0.12 toward fake'). Converts to a fake-score (positive = toward fake).
    Checks whether the net direction agrees with the stated classification.

    Source: Jacovi & Goldberg (2020) 'Towards Faithfully Interpretable NLP Systems'.
    """
    evidence = (parsed or {}).get("evidence") or []
    influences = []
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        m = _INFLUENCE_PAT.search(str(ev.get("influence", "")))
        if not m:
            continue
        val = float(m.group(1).replace(" ", ""))
        direction = m.group(2).lower()
        influences.append(-val if direction == "real" else val)

    if not influences:
        return {"score": None, "net_influence": None, "n_parsed": 0}

    net = sum(influences)
    if classification in ("real", "fake"):
        aligned = (net > 0) == (classification == "fake")
        return {"score": 1.0 if aligned else 0.0,
                "net_influence": round(net, 4),
                "n_parsed": len(influences)}
    return {"score": None, "net_influence": round(net, 4), "n_parsed": len(influences)}


def reasoning_connectivity(raw_text: str) -> dict:
    """Logical connectives per sentence — measures connected vs list-style reasoning.

    Heuristic (own); connective density as a lightweight proxy for connected reasoning.
    """
    text = raw_text or ""
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 10]
    n_sentences = max(len(sentences), 1)
    connective_count = len(CONNECTIVE_TERMS.findall(text))
    return {
        "connective_count": connective_count,
        "connectivity_per_sentence": round(connective_count / n_sentences, 4),
    }


def compactness_score(parsed: dict | None) -> dict:
    """Penalise redundant evidence items that repeat the same forensic category.

    Score = unique_categories / total_items. Score of 1.0 means no redundancy.
    Descriptive redundancy measure ; not a validated construct.
    """
    evidence = (parsed or {}).get("evidence") or []
    if not evidence:
        return {"score": None, "unique_categories": 0, "total_items": 0, "redundant_items": 0}
    cats = [
        ev.get("forensic_category", "").strip().lower()
        for ev in evidence
        if isinstance(ev, dict) and ev.get("forensic_category")
    ]
    total = len(cats)
    unique = len(set(cats))
    redundant = total - unique
    return {
        "score": round(unique / max(total, 1), 4),
        "unique_categories": unique,
        "total_items": total,
        "redundant_items": redundant,
    }


def explanation_metrics(raw_text: str, parsed: dict | None) -> dict:
    """Word count, forensic term count, decision pathway steps, XAI completeness."""
    text = raw_text or ""
    word_count = len(text.split())
    forensic_count = len(FORENSIC_TERMS.findall(text))
    manipulation_count = len(MANIPULATION_TERMS.findall(text))
    content_count = len(CONTENT_TERMS.findall(text))

    pathway = (parsed or {}).get("decision_pathway")
    pathway_steps = len(pathway) if isinstance(pathway, list) else 0

    alt_hyp = (parsed or {}).get("alternative_hypothesis")
    has_alternative_hypothesis = bool(
        alt_hyp and isinstance(alt_hyp, str) and len(alt_hyp.strip()) > 10
    )

    self_eval = (parsed or {}).get("self_evaluation") or {}
    plain = self_eval.get("plain_language_summary")
    has_plain_language = bool(
        plain and isinstance(plain, str) and len(plain.strip()) > 20
    )

    evidence = (parsed or {}).get("evidence") or []
    evidence_count = len(evidence)

    cats = [
        ev.get("forensic_category", "")
        for ev in evidence
        if isinstance(ev, dict) and ev.get("forensic_category")
    ]
    unique_forensic_categories = len(set(cats))

    completeness = _evidence_completeness(evidence)

    return {
        "word_count": word_count,
        "forensic_term_count": forensic_count,
        "forensic_specificity_ratio": forensic_count / max(word_count, 1),
        "manipulation_term_count": manipulation_count,
        "content_term_count": content_count,
        "pathway_steps": pathway_steps,
        "has_alternative_hypothesis": has_alternative_hypothesis,
        "has_plain_language": has_plain_language,
        "evidence_count": evidence_count,
        "unique_forensic_categories": unique_forensic_categories,
        # INTERNAL ONLY. Ratio normalized by the full 9-category taxonomy. NOTE: the
        # prompt's forensic_category enum exposes fewer categories than this (5 in the
        # baseline, 8 in the agentic prompt; promptv6_{baseline,agentic}.txt), so this
        # ratio under-counts and is NOT comparable across conditions. The paper reports
        # the raw distinct-category count (unique_forensic_categories) against the
        # per-condition offered set instead; see scripts/7_export_metric_tables.py.
        "category_coverage_ratio": round(unique_forensic_categories / 9, 4),
        "evidence_completeness_score": completeness,
    }


def score_record(r: dict) -> dict:
    label = r.get("label")
    cls = r.get("classification")
    conf = r.get("confidence")
    parsed = r.get("parsed_json")
    trace = r.get("tool_trace") or []
    correct = (
        int(cls == label)
        if cls in ("real", "fake") and label in ("real", "fake")
        else None
    )

    # confidence is defined in the prompt as P(fake) directly (0=certain real, 1=certain fake).
    p_fake = float(conf) if isinstance(conf, (int, float)) else None

    faith = faithfulness_score(parsed, trace)
    halluc = hallucination_score(parsed, r.get("raw_text", ""), trace)
    tools = tool_stats(trace)
    expl = explanation_metrics(r.get("raw_text", ""), parsed)
    ev_align = evidence_verdict_alignment(parsed, cls or "")
    connectivity = reasoning_connectivity(r.get("raw_text", ""))
    compact = compactness_score(parsed)

    self_eval = (parsed or {}).get("self_evaluation") or {}
    self_conf_mean = None
    rels = self_eval.get("reliability_per_evidence")
    if isinstance(rels, list) and rels:
        try:
            self_conf_mean = float(mean(
                float(x) for x in rels if isinstance(x, (int, float))
            ))
        except (TypeError, ValueError):
            pass

    scored = {
        "correct": correct,
        "p_fake": p_fake,
        "faithfulness": faith,
        "hallucination": halluc,
        "tool_stats": tools,
        "self_conf_mean": self_conf_mean,
        "explanation": expl,
        "evidence_verdict_alignment": ev_align,
        "reasoning_connectivity": connectivity,
        "compactness": compact,
    }
    return {**r, "scored": scored}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def expected_calibration_error(labels, probs, n_bins: int = 10) -> float:
    arr = np.array(probs, dtype=float)
    lab = np.array(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(labels)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (arr >= lo) & (arr <= hi) if i == n_bins - 1 else (arr >= lo) & (arr < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(float(arr[mask].mean()) - float(lab[mask].mean()))
    return ece


# ---------------------------------------------------------------------------
# Cross-condition BLEU-1 and semantic cosine (A vs B, same image)
# ---------------------------------------------------------------------------
def cross_condition_metrics(records: list[dict]) -> dict:
    """
    For each (image_id, provider, scenario) pair that has both condition A and B,
    compute BLEU-1 and semantic cosine between their raw_text outputs.
    Returns mean values across all matched pairs.
    """
    by_key: dict[tuple, dict] = {}
    for r in records:
        key = (r.get("image_id"), r.get("provider"), r.get("scenario"))
        cond = r.get("condition")
        if cond in ("A", "B"):
            by_key.setdefault(key, {})[cond] = r.get("raw_text") or ""

    bleu_scores = []
    cosine_scores = []
    for cond_map in by_key.values():
        if "A" not in cond_map or "B" not in cond_map:
            continue
        ta, tb = cond_map["A"], cond_map["B"]
        bleu_scores.append(bleu1(tb, ta))
        c = semantic_cosine(ta, tb)
        if c is not None:
            cosine_scores.append(c)

    result: dict = {"n_pairs": len(bleu_scores)}
    if bleu_scores:
        result["cross_condition_bleu1_mean"] = float(mean(bleu_scores))
    if cosine_scores:
        result["cross_condition_semantic_cosine_mean"] = float(mean(cosine_scores))
    return result


# ---------------------------------------------------------------------------
# Contrastive sensitivity (Yeh et al. 2019, NeurIPS)
# ---------------------------------------------------------------------------
_S2_PREFIXES = ("s2sc_real_", "s2sc_fake_", "s2li_real_", "s2li_fake_")


def _s2_base(image_id: str) -> str | None:
    """Return base ID for S2 records, None for S1."""
    for pfx in _S2_PREFIXES:
        if image_id.startswith(pfx):
            return image_id[len(pfx):]
    return None


def contrastive_sensitivity(records: list[dict]) -> dict:
    """For each S2 content-matched real/fake pair, compute explanation divergence.

    divergence = 1 - cosine_similarity(embed(real_expl), embed(fake_expl))

    High divergence means the model explained real and fake differently —
    i.e., it noticed manipulation-specific features rather than describing
    shared content both times.

    Source: Yeh et al. (2019) 'On the (In)fidelity and Sensitivity of Explanations',
    NeurIPS 2019.
    """
    pairs: dict[tuple, dict] = defaultdict(dict)
    for r in records:
        base = _s2_base(r.get("image_id", ""))
        if base is None:
            continue
        key = (r.get("provider"), r.get("scenario"), base, r.get("condition"))
        pairs[key][r.get("label")] = r.get("raw_text") or ""

    divergences = []
    for cond_map in pairs.values():
        if "real" not in cond_map or "fake" not in cond_map:
            continue
        sim = semantic_cosine(cond_map["real"], cond_map["fake"])
        if sim is not None:
            divergences.append(1.0 - sim)

    if not divergences:
        return {}
    return {
        "n_pairs": len(divergences),
        "mean_divergence": round(float(mean(divergences)), 4),
        "interpretation": (
            "higher = model explains real vs fake differently (detects manipulation cues); "
            "lower = model describes shared content and flips verdict without distinct reasoning"
        ),
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def aggregate(records: list[dict]) -> dict:
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        groups[(r["provider"], r["scenario"], r["condition"])].append(r)

    summary: dict = {}
    for (prov, scen, cond), recs in groups.items():
        labels, preds, probs = [], [], []
        for r in recs:
            s = r.get("scored", {})
            if s.get("correct") is None:
                continue
            labels.append(1 if r["label"] == "fake" else 0)
            preds.append(1 if r["classification"] == "fake" else 0)
            probs.append(float(s["p_fake"]) if s.get("p_fake") is not None else None)

        n = len(labels)
        if n == 0:
            continue

        metrics: dict = {
            "n": n,
            "model": recs[0].get("model"),
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "mcc": float(matthews_corrcoef(labels, preds)),
            "cohen_kappa": float(cohen_kappa_score(labels, preds)),
            "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        }

        usable = [(l, p) for l, p in zip(labels, probs) if p is not None]
        if len(usable) >= 2 and len({l for l, _ in usable}) > 1:
            ls = [l for l, _ in usable]
            ps = [p for _, p in usable]
            try:
                metrics["auc"] = float(roc_auc_score(ls, ps))
                metrics["pr_auc"] = float(average_precision_score(ls, ps))
                metrics["log_loss"] = float(log_loss(ls, np.clip(ps, 1e-6, 1 - 1e-6)))
                metrics["brier_score"] = float(np.mean((np.array(ps) - np.array(ls, dtype=float))**2))
                metrics["ece"] = float(expected_calibration_error(ls, ps))
            except ValueError:
                pass

        # Confidence statistics (mean/std — diagnose direction of miscalibration)
        conf_vals = [p for p in probs if p is not None]
        if conf_vals:
            metrics["confidence_mean"] = float(np.mean(conf_vals))
            metrics["confidence_std"] = float(np.std(conf_vals))

        # Faithfulness (combined + sub-scores)
        faith_scores = [
            r["scored"]["faithfulness"]["score"]
            for r in recs
            if r["scored"]["faithfulness"]["score"] is not None
        ]
        if faith_scores:
            metrics["faithfulness_mean"] = float(mean(faith_scores))
        tool_pres = [
            r["scored"]["faithfulness"]["tool_presence_score"]
            for r in recs
            if r["scored"]["faithfulness"]["tool_presence_score"] is not None
        ]
        if tool_pres:
            metrics["tool_presence_mean"] = float(mean(tool_pres))
        num_ground = [
            r["scored"]["faithfulness"]["numeric_grounding_score"]
            for r in recs
            if r["scored"]["faithfulness"]["numeric_grounding_score"] is not None
        ]
        if num_ground:
            metrics["numeric_grounding_mean"] = float(mean(num_ground))
        trace_out_rate = float(mean(
            1 if r["scored"]["faithfulness"]["trace_has_output"] else 0 for r in recs
        ))
        metrics["trace_has_output_rate"] = trace_out_rate

        # Hallucination
        metrics["hallucination_rate"] = float(mean(
            1 if r["scored"]["hallucination"]["n_hits"] > 0 else 0 for r in recs
        ))

        # Tool usage
        metrics["tool_stats_mean"] = {
            "n_code": float(mean(r["scored"]["tool_stats"]["n_code"] for r in recs)),
            "n_web": float(mean(r["scored"]["tool_stats"]["n_web"] for r in recs)),
        }

        # Explanation quality
        expl_fields = [
            "word_count", "forensic_term_count", "forensic_specificity_ratio",
            "manipulation_term_count", "content_term_count",
            "pathway_steps", "evidence_count", "unique_forensic_categories",
        ]
        metrics["explanation_mean"] = {
            field: float(mean(r["scored"]["explanation"][field] for r in recs))
            for field in expl_fields
        }
        metrics["explanation_mean"]["has_alternative_hypothesis_rate"] = float(mean(
            1 if r["scored"]["explanation"]["has_alternative_hypothesis"] else 0
            for r in recs
        ))
        metrics["explanation_mean"]["has_plain_language_rate"] = float(mean(
            1 if r["scored"]["explanation"]["has_plain_language"] else 0
            for r in recs
        ))
        completeness_vals = [
            r["scored"]["explanation"]["evidence_completeness_score"]
            for r in recs
            if r["scored"]["explanation"]["evidence_completeness_score"] is not None
        ]
        if completeness_vals:
            metrics["explanation_mean"]["evidence_completeness_score_mean"] = float(mean(completeness_vals))
        tool_success_vals = [
            r["scored"]["tool_stats"]["tool_success_rate"]
            for r in recs
            if r["scored"]["tool_stats"]["tool_success_rate"] is not None
        ]
        if tool_success_vals:
            metrics["tool_stats_mean"]["tool_success_rate"] = float(mean(tool_success_vals))

        # Self-calibration: correlation between model's self-rated evidence reliability
        # and actual per-record correctness (novel XAI metric)
        self_conf_vals = [
            r["scored"]["self_conf_mean"]
            for r in recs
            if r["scored"]["self_conf_mean"] is not None
        ]
        correctness_for_sc = [
            r["scored"]["correct"]
            for r in recs
            if r["scored"]["self_conf_mean"] is not None and r["scored"]["correct"] is not None
        ]
        if len(self_conf_vals) == len(correctness_for_sc) and len(self_conf_vals) >= 3:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pr, _ = pearsonr(self_conf_vals[:len(correctness_for_sc)], correctness_for_sc)
                    sr, _ = spearmanr(self_conf_vals[:len(correctness_for_sc)], correctness_for_sc)
                import math
                metrics["self_calibration"] = {
                    "pearson_r": None if math.isnan(pr) else float(pr),
                    "spearman_r": None if math.isnan(sr) else float(sr),
                    "n": len(correctness_for_sc),
                }
            except Exception:
                pass

        # Evidence-verdict alignment (Jacovi & Goldberg 2020)
        ev_align_scores = [
            r["scored"]["evidence_verdict_alignment"]["score"]
            for r in recs
            if r["scored"]["evidence_verdict_alignment"]["score"] is not None
        ]
        if ev_align_scores:
            metrics["evidence_verdict_alignment_mean"] = float(mean(ev_align_scores))
            metrics["evidence_verdict_alignment_n"] = len(ev_align_scores)

        # Reasoning connectivity (Wei et al. 2022)
        conn_scores = [
            r["scored"]["reasoning_connectivity"]["connectivity_per_sentence"]
            for r in recs
        ]
        metrics["reasoning_connectivity_mean"] = float(mean(conn_scores))
        metrics["connective_count_mean"] = float(mean(
            r["scored"]["reasoning_connectivity"]["connective_count"] for r in recs
        ))

        # Compactness / redundancy (Miller 2019)
        compact_scores = [
            r["scored"]["compactness"]["score"]
            for r in recs
            if r["scored"]["compactness"]["score"] is not None
        ]
        if compact_scores:
            metrics["compactness_mean"] = float(mean(compact_scores))

        # Category coverage ratio (Bhatt et al. 2020)
        metrics["category_coverage_ratio_mean"] = float(mean(
            r["scored"]["explanation"]["category_coverage_ratio"] for r in recs
        ))

        # Cross-condition metrics (computed at the group level)
        cc = cross_condition_metrics(recs)
        if cc.get("n_pairs", 0) > 0:
            metrics["cross_condition"] = cc

        summary[f"{prov}|{scen}|{cond}"] = metrics

    # Add cross-condition metrics across ALL records (not per-group) for A/B pairs
    all_cc = cross_condition_metrics(records)
    if all_cc.get("n_pairs", 0) > 0:
        summary["__cross_condition_all__"] = all_cc

    # Bias-gap metric: Scenario 1 accuracy − mean(Scenario 2) accuracy per (provider, condition)
    # This is the headline metric of the paper.
    bias_gaps: dict = {}
    for prov in {k.split("|")[0] for k in summary if not k.startswith("__")}:
        for cond in ("A", "B"):
            s1_key = f"{prov}|scenario1|{cond}"
            s2_keys = [k for k in summary if k.startswith(f"{prov}|scenario2") and k.endswith(f"|{cond}")]
            if s1_key not in summary or not s2_keys:
                continue
            s1_acc = summary[s1_key].get("accuracy")
            s2_accs = [summary[k].get("accuracy") for k in s2_keys if summary[k].get("accuracy") is not None]
            if s1_acc is None or not s2_accs:
                continue
            s2_mean = float(np.mean(s2_accs))
            bias_gaps[f"{prov}|cond{cond}"] = {
                "scenario1_acc": s1_acc,
                "scenario2_mean_acc": s2_mean,
                "bias_gap": round(s1_acc - s2_mean, 4),
                "interpretation": (
                    "positive = model relies on dataset bias (inflated score in standard benchmark)"
                    if s1_acc > s2_mean
                    else "negative = model performs better on content-matched pairs"
                ),
            }
    if bias_gaps:
        summary["__bias_gap__"] = bias_gaps

    # Agentic uplift: accuracy delta B − A per (provider, scenario)
    agentic_uplift: dict = {}
    all_provs = {k.split("|")[0] for k in summary if not k.startswith("__")}
    all_scens = {k.split("|")[1] for k in summary if not k.startswith("__")}
    for prov in all_provs:
        for scen in all_scens:
            a_key, b_key = f"{prov}|{scen}|A", f"{prov}|{scen}|B"
            if a_key not in summary or b_key not in summary:
                continue
            a_acc = summary[a_key].get("accuracy")
            b_acc = summary[b_key].get("accuracy")
            if a_acc is None or b_acc is None:
                continue
            uplift = round(b_acc - a_acc, 4)
            agentic_uplift[f"{prov}|{scen}"] = {
                "condition_A_acc": a_acc,
                "condition_B_acc": b_acc,
                "uplift": uplift,
                "interpretation": (
                    "positive = agentic tool use improved accuracy"
                    if uplift > 0
                    else ("no change" if uplift == 0
                          else "negative = agentic tool use hurt accuracy")
                ),
            }
    if agentic_uplift:
        summary["__agentic_uplift__"] = agentic_uplift

    # Contrastive sensitivity: S2 real/fake pair explanation divergence (Yeh et al. 2019)
    cs = contrastive_sensitivity(records)
    if cs:
        summary["__contrastive_sensitivity__"] = cs

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="Per-record scored JSONL")
    ap.add_argument("--summary", type=Path, default=None,
                    help="Summary JSON (default: <out>.summary.json)")
    args = ap.parse_args()

    sbert = _get_sbert()
    if sbert is None:
        print("sentence-transformers not available — semantic cosine will be skipped. "
              "Install with: pip install sentence-transformers")
    else:
        print("sentence-transformers loaded — semantic cosine enabled.")

    records = [score_record(r) for r in load_records(args.inp)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")

    summary = aggregate(records)
    summary_path = args.summary or args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.out} ({len(records)} records)")
    print(f"Wrote {summary_path}")

    # Print quick summary table
    print("\n--- Classification metrics ---")
    for key, m in summary.items():
        if key.startswith("__"):
            continue
        prov, scen, cond = key.split("|")
        print(f"  {prov:12s} {scen:30s} cond={cond}  "
              f"acc={m.get('accuracy', float('nan')):.3f}  "
              f"f1={m.get('f1', float('nan')):.3f}  "
              f"mcc={m.get('mcc', float('nan')):.3f}  "
              f"auc={m.get('auc', float('nan')):.3f}")


if __name__ == "__main__":
    main()
