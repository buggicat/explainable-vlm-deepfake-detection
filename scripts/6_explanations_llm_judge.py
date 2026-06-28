#!/usr/bin/env python3
"""LLM-as-judge groundedness audit for agentic forensic explanations.

Uses the Anthropic API (Claude) as an independent judge. For each evaluated
record, the judge receives ONLY:
  - the structured evidence JSON from the model under test
  - the tool execution trace (code outputs, not the model's prose)

The judge scores whether each evidence item is grounded in the trace and
returns an overall 1–5 groundedness rating.

Recommended usage: run Cell 5 in ``paper_results.ipynb``, or from CLI:

  export ANTHROPIC_API_KEY=...
  python scripts/6_explanations_llm_judge.py \\
      --in  results/vlm_eval_scored.jsonl \\
      --out results/explanation_audit.jsonl \\
      --summary results/explanation_audit_summary.json \\
      --sample 50 \\
      --providers openai gemini \\
      --condition B

Resumable: re-running skips (image_id, provider, scenario, condition) keys
already present in --out.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"))

JUDGE_SYSTEM = """You are an independent forensic-audit judge. Your task is to evaluate
whether a vision-language model's EXPLANATION EVIDENCE is grounded in its TOOL EXECUTION
TRACE — not whether the final real/fake verdict is correct.

Rules:
- Use ONLY the tool trace as ground truth for computational claims.
- Visual-observation evidence items (no code) cannot be verified from the trace; mark
  tool_grounded=null and numeric_grounded=null for those.
- For computational items: tool_grounded=yes if the declared library/method appears in
  the trace; numeric_grounded=yes if the specific numbers in "result" appear in trace
  outputs (allow rounding); interpretation_grounded=yes if the interpretation logically
  follows from the trace-supported numbers (not speculative leaps).
- Do NOT reward forensic jargon or confident prose. Penalize invented numbers.
- Respond with EXACTLY ONE fenced JSON block and no other text."""

JUDGE_USER_TEMPLATE = """Audit this forensic analysis record.

MODEL UNDER TEST: {model} | scenario: {scenario} | condition: {condition}
IMAGE ID: {image_id} | label: {label} | model verdict: {classification}

=== EVIDENCE (model's structured claims) ===
{evidence_json}

=== TOOL EXECUTION TRACE (ground truth for computational claims) ===
{trace_text}

Score each evidence item, then give an overall groundedness score 1–5:
  1 = mostly ungrounded / invented numbers
  2 = tools claimed but numbers or interpretations unsupported
  3 = mixed — some items grounded, some not
  4 = mostly grounded with minor gaps
  5 = fully grounded — every computational claim trace-supported

```json
{{
  "per_evidence": [
    {{
      "index": 0,
      "tool_grounded": "yes|no|null",
      "numeric_grounded": "yes|no|null",
      "interpretation_grounded": "yes|no|null",
      "note": "one sentence"
    }}
  ],
  "overall_groundedness": 1,
  "hallucination_detected": true,
  "summary": "2-3 sentences for the paper"
}}
```"""


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_done_keys(out_path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        done.add((r.get("image_id"), r.get("provider"), r.get("scenario"), r.get("condition")))
    return done


def trace_for_judge(trace: list[dict]) -> str:
    """Human-readable trace for the judge (code + outputs only)."""
    lines: list[str] = []
    for i, ev in enumerate(trace or [], 1):
        kind = ev.get("kind", "?")
        payload = ev.get("payload", {})
        lines.append(f"--- event {i} ({kind}) ---")
        if isinstance(payload, dict):
            if payload.get("code"):
                lines.append("CODE:\n" + str(payload["code"])[:4000])
            if payload.get("output"):
                lines.append("OUTPUT:\n" + str(payload["output"])[:4000])
            for out in payload.get("outputs") or []:
                lines.append("OUTPUT:\n" + str(out)[:4000])
            if payload.get("content"):
                lines.append("CONTENT:\n" + str(payload["content"])[:4000])
            if payload.get("input"):
                lines.append("INPUT:\n" + json.dumps(payload["input"])[:1000])
        else:
            lines.append(str(payload)[:4000])
    return "\n".join(lines) if lines else "(empty trace — no code was executed)"


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(text: str) -> dict | None:
    m = _JSON_FENCE.search(text or "")
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = (text or "").rfind("{")
    end = (text or "").rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def call_judge(client, record: dict) -> dict:
    parsed = record.get("parsed_json") or {}
    evidence = parsed.get("evidence") or []
    model_id = record.get("model") or record.get("provider")
    user_msg = JUDGE_USER_TEMPLATE.format(
        model=model_id,
        scenario=record.get("scenario"),
        condition=record.get("condition"),
        image_id=record.get("image_id"),
        label=record.get("label"),
        classification=record.get("classification"),
        evidence_json=json.dumps(evidence, indent=2)[:12000],
        trace_text=trace_for_judge(record.get("tool_trace") or [])[:12000],
    )
    t0 = time.time()
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=2048,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    latency = time.time() - t0
    text = "\n".join(
        getattr(b, "text", "") or ""
        for b in resp.content
        if getattr(b, "type", "") == "text"
    )
    parsed_judge = extract_json(text)
    usage = getattr(resp, "usage", None)
    return {
        "judge_model": JUDGE_MODEL,
        "judge_raw": text,
        "judge_parsed": parsed_judge,
        "judge_latency_s": round(latency, 2),
        "judge_usage": {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        } if usage else {},
        "judge_error": None if parsed_judge else "failed_to_parse_judge_json",
    }


def sample_records(
    records: list[dict],
    n: int,
    providers: list[str] | None,
    condition: str | None,
    seed: int,
) -> list[dict]:
    pool = records
    if providers:
        prov_set = {p.lower() for p in providers}
        pool = [r for r in pool if r.get("provider", "").lower() in prov_set]
    if condition:
        pool = [r for r in pool if r.get("condition") == condition]
    pool = [r for r in pool if r.get("parsed_json") and not r.get("error")]

    if len(pool) <= n:
        return pool

    # Stratify by (provider, scenario) then sample evenly
    buckets: dict[tuple, list] = defaultdict(list)
    for r in pool:
        buckets[(r.get("provider"), r.get("scenario"))].append(r)
    rng = random.Random(seed)
    per_bucket = max(1, n // max(len(buckets), 1))
    chosen: list[dict] = []
    for items in buckets.values():
        rng.shuffle(items)
        chosen.extend(items[:per_bucket])
    rng.shuffle(chosen)
    return chosen[:n]


def summarize(results: list[dict]) -> dict:
    by_group: dict[tuple, list] = defaultdict(list)
    for r in results:
        jp = r.get("judge_parsed") or {}
        score = jp.get("overall_groundedness")
        if isinstance(score, (int, float)):
            label = r.get("model") or r.get("provider")
            by_group[(label, r.get("scenario"), r.get("condition"))].append(float(score))

    summary: dict = {"n_judged": len(results), "by_group": {}}
    all_scores = []
    for key, scores in sorted(by_group.items()):
        model, scen, cond = key
        summary["by_group"][f"{model}|{scen}|{cond}"] = {
            "n": len(scores),
            "groundedness_mean": round(float(mean(scores)), 3),
        }
        all_scores.extend(scores)
    if all_scores:
        summary["groundedness_mean"] = round(float(mean(all_scores)), 3)
    hall_rate = mean(
        1 if (r.get("judge_parsed") or {}).get("hallucination_detected") else 0
        for r in results
    )
    summary["hallucination_detected_rate"] = round(float(hall_rate), 3)
    return summary


def backfill_judge_models(out: Path, model_map: dict[str, str]) -> int:
    """Add pinned model ids to existing judge JSONL rows (no API calls)."""
    if not out.exists() or not model_map:
        return 0
    records = load_records(out)
    changed = 0
    for r in records:
        if not r.get("model"):
            model_id = model_map.get(r.get("provider", ""))
            if model_id:
                r["model"] = model_id
                changed += 1
    if changed:
        with out.open("w") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")
    return changed


def resummarize_judge(out: Path, summary_path: Path | None = None) -> dict:
    """Rebuild summary JSON from judge JSONL using model ids where present."""
    records = load_records(out) if out.exists() else []
    result = summarize(records)
    path = summary_path or out.with_suffix(".summary.json")
    path.write_text(json.dumps(result, indent=2))
    return result


def run_judge(
    inp: Path,
    out: Path,
    summary: Path | None = None,
    sample: int = 50,
    providers: list[str] | None = None,
    condition: str = "B",
    seed: int = 42,
    sleep_s: float = 0.5,
) -> dict:
    """Run Claude groundedness audit; resumable. Returns summary dict."""
    import anthropic
    client = anthropic.Anthropic()

    records = load_records(inp)
    done = load_done_keys(out)
    todo = sample_records(records, sample, providers, condition, seed)
    todo = [r for r in todo if (r.get("image_id"), r.get("provider"),
                                r.get("scenario"), r.get("condition")) not in done]

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Sampled {sample} → {len(todo)} to run ({len(done)} already done)")

    all_results = load_records(out) if out.exists() else []

    with out.open("a") as fout:
        for i, rec in enumerate(todo, 1):
            key = (rec.get("image_id"), rec.get("provider"), rec.get("scenario"), rec.get("condition"))
            print(f"[{i}/{len(todo)}] {key}")
            try:
                judge = call_judge(client, rec)
            except Exception as e:
                judge = {"judge_error": f"{type(e).__name__}: {e}", "judge_parsed": None}
            out_rec = {
                "image_id": rec.get("image_id"),
                "provider": rec.get("provider"),
                "model": rec.get("model"),
                "scenario": rec.get("scenario"),
                "condition": rec.get("condition"),
                "label": rec.get("label"),
                "classification": rec.get("classification"),
                **judge,
            }
            fout.write(json.dumps(out_rec, default=str) + "\n")
            fout.flush()
            all_results.append(out_rec)
            if sleep_s:
                time.sleep(sleep_s)

    result = summarize(all_results)
    summary_path = summary or out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {summary_path}")
    return result


def refresh_judge_labels(
    out: Path,
    summary_path: Path,
    model_map: dict[str, str],
) -> dict:
    """Backfill model ids + rewrite summary — use after model-label migration."""
    n = backfill_judge_models(out, model_map)
    if n:
        print(f"Backfilled model id on {n} judge records")
    return resummarize_judge(out, summary_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path,
                    help="Scored or raw eval JSONL")
    ap.add_argument("--out", required=True, type=Path,
                    help="Per-record judge output JSONL (resumable)")
    ap.add_argument("--summary", type=Path, default=None,
                    help="Aggregate summary JSON")
    ap.add_argument("--sample", type=int, default=50,
                    help="Number of records to judge (stratified sample)")
    ap.add_argument("--providers", nargs="*", default=None,
                    help="Limit to providers, e.g. openai gemini")
    ap.add_argument("--condition", default="B",
                    help="Condition to judge (default: B — agentic only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="Seconds between API calls (rate-limit cushion)")
    args = ap.parse_args()

    result = run_judge(
        args.inp, args.out, args.summary,
        sample=args.sample, providers=args.providers,
        condition=args.condition, seed=args.seed, sleep_s=args.sleep,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
