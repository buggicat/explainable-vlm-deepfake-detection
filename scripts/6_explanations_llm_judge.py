#!/usr/bin/env python3
"""LLM-as-judge groundedness audit for agentic forensic explanations.

Uses the Anthropic API (Claude) as an independent judge. For each evaluated
record, the judge receives ONLY:
  - the structured evidence JSON from the model under test
  - the tool execution trace (code outputs, not the model's prose)

Neither the ground-truth label nor the model's own verdict/classification is
included in the prompt, so the judge scores groundedness purely from the
evidence and the trace. Returns per-evidence tool/numeric/interpretation-
grounded flags plus an overall 1-5 groundedness rating and a hallucination
flag.

Judges a fixed, deterministic stratified sample of the agentic-condition
records (condition B, GPT + Gemini only — Claude is the judge and is not
itself judged): same (provider, scenario) stratification and seed as the
rest of the evaluation pipeline, so re-running this script reproduces the
same sample every time.

Writes results/llm_judge_final.jsonl and results/llm_judge_final.summary.json.

Usage:
  export ANTHROPIC_API_KEY=...
  python3 scripts/6_explanations_llm_judge.py
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

JUDGE_MODEL = "claude-sonnet-4-6"

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

JUDGE_USER_TEMPLATE_FINAL = """Audit this forensic analysis record.

MODEL UNDER TEST: {model} | scenario: {scenario} | condition: {condition}
IMAGE ID: {image_id}

=== EVIDENCE (model's structured claims) ===
{evidence_json}

=== TOOL EXECUTION TRACE (ground truth for computational claims) ===
{trace_text}

Score each evidence item, then give an overall groundedness score 1-5:
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

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FULL_RUN = RESULTS_DIR / "full_run.scored.jsonl"
OUT = RESULTS_DIR / "llm_judge_final.jsonl"
OUT_SUMMARY = RESULTS_DIR / "llm_judge_final.summary.json"

# Deterministic sample: 2 providers x 3 scenarios = 6 strata, 33 per stratum
# = 198 records (99 GPT + 99 Gemini), condition B only.
SAMPLE_N = 198
SAMPLE_SEED = 42
SAMPLE_CONDITION = "B"
SAMPLE_PROVIDERS = ["openai", "gemini"]


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


def sample_records(
    records: list[dict],
    n: int,
    providers: list[str] | None,
    condition: str | None,
    seed: int,
) -> list[dict]:
    """Deterministic stratified sample by (provider, scenario)."""
    pool = records
    if providers:
        prov_set = {p.lower() for p in providers}
        pool = [r for r in pool if r.get("provider", "").lower() in prov_set]
    if condition:
        pool = [r for r in pool if r.get("condition") == condition]
    pool = [r for r in pool if r.get("parsed_json") and not r.get("error")]

    if len(pool) <= n:
        return pool

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


def call_judge_final(client, record: dict) -> dict:
    parsed = record.get("parsed_json") or {}
    evidence = parsed.get("evidence") or []
    model_id = record.get("model") or record.get("provider")
    user_msg = JUDGE_USER_TEMPLATE_FINAL.format(
        model=model_id,
        scenario=record.get("scenario"),
        condition=record.get("condition"),
        image_id=record.get("image_id"),
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


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_done_keys(out_path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if not out_path.exists():
        return done
    for r in load_jsonl(out_path):
        done.add((r.get("image_id"), r.get("provider"), r.get("scenario"), r.get("condition")))
    return done


def main():
    import anthropic
    client = anthropic.Anthropic()

    full_run_records = load_jsonl(FULL_RUN)
    sampled = sample_records(full_run_records, SAMPLE_N, SAMPLE_PROVIDERS,
                              SAMPLE_CONDITION, SAMPLE_SEED)
    target_keys = {(r["image_id"], r["provider"], r["scenario"], r["condition"])
                   for r in sampled}
    print(f"Judging {len(target_keys)} records "
          f"(stratified sample, seed={SAMPLE_SEED}, evidence + trace only).")

    full_run_by_key = {
        (r.get("image_id"), r.get("provider"), r.get("scenario"), r.get("condition")): r
        for r in full_run_records
    }

    done = load_done_keys(OUT)
    todo = [full_run_by_key[k] for k in target_keys
            if k in full_run_by_key and k not in done]
    print(f"{len(done)} already done (resuming), {len(todo)} remaining.")

    with OUT.open("a") as fout:
        for i, rec in enumerate(todo, 1):
            try:
                result = call_judge_final(client, rec)
            except Exception as e:
                result = {"judge_error": str(e)}
            out_rec = {
                "image_id": rec.get("image_id"),
                "provider": rec.get("provider"),
                "scenario": rec.get("scenario"),
                "condition": rec.get("condition"),
                "label": rec.get("label"),              # kept for OUR analysis only,
                "classification": rec.get("classification"),  # never sent to the judge
                **result,
            }
            fout.write(json.dumps(out_rec, default=str) + "\n")
            fout.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(todo)} done")

    results = load_jsonl(OUT)
    parsed_results = [{
        "model": r.get("provider"), "scenario": r.get("scenario"),
        "condition": r.get("condition"), "judge_parsed": r.get("judge_parsed"),
    } for r in results]
    summary = summarize(parsed_results)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
