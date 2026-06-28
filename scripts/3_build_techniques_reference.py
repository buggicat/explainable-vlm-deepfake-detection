#!/usr/bin/env python3
"""Build prompts/techniques_reference.md by distilling chunks.jsonl
against the 54-technique taxonomy.

One-time offline pass. Output is a Markdown reference shipped alongside the
prompt + image on every API call in run_eval.py (multi-part user input).

The reference is curated prompt content, not retrieval: the same document is
sent on every call. The model still decides what to compute and which tools
to use.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


METHODOLOGY_PRIORS = """## Key methodological priors

- **Bias-free evaluation** (B-Free, CVPR 2025): real and fake images must be
  content-matched to isolate generation artifacts from dataset bias. High
  accuracy on standard benchmarks often reflects content/format/resolution
  shortcuts rather than true generation cues.
- **Frequency-domain robustness** (Seeing What Matters, 2025): mid-high
  *diagonal* spatial frequencies are the most robust forensic cues — they
  survive JPEG and codec compression. Horizontal/vertical bands and very-high
  frequencies are contaminated by compression artifacts and are weaker
  discriminators.
- **Self-conditioned reconstruction signal**: passing a real image through a
  diffusion model's inpainting pipeline (empty mask, generic prompt) injects
  the architecture's fingerprint while preserving content; the residual
  between real and self-conditioned versions reveals subtle low-frequency
  artifacts.
"""


def parse_taxonomy(path: Path) -> list[tuple[str, str, str]]:
    """Returns [(number, name, category), ...] for each numbered technique."""
    category = ""
    out: list[tuple[str, str, str]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\.\s+(.+)$", line)
        if m:
            out.append((m.group(1), m.group(2), category))
        else:
            category = line
    return out


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def best_sentence(text: str, keywords: list[str], max_chars: int = 280) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if 20 < len(s.strip()) < 600]
    if not sents:
        return text[:max_chars]
    def score(s: str) -> int:
        sl = s.lower()
        return sum(sl.count(k.lower()) for k in keywords)
    sents.sort(key=score, reverse=True)
    best = sents[0]
    if len(best) > max_chars:
        best = best[:max_chars].rsplit(" ", 1)[0] + "…"
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taxonomy", required=True, type=Path,
                    help="Path to multimedia_forensics_techniques.txt")
    ap.add_argument("--chunks", required=True, type=Path,
                    help="Path to chunks.jsonl")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output Markdown reference (e.g. prompts/techniques_reference.md)")
    ap.add_argument("--top-k", type=int, default=4,
                    help="Top-k chunks to consider per technique (then deduped by document)")
    ap.add_argument("--snippets-per-technique", type=int, default=2,
                    help="Max snippets to emit per technique")
    args = ap.parse_args()

    techniques = parse_taxonomy(args.taxonomy)
    chunks = load_chunks(args.chunks)
    print(f"Loaded {len(techniques)} techniques and {len(chunks)} chunks")

    chunk_texts = [c.get("text", "") for c in chunks]
    vec = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
    M = vec.fit_transform(chunk_texts)

    queries = [f"{name} {category}" for _, name, category in techniques]
    Q = vec.transform(queries)
    sims = cosine_similarity(Q, M)

    md: list[str] = [
        "# Forensic Techniques Reference (auto-generated)\n\n",
        "READ THIS BEFORE ANALYZING. The following is a curated reference of standard\n",
        "image-forensics techniques, distilled from a literature corpus of 270+ papers\n",
        "(`data/chunks.jsonl`). Use it to (a) recall the\n",
        "canonical name of each technique, (b) map your observations to that taxonomy,\n",
        "and (c) recall typical parameters / thresholds. You are not required to use\n",
        "every technique — choose what fits the image.\n\n",
        METHODOLOGY_PRIORS,
        "\n## Techniques (54 total)\n\n",
    ]

    current_cat = ""
    for i, (num, name, category) in enumerate(techniques):
        if category != current_cat:
            md.append(f"### {category}\n\n")
            current_cat = category

        keywords = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]+", name) if len(w) > 2]
        top_idxs = np.argsort(sims[i])[::-1][: args.top_k]

        snippets: list[tuple[str, str, str]] = []
        seen_docs: set[str] = set()
        for idx in top_idxs:
            c = chunks[idx]
            doc_id = c.get("doc_id", "")
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            text = c.get("text", "")
            if not text:
                continue
            snippet = best_sentence(text, keywords)
            snippets.append((snippet, c.get("filename", "?"), str(c.get("page_start", "?"))))
            if len(snippets) >= args.snippets_per_technique:
                break

        md.append(f"**{num}. {name}**\n")
        if snippets:
            for snip, fn, pg in snippets:
                md.append(f"> {snip} *(source: {fn}, p. {pg})*\n")
        else:
            md.append("> (no matching chunk found)\n")
        md.append("\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(md))
    size_kb = args.out.stat().st_size / 1024
    print(f"Wrote {args.out} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
