#!/usr/bin/env python3
"""Scenario-1 baseline for contrastive sensitivity.

Scenario 2's contrastive sensitivity = 1 - cosine_similarity(real_explanation,
fake_explanation) for CONTENT-MATCHED pairs (same underlying face). Scenario 1
has no such pairing (real and fake come from unrelated sources by
construction), so there is no single "correct" baseline pair. Instead we
construct an UNRELATED-PAIR baseline: randomly pair each real-image
explanation with a fake-image explanation from the same model/condition, and
compute the same 1 - cosine divergence over those random pairs. This
estimates the model's "typical" explanation dissimilarity when there is no
manipulation-specific signal to find, i.e. the noise floor that Scenario 2's
0.15-0.20 should be compared against.

To avoid depending on one arbitrary random pairing, we repeat the random
pairing N_PERMUTATIONS times (pooling all resulting divergence values before
bootstrapping), rather than trusting a single shuffle.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import _get_sbert  # reuse the same embedding model as the paper's Sens metric

RESULTS = Path(__file__).resolve().parents[1] / "results" / "full_run.scored.jsonl"
OUT = Path(__file__).resolve().parents[1] / "results" / "s1_baseline_ci.json"

N_BOOT = 2000
SEED = 20260819
ALPHA = 0.05
N_PERMUTATIONS = 5  # independent random pairings, pooled

MODELS = ["openai", "gemini", "anthropic"]


def load_s1_texts():
    """Return {(provider, condition): {"real": [...], "fake": [...]}} of raw_text."""
    out = {}
    with RESULTS.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("scenario") != "scenario1":
                continue
            key = (r["provider"], r["condition"])
            out.setdefault(key, {"real": [], "fake": []})
            text = r.get("raw_text") or ""
            if text:
                out[key][r["label"]].append(text)
    return out


def percentile_ci(vals, alpha=ALPHA):
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def main():
    rng = np.random.default_rng(SEED)
    model_st = _get_sbert()
    if model_st is None:
        raise SystemExit("sentence-transformers not available — install it first "
                          "(same requirement as the Sens-SC/Sens-LI computation).")

    texts_by_key = load_s1_texts()

    out = {"n_boot": N_BOOT, "n_permutations": N_PERMUTATIONS, "seed": SEED, "baseline_ci": {}}

    for model in MODELS:
        real_all, fake_all = [], []
        for cond in ["A", "B"]:
            key = (model, cond)
            if key not in texts_by_key:
                continue
            real_all.extend(texts_by_key[key]["real"])
            fake_all.extend(texts_by_key[key]["fake"])

        if len(real_all) < 10 or len(fake_all) < 10:
            continue

        # Embed once, reuse across all random pairings (expensive step done once).
        real_embs = model_st.encode(real_all, normalize_embeddings=True, batch_size=64,
                                     show_progress_bar=False)
        fake_embs = model_st.encode(fake_all, normalize_embeddings=True, batch_size=64,
                                     show_progress_bar=False)

        n_pairs = min(len(real_embs), len(fake_embs))
        divergences = []
        for _ in range(N_PERMUTATIONS):
            real_idx = rng.permutation(len(real_embs))[:n_pairs]
            fake_idx = rng.permutation(len(fake_embs))[:n_pairs]
            sims = np.einsum("ij,ij->i", real_embs[real_idx], fake_embs[fake_idx])
            divergences.extend((1.0 - sims).tolist())

        arr = np.array(divergences)
        n = len(arr)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        means = arr[idx].mean(axis=1)
        lo, hi = percentile_ci(means)
        out["baseline_ci"][model] = {
            "point": float(arr.mean()), "ci_lo": lo, "ci_hi": hi,
            "n_random_pairs": n, "n_real": len(real_all), "n_fake": len(fake_all),
        }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}\n")
    print("--- Scenario-1 unrelated-pair baseline (95% CI) ---")
    for model, v in sorted(out["baseline_ci"].items()):
        print(f"  {model:12s} baseline={v['point']:.3f}  CI=[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]  "
              f"(n_random_pairs={v['n_random_pairs']})")


if __name__ == "__main__":
    main()
