#!/usr/bin/env python3
"""Orchestrator: runs the (provider x scenario x condition x image) matrix.

Resumable: each output JSONL line is one (image_id, provider, scenario, condition)
record. Re-running with the same --out skips records already present — safe to
stop and restart at any time without wasting API credits.

Scalable: --limit controls images per scenario (all providers). Use
--provider-limits to set different per-provider caps, e.g.:
    --provider-limits openai=200 anthropic=500 gemini=500
To expand a run, simply increase the limits and re-run — only new records
are added (already-done records are skipped via the resume mechanism).

Parallel: by default, all three providers run simultaneously (one thread each).
Use --parallel=1 to disable parallelism (sequential, easier to debug).
Each provider thread processes its own images sequentially so per-provider
rate limits are respected automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_api_client import make_client, VlmCall


PROMPT_FILES = {
    "A": "promptv6_baseline.txt",
    "B": "promptv6_agentic.txt",
}


def load_prompts(prompts_dir: Path) -> dict[str, str]:
    return {cond: (prompts_dir / fname).read_text()
            for cond, fname in PROMPT_FILES.items()}


def compact_jsonl(out_path: Path) -> int:
    """Rewrite the JSONL keeping only the latest successful record per
    (image_id, provider, scenario, condition) key. Error records are always
    removed — they will be retried on the next run.
    Returns the number of records removed."""
    if not out_path.exists():
        return 0
    seen: dict[tuple, str] = {}  # key -> raw line (last wins)
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (r.get("image_id"), r.get("provider"),
               r.get("scenario"), r.get("condition"))
        seen[key] = line
    original_count = sum(1 for l in out_path.read_text().splitlines() if l.strip())
    # Drop any record (including the deduplicated "last") that has an error
    clean = {k: v for k, v in seen.items() if not json.loads(v).get("error")}
    out_path.write_text("\n".join(clean.values()) + ("\n" if clean else ""))
    return original_count - len(clean)


def load_done_keys(out_path: Path) -> set[tuple]:
    done = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("error"):
            continue  # errored records are retried on the next run
        done.add((r.get("image_id"), r.get("provider"),
                  r.get("scenario"), r.get("condition")))
    return done


def parse_provider_limits(raw: list[str]) -> dict[str, int]:
    """Parse ['openai=200', 'anthropic=500'] into {'openai': 200, ...}."""
    result = {}
    for entry in raw:
        if "=" not in entry:
            raise ValueError(f"Invalid --provider-limits entry '{entry}'. "
                             f"Use format provider=N, e.g. openai=200")
        provider, n = entry.split("=", 1)
        result[provider.strip().lower()] = int(n.strip())
    return result


def run_provider(
    provider: str,
    work: list[tuple],
    clients: dict,
    prompts: dict[str, str],
    project_root: Path,
    reference: str | None,
    file_handle,
    file_lock: threading.Lock,
    pbar: tqdm,
    concurrency: int = 1,
) -> dict:
    """Process all work items for one provider.

    concurrency > 1 fires that many API calls simultaneously, cutting
    wall time proportionally (safe as long as it stays within rate limits).
    Returns a summary dict with counts for this provider.
    """
    client = clients[provider]
    counters = {"n_ok": 0, "n_err": 0, "n_tool_used": 0}
    counter_lock = threading.Lock()

    def process_one(work_item: tuple) -> None:
        scen, cond, prov, item = work_item
        image_path = project_root / item["path"]
        t0 = time.time()
        err = None
        call: VlmCall | None = None
        try:
            call = client.call(
                image_path, prompts[cond],
                with_tools=(cond == "B"),
                reference_text=reference,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        record = {
            "image_id":       item["image_id"],
            "label":          item["label"],
            "scenario":       scen,
            "condition":      cond,
            "provider":       prov,
            "image_path":     item["path"],
            "wall_s":         round(time.time() - t0, 3),
            "reference_used": reference is not None,
        }
        if call is not None:
            record.update(call.to_dict())
        if err:
            record["error"] = err

        with file_lock:
            file_handle.write(json.dumps(record, default=str) + "\n")
            file_handle.flush()

        pbar.update(1)

        with counter_lock:
            if call is not None:
                counters["n_ok"] += 1
                if cond == "B" and call.tool_trace:
                    counters["n_tool_used"] += 1
            if err:
                counters["n_err"] += 1

    if concurrency <= 1:
        for item in work:
            process_one(item)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as inner_pool:
            list(inner_pool.map(process_one, work))

    return {"provider": provider, **counters}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--prompts-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "prompts",
                    help="Directory containing promptv5_*.txt")
    ap.add_argument("--project-root", type=Path, default=Path.cwd(),
                    help="Root for resolving relative paths in the manifest")
    ap.add_argument("--models", nargs="+",
                    default=["openai", "anthropic", "gemini"],
                    choices=["openai", "anthropic", "gemini"])
    ap.add_argument("--scenarios", nargs="+",
                    default=["scenario1", "scenario2_self_cond",
                             "scenario2_local_inpaint"])
    ap.add_argument("--conditions", nargs="+", default=["A", "B"],
                    choices=["A", "B"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Max images per scenario per provider (applies to all "
                         "providers unless overridden by --provider-limits)")
    ap.add_argument("--provider-limits", nargs="+", default=[],
                    metavar="PROVIDER=N",
                    help="Per-provider image limits, e.g. openai=200 anthropic=500. "
                         "Overrides --limit for the named provider. To scale up "
                         "later, increase N and re-run — already-done records are "
                         "skipped automatically.")
    ap.add_argument("--parallel", type=int, default=3,
                    help="Number of provider threads to run simultaneously "
                         "(default 3 = all providers in parallel; use 1 to debug "
                         "sequentially)")
    ap.add_argument("--provider-concurrency", type=int, default=1,
                    metavar="N",
                    help="Concurrent API calls per provider thread (default 1 = "
                         "sequential). Set to 4–8 to cut wall time proportionally. "
                         "Stay within your API rate limits.")
    ap.add_argument("--reference-file", type=Path,
                    default=Path(__file__).resolve().parents[1] / "prompts"
                            / "techniques_reference.md",
                    help="Markdown reference card sent alongside each call.")
    ap.add_argument("--no-reference", action="store_true",
                    help="Disable the reference card (ablation run)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output JSONL (append-only, resumable)")
    args = ap.parse_args()

    prov_limits = parse_provider_limits(args.provider_limits)

    def get_limit(provider: str) -> int | None:
        return prov_limits.get(provider, args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    removed = compact_jsonl(args.out)
    if removed:
        print(f"Compacted {args.out.name}: removed {removed} duplicate/error record(s).",
              flush=True)

    manifest = json.loads(args.manifest.read_text())
    prompts  = load_prompts(args.prompts_dir)
    done     = load_done_keys(args.out)

    reference = None
    if not args.no_reference and args.reference_file.exists():
        reference = args.reference_file.read_text()
        print(f"Reference loaded: {args.reference_file.name} "
              f"({len(reference)} chars)", flush=True)
    else:
        print("No reference card in use.", flush=True)

    clients = {p: make_client(p) for p in args.models}

    print("Effective limits per provider:", flush=True)
    for prov in args.models:
        lim = get_limit(prov)
        print(f"  {prov}: {lim if lim is not None else 'all'} images/scenario",
              flush=True)

    # Build per-provider work lists
    work_by_provider: dict[str, list] = {p: [] for p in args.models}
    for prov in args.models:
        lim = get_limit(prov)
        for scen in args.scenarios:
            items = manifest["scenarios"].get(scen, [])
            if lim is not None:
                items = items[:lim]
            for item in items:
                for cond in args.conditions:
                    key = (item["image_id"], prov, scen, cond)
                    if key in done:
                        continue
                    work_by_provider[prov].append((scen, cond, prov, item))

    total_pending = sum(len(w) for w in work_by_provider.values())
    print(f"Pending: {total_pending} calls (already done: {len(done)})",
          flush=True)
    if total_pending == 0:
        print("Nothing to do — all records already present in output.",
              flush=True)
        return

    for prov, w in work_by_provider.items():
        print(f"  {prov}: {len(w)} calls", flush=True)

    n_threads = min(args.parallel, len(args.models))
    print(f"Running {n_threads} provider thread(s) in parallel, "
          f"{args.provider_concurrency} concurrent call(s) each.", flush=True)

    file_lock = threading.Lock()
    pbar = tqdm(total=total_pending, desc="eval", unit="call")

    with args.out.open("a") as f:
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = {
                pool.submit(
                    run_provider,
                    prov, work_by_provider[prov], clients, prompts,
                    args.project_root, reference, f, file_lock, pbar,
                    args.provider_concurrency,
                ): prov
                for prov in args.models
                if work_by_provider[prov]
            }
            for future in as_completed(futures):
                summary = future.result()
                prov = summary["provider"]
                cond_b_total = sum(
                    1 for _, cond, _, _ in work_by_provider[prov]
                    if cond == "B"
                )
                tool_pct = (
                    f"{summary['n_tool_used']}/{cond_b_total} "
                    f"({100*summary['n_tool_used']/cond_b_total:.0f}%)"
                    if cond_b_total else "n/a"
                )
                print(
                    f"\n{prov}: done — "
                    f"{summary['n_ok']} OK, {summary['n_err']} errors, "
                    f"tool use in cond-B: {tool_pct}",
                    flush=True,
                )

    pbar.close()


if __name__ == "__main__":
    main()
