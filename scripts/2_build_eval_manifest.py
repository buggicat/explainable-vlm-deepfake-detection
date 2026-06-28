#!/usr/bin/env python3
"""Build the unified evaluation manifest.

Produces a JSON file with three scenarios:
  scenario1                RVF10K valid/ as-is (1500 real GAN + 1500 fake)
  scenario2_self_cond      real_512 + self_cond fakes (content-matched)
  scenario2_local_inpaint  real_512 + local_inpaint fakes (content-matched)

Paths are stored relative to --project-root for portability.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def list_images(d: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in d.iterdir() if p.suffix.lower() in exts)


def rel(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p.resolve())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rvf10k-dir", required=True, type=Path,
                    help="RVF10K valid dir containing real/ and fake/ subdirs")
    ap.add_argument("--scenario2-dir", required=True, type=Path,
                    help="Scenario-2 dir from generate_fakes.py (real_512/, self_cond/, local_inpaint/)")
    ap.add_argument("--project-root", type=Path, default=Path.cwd(),
                    help="Root for relative paths (default: cwd)")
    ap.add_argument("--reals-subdir", default="auto",
                    help="Subdir of --scenario2-dir holding the resized reals "
                         "(default: 'auto' picks the largest real_* directory)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    s1_real = args.rvf10k_dir / "real"
    s1_fake = args.rvf10k_dir / "fake"
    if args.reals_subdir == "auto":
        cands = sorted(args.scenario2_dir.glob("real_*"))
        s2_real = cands[-1] if cands else args.scenario2_dir / "real_1024"
    else:
        s2_real = args.scenario2_dir / args.reals_subdir
    s2_self = args.scenario2_dir / "self_cond"
    s2_local = args.scenario2_dir / "local_inpaint"

    scenario1 = []
    for p in list_images(s1_real):
        scenario1.append({"image_id": f"s1_real_{p.stem}", "path": rel(p, args.project_root), "label": "real"})
    for p in list_images(s1_fake):
        scenario1.append({"image_id": f"s1_fake_{p.stem}", "path": rel(p, args.project_root), "label": "fake"})

    scenario2_self = []
    if s2_real.exists() and s2_self.exists():
        reals_by_stem = {p.stem: p for p in list_images(s2_real)}
        for stem, real_p in reals_by_stem.items():
            scenario2_self.append({"image_id": f"s2sc_real_{stem}", "path": rel(real_p, args.project_root), "label": "real"})
            fake_p = s2_self / f"{stem}.png"
            if fake_p.exists():
                scenario2_self.append({
                    "image_id": f"s2sc_fake_{stem}", "path": rel(fake_p, args.project_root),
                    "label": "fake", "source_real": stem,
                })

    scenario2_local = []
    if s2_real.exists() and s2_local.exists():
        reals_by_stem = {p.stem: p for p in list_images(s2_real)}
        for stem, real_p in reals_by_stem.items():
            scenario2_local.append({"image_id": f"s2li_real_{stem}", "path": rel(real_p, args.project_root), "label": "real"})
            fake_p = s2_local / f"{stem}.png"
            if fake_p.exists():
                scenario2_local.append({
                    "image_id": f"s2li_fake_{stem}", "path": rel(fake_p, args.project_root),
                    "label": "fake", "source_real": stem,
                })

    manifest = {
        "meta": {
            "created": dt.datetime.now(dt.timezone.utc).isoformat(),
            "rvf10k_dir": str(args.rvf10k_dir.resolve()),
            "scenario2_dir": str(args.scenario2_dir.resolve()),
            "project_root": str(args.project_root.resolve()),
        },
        "scenarios": {
            "scenario1": scenario1,
            "scenario2_self_cond": scenario2_self,
            "scenario2_local_inpaint": scenario2_local,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {args.out}")
    print(f"  scenario1: {len(scenario1)} images")
    print(f"  scenario2_self_cond: {len(scenario2_self)} images")
    print(f"  scenario2_local_inpaint: {len(scenario2_local)} images")


if __name__ == "__main__":
    main()
