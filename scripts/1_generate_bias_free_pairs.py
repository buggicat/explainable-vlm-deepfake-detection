#!/usr/bin/env python3
"""Generate Scenario-2 bias-free fakes from real RVF10K faces.

  self_cond       SDXL base + img2img at controllable strength (default 0.5):
                  whole image is partially noised then denoised, preserving
                  identity/pose while injecting diffusion artifacts.
                  This is what B-Free calls 'self-conditioned reconstruction'.

  local_inpaint   SDXL inpainting with a central face-region rectangle mask:
                  only the masked region is regenerated, surrounding pixels
                  stay as the original real. Partial / 'augmented' fake.

Requires a GPU with sufficient VRAM (≥16 GB recommended for SDXL). Resumable: skips images whose output already exists.

Note: Stability AI's SD 2.x repos were pulled from HF; we use SDXL instead.
Pass --img2img-model / --inpaint-model to override either model.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable

import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting
from PIL import Image
from tqdm import tqdm


SD_IMG2IMG_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SD_INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_CAPTION = "a portrait photograph of a person, photorealistic, high detail"
TARGET_SIZE = 1024


def seeded_generator(name: str, device: str) -> torch.Generator:
    h = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:16], 16)
    g = torch.Generator(device=device)
    g.manual_seed(h % (2**63 - 1))
    return g


def make_local_mask(size: int, top=0.30, bottom=0.55, left=0.15, right=0.85) -> Image.Image:
    img = Image.new("L", (size, size), color=0)
    pixels = img.load()
    y0, y1 = int(size * top), int(size * bottom)
    x0, x1 = int(size * left), int(size * right)
    for y in range(y0, y1):
        for x in range(x0, x1):
            pixels[x, y] = 255
    return img


def iter_real_images(rvf10k_dir: Path, limit: int | None) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in rvf10k_dir.iterdir() if p.suffix.lower() in exts)
    if limit:
        paths = paths[:limit]
    yield from paths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rvf10k-dir", required=True, type=Path,
                    help="Directory of RVF10K real faces (e.g. .../rvf10k/valid/real)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output root (will contain real_{size}/, self_cond/, local_inpaint/)")
    ap.add_argument("--mode", choices=["self_cond", "local_inpaint", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N images")
    ap.add_argument("--caption", default=DEFAULT_CAPTION)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--self-cond-strength", type=float, default=0.1,
                    help="img2img strength for self_cond (0.0=identical, 1.0=fully imagined; "
                         "0.1 keeps identity while injecting subtle diffusion artifacts)")
    ap.add_argument("--img2img-model", default=SD_IMG2IMG_MODEL,
                    help="HF model id for self_cond (img2img). Default: SDXL base.")
    ap.add_argument("--inpaint-model", default=SD_INPAINT_MODEL,
                    help="HF model id for local_inpaint. Default: SDXL inpainting.")
    ap.add_argument("--target-size", type=int, default=TARGET_SIZE,
                    help="Square pixel size for generation (default 1024 for SDXL; "
                         "use 512 to roughly 4x speed at the cost of quality).")
    args = ap.parse_args()

    out_real = args.out_dir / f"real_{args.target_size}"
    out_self = args.out_dir / "self_cond"
    out_local = args.out_dir / "local_inpaint"
    for d in (out_real, out_self, out_local):
        d.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    do_self = args.mode in ("self_cond", "both")
    do_local = args.mode in ("local_inpaint", "both")

    img2img_pipe = None
    inpaint_pipe = None
    if do_self:
        print(f"Loading img2img: {args.img2img_model}...", flush=True)
        img2img_pipe = AutoPipelineForImage2Image.from_pretrained(args.img2img_model, torch_dtype=dtype)
        img2img_pipe = img2img_pipe.to(args.device)
        img2img_pipe.set_progress_bar_config(disable=True)
        if args.device == "cuda":
            img2img_pipe.enable_attention_slicing()
    if do_local:
        print(f"Loading inpaint: {args.inpaint_model}...", flush=True)
        inpaint_pipe = AutoPipelineForInpainting.from_pretrained(args.inpaint_model, torch_dtype=dtype)
        inpaint_pipe = inpaint_pipe.to(args.device)
        inpaint_pipe.set_progress_bar_config(disable=True)
        if args.device == "cuda":
            inpaint_pipe.enable_attention_slicing()

    local_mask = make_local_mask(args.target_size) if do_local else None

    paths = list(iter_real_images(args.rvf10k_dir, args.limit))
    print(f"Found {len(paths)} real images.", flush=True)

    for src in tqdm(paths, desc="generate"):
        try:
            init = Image.open(src).convert("RGB").resize((args.target_size, args.target_size), Image.LANCZOS)
        except Exception as e:
            print(f"[skip] {src.name}: {e}", file=sys.stderr)
            continue

        real_out = out_real / f"{src.stem}.png"
        if not real_out.exists():
            init.save(real_out)

        if do_self:
            dst = out_self / f"{src.stem}.png"
            if not dst.exists():
                gen = seeded_generator(f"self_cond:{src.name}", args.device)
                with torch.inference_mode():
                    result = img2img_pipe(
                        prompt=args.caption,
                        image=init,
                        strength=args.self_cond_strength,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        generator=gen,
                    ).images[0]
                result.save(dst)

        if do_local:
            dst = out_local / f"{src.stem}.png"
            if not dst.exists():
                gen = seeded_generator(f"local_inpaint:{src.name}", args.device)
                with torch.inference_mode():
                    result = inpaint_pipe(
                        prompt=args.caption,
                        image=init,
                        mask_image=local_mask,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        generator=gen,
                    ).images[0]
                result.save(dst)


if __name__ == "__main__":
    main()
