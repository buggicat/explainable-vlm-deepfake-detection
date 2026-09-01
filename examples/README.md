# Example Images

A small set of representative images illustrating the dataset and both manipulation types
used in this study. Full datasets are linked in the main [README](../README.md#datasets);
this folder is only for quick visual inspection.

## `scenario1_gan_fake/` — Standard benchmark (RVF10K)

Real and GAN-generated fake faces from RVF10K. Real and fake images are **unrelated by
construction** in Scenario 1 (different underlying identities) — this is the standard,
potentially biased benchmark the paper contrasts against Scenario 2.

| File | Description |
|---|---|
| `real_00092.jpg`, `real_00202.jpg` | Real faces (RVF10K `valid/real/`) |
| `fake_022Y921TKX.jpg`, `fake_02P5FJ8I1D.jpg` | GAN-generated fake faces (RVF10K `valid/fake/`) |

## `scenario2_bias_free/` — Content-matched, bias-free pairs (our contribution)

Three comparison panels, each showing the **same real face** side by side with both
SDXL-based manipulation types generated from it (Real | Self-Conditioned Fake |
Local-Inpaint Fake), labeled directly on the image. Because the fake images are
content-matched to their real counterpart (same underlying identity/pose/lighting), any
detector relying on dataset-level shortcuts rather than manipulation-specific evidence
should perform worse here than on Scenario 1 — this is the paper's central bias-gap
measurement.

| File | Description |
|---|---|
| `00328_comparison.jpg` | Real / self-cond / local-inpaint triplet, image ID 00328 |
| `00407_comparison.jpg` | Real / self-cond / local-inpaint triplet, image ID 00407 |
| `00584_comparison.jpg` | Real / self-cond / local-inpaint triplet, image ID 00584 |

Self-conditioned reconstruction: SDXL img2img, denoising strength 0.1 (whole-frame,
identity-preserving). Local inpainting: SDXL inpainting on a fixed central face-region mask
(partial manipulation, surrounding pixels untouched).
