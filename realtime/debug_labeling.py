"""
Diagnose "Found 0 object(s)" by running the labeler's steps one at a time on
saved frames and printing how many detections survive each stage.

    1. In the viewer, scan something and click Save State (writes saved_frames.npz)
    2. cd realtime && python debug_labeling.py "table" [num_frames]

Unlike object_labeler.py, nothing here swallows exceptions, and SAM3 runs at a
very low threshold so you can see scores that the real 0.3 threshold hides.
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

PROMPT = sys.argv[1] if len(sys.argv) > 1 else "table"
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 6
REAL_THRESHOLD = 0.3   # what the viewer uses by default
PROBE_THRESHOLD = 0.05  # low, so we see near-misses too
MIN_POINTS = 20         # same cutoff as object_labeler._detect_in_frame

here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "saved_frames.npz")
if not os.path.exists(path):
    sys.exit("No saved_frames.npz. Scan something, click Save State in the viewer, then rerun.")

d = np.load(path)
images, pmaps, cmasks = d["images"], d["point_maps"], d["conf_masks"]
print(f"saved frames: {len(images)}  image {images.shape[1:]} {images.dtype}  "
      f"point_map {pmaps.shape[1:]}  conf_mask {cmasks.shape[1:]}")

print(f"CUDA: {torch.cuda.is_available()}, free VRAM: "
      f"{torch.cuda.mem_get_info()[0] / 1e9:.1f} GB" if torch.cuda.is_available() else "CUDA: NO")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

t0 = time.time()
model = build_sam3_image_model()
proc = Sam3Processor(model)
print(f"SAM3 loaded in {time.time() - t0:.1f}s\n")

idxs = np.linspace(0, len(images) - 1, min(N_FRAMES, len(images)), dtype=int)
totals = {"sam_any": 0, "above_0.3": 0, "enough_points": 0}
best = None

for i in idxs:
    img, pm, cm = images[i], pmaps[i], cmasks[i]
    with torch.autocast("cuda", dtype=torch.bfloat16):  # SAM3 forces FlashAttention, needs bf16
        state = proc.set_image(Image.fromarray(img))
        proc.set_confidence_threshold(PROBE_THRESHOLD, state)
        out = proc.set_text_prompt(state=state, prompt=PROMPT)
    masks, scores = out["masks"], out["scores"]

    print(f"frame {i}: conf_mask keeps {cm.mean() * 100:.0f}% of pixels, "
          f"SAM3 returned {len(masks)} candidate(s) above {PROBE_THRESHOLD}")
    for k in range(len(masks)):
        s = float(scores[k].float())
        m = masks[k, 0].cpu().numpy()
        if m.shape != cm.shape:
            m = cv2.resize(m.astype(np.uint8), (cm.shape[1], cm.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        comb = m & cm
        pts = pm[comb]
        n_finite = int(np.isfinite(pts).all(axis=1).sum()) if len(pts) else 0
        passes_thr = s > REAL_THRESHOLD
        passes_pts = n_finite >= MIN_POINTS
        totals["sam_any"] += 1
        totals["above_0.3"] += passes_thr
        totals["enough_points"] += passes_thr and passes_pts
        verdict = "KEPT" if passes_thr and passes_pts else (
            "dropped: score below 0.3" if not passes_thr else "dropped: too few confident 3D points")
        print(f"   cand {k}: score {s:.3f}  mask {int(m.sum())} px  "
              f"after conf_mask {int(comb.sum())} px  finite 3D {n_finite}  -> {verdict}")
        if best is None or s > best[0]:
            best = (s, i, m)

print(f"\nSUMMARY for '{PROMPT}': {totals['sam_any']} candidates, "
      f"{totals['above_0.3']} above 0.3, {totals['enough_points']} would be kept")

if best is not None:
    s, i, m = best
    over = images[i].copy()
    over[m] = (over[m] * 0.5 + np.array([0, 255, 100]) * 0.5).astype(np.uint8)
    out_path = os.path.join(here, "debug_best_mask.png")
    cv2.imwrite(out_path, cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
    print(f"best mask (score {s:.3f}, frame {i}) written to {out_path}")

# also dump one raw frame so you can see what SAM3 is actually looking at
cv2.imwrite(os.path.join(here, "debug_frame.png"),
            cv2.cvtColor(images[idxs[len(idxs) // 2]], cv2.COLOR_RGB2BGR))
print(f"sample input frame written to {os.path.join(here, 'debug_frame.png')}")
