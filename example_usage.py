"""Minimal example for running the amoeba segmentation pipeline.

The script reads one microscopy image, runs the first-stage segmentation, and
saves four files in example_outputs/:

- candidate_mask.png: binary candidate mask
- processed_image.png: filtered grayscale image returned by the pipeline
- labels.png: watershed labels converted to a visible color map
- overlay.png: original image with the candidate mask highlighted
"""

from pathlib import Path

import cv2
import numpy as np

from Metodo_SOMIB import segmentar_ameba_completa


INPUT_IMAGE = Path("images1/amoeba_0001.jpg")
OUTPUT_DIR = Path("example_outputs")


def build_overlay(img_rgb: np.ndarray, mask_bin: np.ndarray) -> np.ndarray:
    """Return an RGB image with the candidate mask highlighted in red."""
    overlay = img_rgb.copy().astype(np.float32)
    mask = mask_bin > 0
    color = np.array([255, 0, 0], dtype=np.float32)
    overlay[mask] = 0.55 * overlay[mask] + 0.45 * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def labels_to_color(labels: np.ndarray) -> np.ndarray:
    """Convert integer watershed labels into a visible color image."""
    labels = labels.astype(np.int32)
    if labels.max() <= 0:
        return np.zeros((*labels.shape, 3), dtype=np.uint8)

    normalized = cv2.normalize(labels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color_bgr[labels == 0] = 0
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

    img_bgr = cv2.imread(str(INPUT_IMAGE))
    if img_bgr is None:
        raise ValueError(f"Could not read image: {INPUT_IMAGE}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask_bin, labels_ws, processed_img = segmentar_ameba_completa(img_rgb=img_rgb)

    OUTPUT_DIR.mkdir(exist_ok=True)

    candidate_mask = (mask_bin > 0).astype(np.uint8) * 255
    cv2.imwrite(str(OUTPUT_DIR / "candidate_mask.png"), candidate_mask)
    cv2.imwrite(str(OUTPUT_DIR / "processed_image.png"), processed_img)

    labels_rgb = labels_to_color(labels_ws)
    overlay_rgb = build_overlay(img_rgb, mask_bin)

    labels_bgr = cv2.cvtColor(labels_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(OUTPUT_DIR / "labels.png"), labels_bgr)
    cv2.imwrite(str(OUTPUT_DIR / "overlay.png"), overlay_bgr)

    print(f"Saved example outputs in: {OUTPUT_DIR.resolve()}")
    print(f"Detected watershed objects: {int(labels_ws.max())}")


if __name__ == "__main__":
    main()
