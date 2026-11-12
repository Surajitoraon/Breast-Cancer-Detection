"""
generate_synthetic_data.py
---------------------------
Generates realistic synthetic mammography-like training images.

Benign class  → smooth, oval/circular mass; well-defined border;
                uniform internal texture; no calcifications.

Malignant class → irregular, spiculated mass; ill-defined border;
                  heterogeneous internal texture; micro-calcifications.

Run from project root:
    python src/generate_synthetic_data.py
"""

import os
import sys
import random
import math

import numpy as np
import cv2

# Allow running from any working directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

import config

# ── Hyperparams ──────────────────────────────────────────────────
IMG_H, IMG_W   = config.IMG_SIZE          # (224, 224)
N_BENIGN       = 350
N_MALIGNANT    = 350
RANDOM_SEED    = 42
# ────────────────────────────────────────────────────────────────

rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ── Helpers ──────────────────────────────────────────────────────

def mammogram_background(h: int, w: int) -> np.ndarray:
    """Simulate a greyscale mammogram background with glandular tissue texture."""
    base = rng.integers(18, 42, size=(h, w), dtype=np.uint8)
    # Add Perlin-like low-frequency noise via downsampled Gaussian
    noise_small = rng.integers(0, 60, size=(h // 8, w // 8)).astype(np.float32)
    noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    img = cv2.add(base, noise)
    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    return img


def add_glandular_streaks(img: np.ndarray) -> np.ndarray:
    """Add subtle bright fibroglandular streaks to background."""
    overlay = img.copy()
    h, w = img.shape
    for _ in range(rng.integers(4, 10)):
        x1 = rng.integers(0, w)
        y1 = rng.integers(0, h)
        angle = rng.uniform(0, math.pi)
        length = rng.integers(30, 90)
        x2 = int(x1 + length * math.cos(angle))
        y2 = int(y1 + length * math.sin(angle))
        brightness = rng.integers(40, 80)
        thick = rng.integers(1, 3)
        cv2.line(overlay, (x1, y1), (x2, y2), int(brightness), thick)
    alpha = rng.uniform(0.25, 0.5)
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)


def draw_benign_mass(img: np.ndarray) -> np.ndarray:
    """
    Benign mass: smooth oval/circle with well-defined border,
    uniform density, possible small benign calcification (round).
    """
    h, w = img.shape
    cx = rng.integers(w // 4, 3 * w // 4)
    cy = rng.integers(h // 4, 3 * h // 4)
    rx = rng.integers(18, 40)
    ry = rng.integers(18, 40)
    angle = rng.integers(0, 180)

    # Core mass – bright oval
    mass_brightness = rng.integers(140, 210)
    cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360,
                int(mass_brightness), -1)

    # Soft gradient edge – blur the mass region slightly
    mask = np.zeros_like(img)
    cv2.ellipse(mask, (cx, cy), (rx + 4, ry + 4), angle, 0, 360, 255, -1)
    img_blur = cv2.GaussianBlur(img, (9, 9), 3)
    img = np.where(mask > 0, cv2.addWeighted(img, 0.6, img_blur, 0.4, 0), img)

    # Thin bright halo (capsule)
    cv2.ellipse(img, (cx, cy), (rx + 2, ry + 2), angle, 0, 360,
                int(mass_brightness * 0.85), 2)

    # Optional benign round calcification (1–2 round bright dots)
    if rng.random() < 0.4:
        for _ in range(rng.integers(1, 3)):
            ox = rng.integers(-rx // 3, rx // 3)
            oy = rng.integers(-ry // 3, ry // 3)
            cv2.circle(img, (cx + ox, cy + oy), rng.integers(2, 4), 240, -1)
    return img


def draw_malignant_mass(img: np.ndarray) -> np.ndarray:
    """
    Malignant mass: irregular spiculated star-shape with ill-defined border,
    heterogeneous interior, micro-calcifications (tiny bright dots).
    """
    h, w = img.shape
    cx = rng.integers(w // 4, 3 * w // 4)
    cy = rng.integers(h // 4, 3 * h // 4)
    base_r = rng.integers(16, 36)

    # Build irregular polygon (spiculated outline)
    n_spikes = rng.integers(12, 22)
    pts = []
    for i in range(n_spikes):
        theta = 2 * math.pi * i / n_spikes + rng.uniform(-0.15, 0.15)
        # Alternate long/short spicules
        if i % 2 == 0:
            r = base_r + rng.integers(8, 24)   # spike tip
        else:
            r = base_r - rng.integers(2, 10)   # valley
        x = int(cx + r * math.cos(theta))
        y = int(cy + r * math.sin(theta))
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32)

    mass_brightness = rng.integers(130, 195)
    cv2.fillPoly(img, [pts], int(mass_brightness))

    # Heterogeneous interior – add noise inside
    interior_mask = np.zeros_like(img)
    cv2.fillPoly(interior_mask, [pts], 255)
    interior_noise = rng.integers(-30, 30, size=img.shape).astype(np.int16)
    img_int = img.astype(np.int16)
    img_int = np.where(
        interior_mask > 0,
        np.clip(img_int + interior_noise, 0, 255),
        img_int
    )
    img = img_int.astype(np.uint8)

    # Ill-defined border – slight blur only at edge
    edge_mask = np.zeros_like(img)
    cv2.polylines(edge_mask, [pts], True, 255, 6)
    img_blur = cv2.GaussianBlur(img, (7, 7), 2)
    img = np.where(edge_mask > 0, img_blur, img)

    # Draw individual spicule lines radiating outward
    for _ in range(rng.integers(8, 18)):
        theta = rng.uniform(0, 2 * math.pi)
        r_start = int(base_r * 0.8)
        r_end   = base_r + rng.integers(20, 50)
        x1 = int(cx + r_start * math.cos(theta))
        y1 = int(cy + r_start * math.sin(theta))
        x2 = int(cx + r_end   * math.cos(theta))
        y2 = int(cy + r_end   * math.sin(theta))
        thick = rng.integers(1, 2)
        cv2.line(img, (x1, y1), (x2, y2), int(mass_brightness * 0.75), thick)

    # Micro-calcifications: scatter tiny bright dots
    n_calcs = rng.integers(6, 20)
    for _ in range(n_calcs):
        spread = base_r + 25
        ox = rng.integers(-spread, spread)
        oy = rng.integers(-spread, spread)
        px, py = cx + ox, cy + oy
        if 0 <= px < w and 0 <= py < h:
            radius = rng.integers(1, 3)
            cv2.circle(img, (px, py), radius, 245, -1)

    return img


def to_rgb(gray: np.ndarray) -> np.ndarray:
    """Convert greyscale HxW to RGB HxWx3 (pseudo-mammogram coloring)."""
    # Slight warm tint (mammograms often have a warm greyscale look)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    # Subtle red/warm channel enhancement
    bgr[:, :, 2] = np.clip(bgr[:, :, 2] * 1.04, 0, 255)
    bgr[:, :, 0] = np.clip(bgr[:, :, 0] * 0.97, 0, 255)
    return bgr.astype(np.uint8)


def generate_image(label: str) -> np.ndarray:
    """Generate one synthetic mammogram image for the given label."""
    img = mammogram_background(IMG_H, IMG_W)
    img = add_glandular_streaks(img)

    if label == "benign":
        # Sometimes add 2 separate benign masses
        img = draw_benign_mass(img)
        if rng.random() < 0.25:
            img = draw_benign_mass(img)
    else:  # malignant
        img = draw_malignant_mass(img)
        # Possibly add a small satellite lesion
        if rng.random() < 0.3:
            img = draw_malignant_mass(img)

    # Apply subtle overall blur (simulating tissue scatter)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)

    # CLAHE-like contrast enhancement (match preprocessing pipeline)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return to_rgb(img)


def generate_split(split: str, label: str, n: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Generating {n} {label} images for {split} → {out_dir}")
    for i in range(n):
        img = generate_image(label)
        path = os.path.join(out_dir, f"{label}_{i:04d}.png")
        cv2.imwrite(path, img)
    print(f"  ✓ Done ({n} images)")


def main():
    print("=" * 60)
    print("Synthetic Mammography Dataset Generator")
    print("=" * 60)

    splits = {
        "train":  {"benign": N_BENIGN,         "malignant": N_MALIGNANT},
        "val":    {"benign": N_BENIGN // 5,     "malignant": N_MALIGNANT // 5},
        "test":   {"benign": N_BENIGN // 5,     "malignant": N_MALIGNANT // 5},
    }

    for split, counts in splits.items():
        print(f"\n[{split.upper()}]")
        for label, n in counts.items():
            out_dir = os.path.join(config.DATA_DIR, split, label)
            # Remove old placeholder images
            if os.path.isdir(out_dir):
                for f in os.listdir(out_dir):
                    if f.endswith(".png"):
                        os.remove(os.path.join(out_dir, f))
            generate_split(split, label, n, out_dir)

    total = sum(
        c for sp in splits.values() for c in sp.values()
    )
    print(f"\n✅ Dataset generation complete — {total} images total.")
    print("   Run 'python src/train.py' to retrain the model.")


if __name__ == "__main__":
    main()
