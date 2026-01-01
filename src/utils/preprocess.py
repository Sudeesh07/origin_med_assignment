from __future__ import annotations

from typing import Union

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random


def apply_clahe(img: Image.Image, clip_limit: float = 1.5, tile_grid_size: tuple[int, int] = (8, 8)) -> Image.Image:
    """
    Contrast Limited Adaptive Histogram Equalization tuned for ultrasound.
    """
    arr = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(arr)
    return Image.fromarray(enhanced)


def denoise_image(img: Union[Image.Image, np.ndarray], h: int = 10) -> Image.Image:
    """
    Light denoising for fetal brain ultrasound using OpenCV fastNlMeansDenoising.
    Args:
        img: PIL grayscale image or numpy array.
        h: filter strength; larger removes more noise but can oversmooth.
    """
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("L"))
    else:
        arr = img
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    arr = arr.astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(arr, h=h, templateWindowSize=7, searchWindowSize=21)
    return Image.fromarray(denoised)


def random_augment(
    img: Image.Image,
    mask: Image.Image,
    p_flip: float = 0.5,
    p_vflip: float = 0.2,
    max_rot: float = 10.0,
    max_scale: float = 0.05,
    p_speckle: float = 0.3,
    speckle_std: float = 0.02,
    p_blur: float = 0.2,
    blur_kernel: int = 3,
    p_intensity: float = 0.4,
    brightness_delta: float = 0.08,
    contrast_delta: float = 0.08,
) -> tuple[Image.Image, Image.Image]:
    """
    Apply paired augmentations (same params) to ultrasound image and mask.
    """
    # Horizontal flip
    if random.random() < p_flip:
        img = TF.hflip(img)
        mask = TF.hflip(mask)
    # Vertical flip (less frequent)
    if random.random() < p_vflip:
        img = TF.vflip(img)
        mask = TF.vflip(mask)

    # Small random rotation + scale via affine
    angle = random.uniform(-max_rot, max_rot)
    scale = 1.0 + random.uniform(-max_scale, max_scale)
    shear = 0.0
    translate = (0, 0)

    img = TF.affine(img, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
    mask = TF.affine(mask, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=TF.InterpolationMode.NEAREST, fill=0)

    # Ultrasound-specific intensity tweaks (image only)
    if random.random() < p_intensity:
        # brightness/contrast jitter within small bounds
        b_factor = 1.0 + random.uniform(-brightness_delta, brightness_delta)
        c_factor = 1.0 + random.uniform(-contrast_delta, contrast_delta)
        img = TF.adjust_brightness(img, b_factor)
        img = TF.adjust_contrast(img, c_factor)

    if random.random() < p_blur:
        k = max(3, blur_kernel | 1)  # ensure odd kernel
        arr = np.array(img)
        arr = cv2.GaussianBlur(arr, (k, k), 0)
        img = Image.fromarray(arr)

    if random.random() < p_speckle:
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0.0, speckle_std, size=arr.shape).astype(np.float32)
        arr = arr + arr * noise
        arr = np.clip(arr, 0.0, 1.0)
        img = Image.fromarray((arr * 255).astype(np.uint8))
    return img, mask

