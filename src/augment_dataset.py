import argparse
from pathlib import Path
import random
from typing import Optional, Tuple

from PIL import Image

from utils.preprocess import apply_clahe, denoise_image, random_augment


def find_mask(img_path: Path, masks_dir: Optional[Path]) -> Optional[Path]:
    if masks_dir is None:
        return None
    candidates = [
        masks_dir / f"{img_path.stem}.png",
        masks_dir / f"{img_path.stem}_Annotation.png",
        masks_dir / f"{img_path.stem}_annotation.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def preprocess(img: Image.Image, clahe_clip: float, denoise_h: int) -> Image.Image:
    img = apply_clahe(img, clip_limit=clahe_clip, tile_grid_size=(8, 8))
    img = denoise_image(img, h=denoise_h)
    return img


def process_one(
    img_path: Path,
    mask_path: Optional[Path],
    out_images: Path,
    out_masks: Optional[Path],
    num_aug: int,
    clahe_clip: float,
    denoise_h: int,
    augment_kwargs: dict,
    seed: Optional[int] = None,
) -> None:
    base_img = Image.open(img_path).convert("L")
    base_mask = Image.open(mask_path).convert("L") if mask_path else None

    # deterministic seed per image for reproducibility if provided
    if seed is not None:
        random.seed(seed)

    # Always save a preprocessed "base" version
    img_prep = preprocess(base_img, clahe_clip, denoise_h)
    out_images.mkdir(parents=True, exist_ok=True)
    base_name = img_path.stem
    img_prep.save(out_images / f"{base_name}_prep.png")
    if base_mask and out_masks:
        out_masks.mkdir(parents=True, exist_ok=True)
        base_mask.save(out_masks / f"{base_name}_mask.png")

    # Generate augmentations
    for i in range(num_aug):
        img_aug, mask_aug = random_augment(img_prep, base_mask if base_mask else Image.new("L", base_img.size), **augment_kwargs)
        img_aug.save(out_images / f"{base_name}_aug{i+1}.png")
        if base_mask and out_masks:
            mask_aug.save(out_masks / f"{base_name}_aug{i+1}_mask.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply ultrasound preprocessing + augmentation to a dataset.")
    p.add_argument("--images_dir", type=str, required=True, help="Folder with input images (PNG).")
    p.add_argument("--masks_dir", type=str, default=None, help="Optional folder with masks; matched by stem or stem_Annotation.")
    p.add_argument("--output_dir", type=str, default="augmented", help="Root folder to write processed data.")
    p.add_argument("--num_aug", type=int, default=3, help="Number of augmented variants per image.")
    p.add_argument("--clahe_clip", type=float, default=1.5, help="CLAHE clip limit.")
    p.add_argument("--denoise_h", type=int, default=8, help="Denoise strength for fastNlMeansDenoising.")
    p.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility.")
    return p.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    out_root = Path(args.output_dir)
    out_images = out_root / "images"
    out_masks = out_root / "masks" if masks_dir else None

    augment_kwargs = dict(
        p_flip=0.5,
        p_vflip=0.2,
        max_rot=8.0,
        p_speckle=0.4,
        speckle_std=0.02,
        p_blur=0.2,
        blur_kernel=3,
    )

    img_paths = sorted(images_dir.glob("*.png"))
    if not img_paths:
        raise FileNotFoundError(f"No PNG images found in {images_dir}")

    for img_path in img_paths:
        mask_path = find_mask(img_path, masks_dir)
        process_one(
            img_path,
            mask_path,
            out_images,
            out_masks,
            num_aug=args.num_aug,
            clahe_clip=args.clahe_clip,
            denoise_h=args.denoise_h,
            augment_kwargs=augment_kwargs,
            seed=args.seed,
        )

    print(f"Saved preprocessed/augmented images to {out_images}")
    if out_masks:
        print(f"Saved masks to {out_masks}")


if __name__ == "__main__":
    main()

