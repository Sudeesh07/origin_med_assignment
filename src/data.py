import os
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class FetalBrainDataset(Dataset):
    """
    Dataset that pairs images and masks located in two folders.
    Filenames are matched by base name (e.g., 000_HC.png ↔ 000_HC_Annotation.png).
    """

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None,
        samples: Optional[list[tuple[Path, Path]]] = None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.joint_transform = joint_transform

        self.samples = samples if samples is not None else self._gather_samples()
        if not self.samples:
            raise ValueError("No image/mask pairs were found.")

    def _gather_samples(self) -> list[tuple[Path, Path]]:
        image_files = {p.stem: p for p in self.images_dir.glob("*.png")}
        pairs: list[tuple[Path, Path]] = []
        for mask_path in self.masks_dir.glob("*.png"):
            # remove trailing suffix after last underscore for mask stem match
            base = mask_path.stem
            # attempt direct match first
            if base in image_files:
                img_path = image_files[base]
                pairs.append((img_path, mask_path))
                continue
            # fallback: strip common annotation suffix
            if base.endswith("_Annotation"):
                trimmed = base.replace("_Annotation", "")
                img_path = image_files.get(trimmed)
                if img_path:
                    pairs.append((img_path, mask_path))
        return sorted(pairs, key=lambda t: t[0].name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = T.ToTensor()(mask)
        mask = (mask > 0.5).float()

        return {"image": image, "mask": mask, "image_path": str(img_path)}

