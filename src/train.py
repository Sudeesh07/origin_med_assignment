import argparse
from datetime import datetime
from pathlib import Path
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from tqdm import tqdm

from data import FetalBrainDataset
from models.unet import UNet
from models.att_unet import AttentionUNet
from utils.metrics import dice_coefficient
from utils.preprocess import denoise_image, apply_clahe, random_augment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UNet for fetal brain cranium segmentation")
    parser.add_argument("--data_root", type=str, default="Task - Segmentation", help="Root containing images/ and masks/")
    parser.add_argument("--images_dir", type=str, default=None, help="Override images directory")
    parser.add_argument("--masks_dir", type=str, default=None, help="Override masks directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="weights")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "attention_unet"], help="Backbone model")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_datasets(args: argparse.Namespace):
    images_dir = Path(args.images_dir) if args.images_dir else Path(args.data_root) / "images"
    masks_dir = Path(args.masks_dir) if args.masks_dir else Path(args.data_root) / "masks"

    train_joint = partial(
        random_augment,
        p_flip=0.5,
        p_vflip=0.2,
        max_rot=10,
        max_scale=0.05,
        p_speckle=0.4,
        speckle_std=0.025,
        p_blur=0.2,
        blur_kernel=3,
        p_intensity=0.5,
        brightness_delta=0.08,
        contrast_delta=0.1,
    )
    clahe_fn = partial(apply_clahe, clip_limit=2.0, tile_grid_size=(8, 8))
    denoise_fn = partial(denoise_image, h=8)

    train_image_tf = T.Compose(
        [
            T.Lambda(clahe_fn),
            T.Lambda(denoise_fn),
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    val_image_tf = T.Compose(
        [
            T.Lambda(clahe_fn),
            T.Lambda(denoise_fn),
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    mask_tf = T.Compose(
        [
            T.Resize((args.image_size, args.image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ]
    )

    # Build base dataset to get sample list, then split indices for separate transforms
    base_dataset = FetalBrainDataset(images_dir, masks_dir, transform=None, mask_transform=None)
    val_size = max(1, int(len(base_dataset) * args.val_split))
    train_size = len(base_dataset) - val_size
    g = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(base_dataset), generator=g).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_samples = [base_dataset.samples[i] for i in train_indices]
    val_samples = [base_dataset.samples[i] for i in val_indices]

    train_ds = FetalBrainDataset(
        images_dir,
        masks_dir,
        transform=train_image_tf,
        mask_transform=mask_tf,
        joint_transform=train_joint,
        samples=train_samples,
    )
    val_ds = FetalBrainDataset(
        images_dir,
        masks_dir,
        transform=val_image_tf,
        mask_transform=mask_tf,
        joint_transform=None,
        samples=val_samples,
    )
    return train_ds, val_ds


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss_bce = criterion(logits, masks)
            probs = torch.sigmoid(logits)
            dice = dice_coefficient(probs, masks)
            loss = 0.5 * loss_bce + (1 - dice)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        epoch_dice += dice.item()

    return epoch_loss / len(loader), epoch_dice / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    epoch_dice = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss_bce = criterion(logits, masks)
            probs = torch.sigmoid(logits)
            dice = dice_coefficient(probs, masks)
            loss = 0.5 * loss_bce + (1 - dice)
            epoch_loss += loss.item()
            epoch_dice += dice.item()

    return epoch_loss / len(loader), epoch_dice / len(loader)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = get_datasets(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.model == "attention_unet":
        model = AttentionUNet(n_channels=1, n_classes=1).to(device)
    else:
        model = UNet(n_channels=1, n_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = save_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_dice = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.4f}, dice: {train_dice:.4f} | "
            f"Val loss: {val_loss:.4f}, dice: {val_dice:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "val_loss": val_loss,
                "val_dice": val_dice,
            }
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_dice": val_dice,
                    "epoch": epoch,
                    "args": vars(args),
                    "run_id": run_id,
                    "history": history,
                },
                run_dir / "best.pt",
            )
            print(f"Saved new best model with Dice {val_dice:.4f}")

    # Save final checkpoint for this run with full metadata (including history)
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": args.epochs,
            "best_val_dice": best_dice,
            "args": vars(args),
            "run_id": run_id,
            "history": history,
        },
        run_dir / "last.pt",
    )


if __name__ == "__main__":
    main()

