import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from models.unet import UNet
from models.att_unet import AttentionUNet
from utils.postprocess import fit_cranium_ellipse, mask_from_logits
from utils.preprocess import denoise_image, apply_clahe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer cranium mask and extract BPD/CFD points")
    parser.add_argument("--weights", type=str, default=r"/workspace/origin_med_assignment/weights/run_20260102_001846/best.pt")
    parser.add_argument("--model", type=str, default="attention_unet", choices=["unet", "attention_unet"])
    parser.add_argument("--images_dir", type=str, default=r"/workspace/origin_med_assignment/Task - Segmentation/images")
    parser.add_argument("--output_dir", type=str, default="/workspace/outputs")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask probability threshold")
    return parser.parse_args()


def load_model(weights_path: Path, device: torch.device, model_name: str) -> torch.nn.Module:
    checkpoint = torch.load(weights_path, map_location=device)
    if model_name == "attention_unet":
        model = AttentionUNet(n_channels=1, n_classes=1)
    else:
        model = UNet(n_channels=1, n_classes=1)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = Path(args.weights)
    model = load_model(weights_path, device, args.model)

    image_tf = T.Compose(
        [
            T.Lambda(lambda img: apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))),
            T.Lambda(lambda img: denoise_image(img, h=8)),
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    vis_dir = output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {images_dir}")

    for img_path in image_paths:
        img_pil = Image.open(img_path).convert("L")
        orig_w, orig_h = img_pil.size
        tensor = image_tf(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)[0, 0].cpu().numpy()

        # Resize back to original size for measurements
        logits_resized = cv2.resize(logits, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask_bin = mask_from_logits(logits_resized, threshold=args.threshold)

        measurement = fit_cranium_ellipse(mask_bin)
        if measurement is None:
            print(f"[WARN] Could not fit ellipse for {img_path.name}")
        else:
            bpd = measurement.bpd_points
            cfd = measurement.cfd_points
            print(
                f"{img_path.name}: BPD pts={bpd}, length={measurement.major_axis:.2f} | "
                f"CFD pts={cfd}, length={measurement.minor_axis:.2f} | angle={measurement.angle_deg:.1f}"
            )

        # Save binary mask
        mask_save_path = mask_dir / img_path.name
        Image.fromarray(mask_bin).save(mask_save_path)

        # Save overlay visualization
        overlay = np.array(img_pil.convert("RGB"))
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(overlay, 0.7, cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR), 0.3, 0)
        if measurement:
            cv2.ellipse(
                overlay,
                (int(measurement.center[0]), int(measurement.center[1])),
                (int(measurement.major_axis / 2), int(measurement.minor_axis / 2)),
                measurement.angle_deg,
                0,
                360,
                (0, 255, 0),
                2,
            )
            cv2.line(overlay, measurement.bpd_points[0], measurement.bpd_points[1], (0, 0, 255), 2)
            cv2.line(overlay, measurement.cfd_points[0], measurement.cfd_points[1], (255, 0, 0), 2)
            cv2.circle(overlay, measurement.bpd_points[0], 3, (0, 0, 255), -1)
            cv2.circle(overlay, measurement.bpd_points[1], 3, (0, 0, 255), -1)
            cv2.circle(overlay, measurement.cfd_points[0], 3, (255, 0, 0), -1)
            cv2.circle(overlay, measurement.cfd_points[1], 3, (255, 0, 0), -1)

        vis_save_path = vis_dir / img_path.name
        cv2.imwrite(str(vis_save_path), overlay)


if __name__ == "__main__":
    main()

