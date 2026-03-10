import argparse
import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model import SegmentationModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained segmentation model.")
    parser.add_argument("--images", required=True, help="Directory with input images to segment.")
    parser.add_argument("--model", required=True, help="Path to saved model weights (.pth).")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of classes the model predicts.")
    parser.add_argument("--backbone", choices=["deeplabv3", "unet", "transformer"], default="deeplabv3")
    parser.add_argument("--output-dir", default=None, help="Directory where overlay images will be saved. If omitted, visualizations are shown interactively.")
    return parser.parse_args()


def visualize_prediction(image: Image.Image, mask: np.ndarray, num_classes: int):
    # create color map for classes
    colors = plt.get_cmap("tab20")(range(num_classes))[:, :3]  # RGB
    colors = (colors * 255).astype(np.uint8)

    mask_color = colors[mask]
    mask_img = Image.fromarray(mask_color)
    overlay = Image.blend(image.convert("RGBA"), mask_img.convert("RGBA"), alpha=0.5)
    return overlay


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SegmentationModel(num_classes=args.num_classes, backbone=args.backbone)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    transform = T.Compose([T.ToTensor()])

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for fname in sorted(os.listdir(args.images)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        path = os.path.join(args.images, fname)
        image = Image.open(path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, dict):
                output = output["out"]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        overlay = visualize_prediction(image, pred, args.num_classes)

        if args.output_dir:
            out_path = os.path.join(args.output_dir, fname)
            overlay.save(out_path)
            print(f"Saved overlay to {out_path}")
        else:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Input")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(overlay)
            plt.title("Overlay")
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    main()
