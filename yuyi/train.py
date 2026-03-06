import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import SegmentationDataset
from model import SegmentationModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument("--images", required=True, help="Directory containing input images.")
    parser.add_argument("--annotations", required=True, help="Path to JSON annotations file.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of segmentation classes (including background).")
    parser.add_argument("--output", default="model.pth", help="Path to save the trained model.")
    parser.add_argument("--backbone", choices=["deeplabv3", "unet"], default="deeplabv3", help="Model backbone to use.")
    args = parser.parse_args()
    return args


def train_one_epoch(
    model, loader, criterion, optimizer, device, epoch, print_freq=10
):
    model.train()
    running_loss = 0.0
    for i, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)["out"] if isinstance(model(images), dict) else model(images)
        loss = criterion(outputs, masks.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % print_freq == 0:
            avg = running_loss / print_freq
            print(f"Epoch {epoch} [{i+1}/{len(loader)}] loss={avg:.4f}")
            running_loss = 0.0


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = T.Compose([T.ToTensor()])
    target_transform = torch.from_numpy

    dataset = SegmentationDataset(
        images_dir=args.images,
        json_path=args.annotations,
        transform=transform,
        target_transform=target_transform,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = SegmentationModel(num_classes=args.num_classes, backbone=args.backbone)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, loader, criterion, optimizer, device, epoch)

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed/60:.2f} minutes")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
