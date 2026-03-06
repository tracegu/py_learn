import os
import json
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation.

    Expects a directory of images and a JSON annotation file describing
    segmentation masks in COCO-like format. The JSON should contain an
    "annotations" list with items that have at least the keys:
        - "image_id": filename of the image (not full path)
        - "segmentation": list of polygons (each polygon is a list of
          x,y coordinates) or a single polygon
        - "category_id": integer label for the region

    Example:
        {
            "annotations": [
                {
                    "image_id": "img1.png",
                    "segmentation": [[x1, y1, x2, y2, ..., xn, yn], ...],
                    "category_id": 1
                },
                ...
            ]
        }

    The dataset will rasterize the polygons to create a per-pixel mask
    with the category ids as integer labels.
    """

    def __init__(
        self,
        images_dir: str,
        json_path: str,
        transform=None,
        target_transform=None,
    ):
        self.images_dir = images_dir
        self.transform = transform
        self.target_transform = target_transform

        # collect image files (recursive?) just top-level
        self.images = []
        for fname in sorted(os.listdir(images_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                self.images.append(os.path.join(images_dir, fname))

        # load annotations and index by image filename
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.annotations = {}
        for ann in data.get("annotations", []):
            img_name = ann.get("image_id")
            if img_name is None:
                continue
            seg = ann.get("segmentation")
            cat = ann.get("category_id", 0)
            if seg is None:
                continue
            self.annotations.setdefault(img_name, []).append((seg, cat))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_name = os.path.basename(img_path)
        image = Image.open(img_path).convert("RGB")

        # create mask of same size
        width, height = image.size
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        anns = self.annotations.get(img_name, [])
        for seg, cat in anns:
            # segmentation may be nested list (multiple polygons)
            if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
                # multiple polygons
                for poly in seg:
                    draw.polygon(poly, fill=int(cat))
            else:
                draw.polygon(seg, fill=int(cat))

        mask = np.array(mask, dtype=np.int64)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(mask)

        return image, mask
