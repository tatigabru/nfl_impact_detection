import sys
import os
import cv2
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


def load_image_and_boxes(
    images_dir: str, image_id: str,
):

    image_path = os.path.join(self.images_dir, image_id)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    records = self.marking[self.marking["image_name"] == image_id]
    boxes = records[["x", "y", "w", "h"]].values
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    labels = records["impact"].values
    return image, boxes, labels


def load_image_boxes(
    images_dir: str, image_id: str, labels: pd.DataFrame, format: str = "pascal_voc"
) -> Tuple[np.array, List[int]]:
    """
    Load image and boxes in coco or pascal_voc format
    Args:
            
    """
    image = cv2.imread(f"{images_dir}/{image_id}", cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    records = labels[labels["image"] == image_id]
    # coco format
    boxes = records[["left", "top", "width", "height"]].values
    # pascal voc format
    if format == "pascal_voc":  # xyxy
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    return image, boxes
