# sculpture_detector.py  (clean, import-safe)
"""
Sculpture dataset and training helper for Mask R-CNN.

Place this file in:
D:\heritage_project\src\MASK_RCNN\sculpture_detector.py

Import-safe: top-level only defines classes and helpers. Heavy imports are local.
"""

from __future__ import annotations
import os
import json
import datetime
import numpy as np
import skimage.draw
import skimage.io
import skimage.color

# Import Mask R-CNN classes (these should resolve to your local mrcnn package)
# Keep these top-level imports minimal; if you have circular import problems,
# move them inside functions that need them.
try:
    from mrcnn.config import Config
    from mrcnn import utils
except Exception as e:
    # Raise a clear error so import-time failures are easier to diagnose
    raise ImportError(
        "Failed to import mrcnn package. Ensure the repository root is on PYTHONPATH "
        "and mrcnn is available. Original error: " + str(e)
    )

# Default root and weights (adjust if needed)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


############################################################
#  Configurations
############################################################
class SculptureConfig(Config):
    """Configuration for training on the Sculpture dataset."""
    NAME = "sculpture"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1  # background + sculpture
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9
    # You can add other config overrides here if needed.


############################################################
#  Dataset
############################################################
class SculptureDataset(utils.Dataset):
    """Dataset class for the Sculpture dataset stored in VIA JSON format."""

    def load_sculpture(self, dataset_dir: str, subset: str):
        """Load a subset of the Sculpture dataset.
        dataset_dir: root dir containing 'train' and 'val' subfolders.
        subset: 'train' or 'val'
        """
        assert subset in ("train", "val"), "subset must be 'train' or 'val'"
        subset_dir = os.path.join(dataset_dir, subset)

        # Add class
        self.add_class("sculpture", 1, "sculpture")

        # VIA annotation file expected at dataset_dir/subset/via_region_data.json
        ann_path = os.path.join(subset_dir, "via_region_data.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"VIA annotations not found: {ann_path}")

        with open(ann_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        # Convert to list (VIA stores an object keyed by filenames in some versions)
        annotations = list(annotations.values())

        # Keep only annotated images
        annotations = [a for a in annotations if a.get("regions")]

        for a in annotations:
            regions = a["regions"]
            # VIA 1.x: regions is dict keyed by region_id; VIA 2.x: regions is list
            if isinstance(regions, dict):
                regions = [r["shape_attributes"] for r in regions.values()]
            else:
                regions = [r["shape_attributes"] for r in regions]

            image_path = os.path.join(subset_dir, a["filename"])
            if not os.path.exists(image_path):
                # Skip missing images (warn)
                print(f"Warning: {image_path} not found. Skipping.")
                continue

            # Read image shape
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            # Save image info with polygons
            self.add_image(
                "sculpture",
                image_id=a["filename"],
                path=image_path,
                width=width, height=height,
                polygons=regions
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image_id.

        Returns:
          masks: bool array [height, width, instance_count]
          class_ids: 1D array of class IDs for each instance mask
        """
        info = self.image_info[image_id]
        if info["source"] != "sculpture":
            # Delegate to parent if not sculpture image
            return super().load_mask(image_id)

        polygons = info.get("polygons", [])
        if not polygons:
            return np.empty((0, 0, 0)), np.array([], dtype=np.int32)

        height = info["height"]
        width = info["width"]
        masks = np.zeros((height, width, len(polygons)), dtype=np.uint8)

        for i, p in enumerate(polygons):
            all_x = p.get("all_points_x", []) or p.get("all_x", [])
            all_y = p.get("all_points_y", []) or p.get("all_y", [])
            if not all_x or not all_y:
                # Skip invalid polygon
                continue
            # Convert to integer indices and clip to image
            rr, cc = skimage.draw.polygon(all_y, all_x, shape=(height, width))
            rr = np.clip(rr, 0, height - 1)
            cc = np.clip(cc, 0, width - 1)
            masks[rr, cc, i] = 1

        if masks.size == 0:
            return np.empty((0, 0, 0)), np.array([], dtype=np.int32)

        # Return boolean mask and class IDs (all ones)
        return masks.astype(bool), np.ones([masks.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "sculpture":
            return info["path"]
        return super().image_reference(image_id)


############################################################
#  Training helper (only used when executing the script)
############################################################
def train(model, dataset_root: str):
    """Train the model's heads on the dataset in dataset_root."""
    dataset_train = SculptureDataset()
    dataset_train.load_sculpture(dataset_root, "train")
    dataset_train.prepare()

    dataset_val = SculptureDataset()
    dataset_val.load_sculpture(dataset_root, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=model.config.LEARNING_RATE,
                epochs=30,
                layers='heads')


############################################################
#  Utilities for color splash and CLI
############################################################
def color_splash(image, mask):
    """Apply color splash effect to the image using instance masks."""
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    if mask.shape[-1] > 0:
        mask_combined = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask_combined, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    """Run detection and save result; heavy libs imported locally."""
    assert image_path or video_path

    if image_path:
        image = skimage.io.imread(image_path)
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        r = model.detect([image], verbose=1)[0]
        splash = color_splash(image, r['masks'])
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
        print("Saved to", file_name)
        return file_name

    # Video branch (optional)
    import cv2  # local import
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)

    file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
    vwriter = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

    count = 0
    success = True
    while success:
        success, image = vcapture.read()
        if not success:
            break
        image = image[..., ::-1]  # BGR -> RGB
        r = model.detect([image], verbose=0)[0]
        splash = color_splash(image, r['masks'])
        splash = splash[..., ::-1]  # RGB -> BGR
        vwriter.write(splash)
        count += 1

    vwriter.release()
    print("Saved to", file_name)
    return file_name


############################################################
#  Command-line interface (only executed when running this script)
############################################################
if __name__ == "__main__":
    import argparse
    from mrcnn import model as modellib

    parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect sculptures.')
    parser.add_argument("command", metavar="<command>", help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False, metavar="/path/to/sculpture/dataset/", help='Directory of the Sculpture dataset')
    parser.add_argument('--weights', required=True, metavar="/path/to/weights.h5", help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False, default=os.path.join(ROOT_DIR, "logs"), metavar="/path/to/logs/", help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False, metavar="path or URL to image", help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False, metavar="path or URL to video", help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "--dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, "Provide --image or --video for splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = SculptureConfig()
    else:
        class InferenceConfig(SculptureConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights path to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Run the requested command
    if args.command == "train":
        train(model, args.dataset)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image, video_path=args.video)
    else:
        print(f"Unknown command: {args.command}")
