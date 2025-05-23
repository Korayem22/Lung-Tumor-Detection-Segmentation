import numpy as np
import torch
import torch.nn.functional as F
import cv2
import logging
import os
import sys
from glob import glob
import matplotlib.pyplot as plt
import random

# Set up project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.unet import UNet

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("segmentation.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Device & Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet()
logger.info("Loading UNet model weights...")
model.load_state_dict(torch.load("FULL_IMAGE/Full_Image_segmentation_pytorch_unet_model.pth", map_location=device))
model.to(device)
model.eval()
logger.info("Model loaded successfully.")

# --- Utils ---
def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image).unsqueeze(0).to(device)

def postprocess_mask(mask, original_shape):
    mask = F.interpolate(mask, size=original_shape, mode='bilinear', align_corners=False)
    return (mask > 0.5).float().squeeze().cpu().numpy().astype(np.uint8)

def segment_full_image(image):
    try:
        original_shape = image.shape[:2]
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
        return postprocess_mask(output, original_shape)
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise

# --- Metrics ---
def dice_score(pred, target):
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    return (2. * intersection + 1e-6) / (union + 1e-6)

def iou_score(pred, target):
    intersection = np.sum(pred * target)
    union = np.sum((pred + target) > 0)
    return (intersection + 1e-6) / (union + 1e-6)

def precision_score(pred, target):
    tp = np.sum(pred * target)
    fp = np.sum(pred * (1 - target))
    return (tp + 1e-6) / (tp + fp + 1e-6)

def recall_score(pred, target):
    tp = np.sum(pred * target)
    fn = np.sum((1 - pred) * target)
    return (tp + 1e-6) / (tp + fn + 1e-6)

# --- Inference ---
def run_batch_inference(val_root="val", display_images=5):
    image_paths = glob(os.path.join(val_root, "images", "*", "*.png"))
    random.shuffle(image_paths)
    logger.info(f"Found {len(image_paths)} total images.")

    all_dice, all_iou, all_precision, all_recall = [], [], [], []
    display_data = []

    for i, image_path in enumerate(image_paths):
        subject = os.path.basename(os.path.dirname(image_path))
        filename = os.path.basename(image_path)

        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            continue

        mask_path = os.path.join(val_root, "masks", subject, filename)
        mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gt is None:
            logger.warning(f"No ground truth for {filename}. Skipping.")
            continue

        pred_mask = segment_full_image(image)

        if mask_gt.shape != pred_mask.shape:
            mask_gt = cv2.resize(mask_gt, (pred_mask.shape[1], pred_mask.shape[0]))
        mask_gt_bin = (mask_gt > 127).astype(np.uint8)

        dice = dice_score(pred_mask, mask_gt_bin)
        iou = iou_score(pred_mask, mask_gt_bin)
        precision = precision_score(pred_mask, mask_gt_bin)
        recall = recall_score(pred_mask, mask_gt_bin)

        all_dice.append(dice)
        all_iou.append(iou)
        all_precision.append(precision)
        all_recall.append(recall)

        logger.info(f"{filename} | Dice: {dice:.4f}, IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        out_dir = os.path.join(val_root, "segmentations_prediction", subject)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, filename), pred_mask * 255)

        if len(display_data) < display_images:
            display_data.append((image, mask_gt_bin, pred_mask, f"{subject}/{filename}\nDice: {dice:.4f}"))

    # --- Combined Visualization (3 rows x 5 cols) ---
    logger.info("Generating combined visualization...")
    fig, axs = plt.subplots(3, display_images, figsize=(4 * display_images, 10))

    for i in range(display_images):
        img, gt, pred, title = display_data[i]

        axs[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0, i].set_title("Original")
        axs[0, i].axis("off")

        axs[1, i].imshow(gt, cmap="gray")
        axs[1, i].set_title("Ground Truth")
        axs[1, i].axis("off")

        axs[2, i].imshow(pred, cmap="gray")
        axs[2, i].set_title(title)
        axs[2, i].axis("off")

    plt.tight_layout()
    plt.show()

    # --- Final Report ---
    logger.info("====== AVERAGE METRICS ======")
    logger.info(f"Dice:     {np.mean(all_dice):.4f}")
    logger.info(f"IoU:      {np.mean(all_iou):.4f}")
    logger.info(f"Precision:{np.mean(all_precision):.4f}")
    logger.info(f"Recall:   {np.mean(all_recall):.4f}")

# --- Main ---
if __name__ == "__main__":
    run_batch_inference(val_root="val")
    logger.info("Segmentation Inference complete.")
