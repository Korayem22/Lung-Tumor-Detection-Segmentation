import cv2
from models.detection_interface import detect_tumor
from models.segmentation_full import segment_full_image
from models.segmentation_cropped import segment_cropped
from utils.visualization import draw_boxes, overlay_mask
from utils.postprocessing import count_fragments, calculate_area
from utils.pdf_report import generate_report

image_path = r"C:\\Users\\alaa6\Documents\\ITI\Advanced Computer Vision\\Lung-Tumor-Detection-Segmentation\Screenshot 2025-05-02 163905.png"
image = cv2.imread(image_path)

# Detection
bboxes = detect_tumor(image_path)
image_with_boxes = draw_boxes(image.copy(), bboxes)

# Choose strategy
use_cropped = True
if use_cropped:
    combined_mask = image[:, :, 0] * 0
    for box in bboxes:
        x1, y1, x2, y2 = box
        cropped = image[y1:y2, x1:x2]
        mask = segment_cropped(image, box)
        combined_mask[y1:y2, x1:x2] = mask
else:
    combined_mask = segment_full_image(image)

# Postprocess
fragments = count_fragments(combined_mask)
area = calculate_area(combined_mask)
existence = "Yes" if area > 0 else "No"

# Save outputs
cv2.imwrite("outputs/predictions/result_overlay.png", overlay_mask(image, combined_mask))
generate_report("outputs/reports/tumor_report.pdf", existence, bboxes, area, fragments)
