import cv2
import matplotlib.pyplot as plt

def draw_boxes(image, bboxes):
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

def overlay_mask(image, mask):
    overlay = image.copy()
    overlay[mask > 0] = [255, 0, 0]  # Red mask
    return overlay
