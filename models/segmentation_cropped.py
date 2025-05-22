import numpy as np

def segment_cropped(image, bbox):

    print("Mock: Running cropped segmentation...")
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    return np.zeros((h, w), dtype=np.uint8)
