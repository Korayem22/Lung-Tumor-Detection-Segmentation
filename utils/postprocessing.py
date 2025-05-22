from skimage.measure import label

def count_fragments(mask):
    """Count number of connected components in binary mask"""
    return label(mask).max()

def calculate_area(mask):
    """Returns area in pixels"""
    return int(mask.sum())
