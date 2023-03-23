
def iou_score(pred_mask, target_mask):
    assert pred_mask.shape == target_mask.shape, "Shape mismatch in input masks."

    # Make sure the input tensors are boolean
    pred_mask = pred_mask.bool()
    target_mask = target_mask.bool()

    # Calculate intersection
    intersection = (pred_mask & target_mask).float().sum()

    # Calculate union
    union = (pred_mask | target_mask).float().sum()

    # Calculate IoU
    iou = intersection / (union + 1e-6)  # Adding a small epsilon to avoid division by zero

    return iou


