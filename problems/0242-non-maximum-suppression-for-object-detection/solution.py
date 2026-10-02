import numpy as np

def non_maximum_suppression(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression (NMS) to bounding boxes.
    
    Args:
        boxes: Array-like of shape (N, 4) with boxes in format [x1, y1, x2, y2]
        scores: Array-like of shape (N,) with confidence scores
        iou_threshold: float, IoU threshold for suppression (0 to 1)
    
    Returns:
        List of indices of kept boxes, ordered by descending score
        Returns -1 for invalid inputs
    """
    try:
        boxes = np.array(boxes, dtype=np.float64)
        scores = np.array(scores, dtype=np.float64)
    except:
        return -1
    
    if boxes.size == 0:
        return []
    
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        return -1
    
    if len(boxes) != len(scores):
        return -1
    
    if iou_threshold < 0 or iou_threshold > 1:
        return -1
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        
        if order.size == 1:
            break
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = np.where(union > 0, intersection / union, 0)
        
        mask = iou <= iou_threshold
        order = order[1:][mask]
    
    return keep