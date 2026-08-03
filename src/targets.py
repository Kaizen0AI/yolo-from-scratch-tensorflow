import tensorflow as tf
import numpy as np
from src.constants import GRID_SIZE, IMAGE_SIZE, NUM_CLASSES, NUM_ANCHORS, ANCHORS

def normalize_box(box, image_size):
    """
    Normalize bounding boxes to [0, 1] range based on image dimensions.
    """
    h, w = image_size
    x1, y1, x2, y2 = box
    return (x1/w, y1/h, x2/w, y2/h)

def xyxy_to_xywh(box):
    """
    Convert bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h) format.
    The returened x/y are center coordinates, which is the form commonly used by YOLO,
    """
    x1, y1, x2, y2 = box
    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return (x, y, w, h)

def get_grid_cell(box, grid_size):
    """
    Find the grid cell for a given bounding box based on its center.
    """
    x, y, _, _ = box
    grid_h, grid_w = grid_size
    cell_x = min(int(x*grid_w), grid_w - 1)
    cell_y = min(int(y*grid_h), grid_h - 1)
    return (cell_x, cell_y)

def compute_anchor_iou(box, anchors):
    """
    Compute the IoU between a bounding box and a set of anchor boxes.
    """
    _, _, w, h = box
    box_area = w * h
    ious = []
    for anc_w, anc_h in anchors:
        inter_w = min(w, anc_w)
        inter_h = min(h, anc_h)
        inter_area = max(0, inter_w) * max(0, inter_h)
        union_area = box_area + (anc_w*anc_h) - inter_area
        iou = inter_area/union_area if union_area > 0 else 0
        ious.append(iou)
    return np.array(ious, dtype=np.float32)

def assign_box_to_anchor(box, anchors):
    """
    Assign a bounding box to the best matching anchor based on IoU.
    Returns the index of the best anchor and the corresponding IoU.
    """
    ious = compute_anchor_iou(box, anchors)
    return int(np.argmax(ious))

def generate_yolo_target(boxes, labels, image_size=IMAGE_SIZE, grid_size=GRID_SIZE, anchors=ANCHORS, num_anchors=NUM_ANCHORS, num_classes=NUM_CLASSES):
    """
    Generate a YOLO-style target tensor with shape
    (grid_height, grid_width, num_anchors, 5 + num_classes).
    """
    grid_h, grid_w = grid_size
    target = np.zeros((grid_h, grid_w, num_anchors, 5+num_classes), dtype=np.float32)

    for box, label in zip(boxes, labels):
        if box is None or len(box) != 4:
            continue  # Skip invalid boxes
        # Normalize box coordinates
        norm_box = normalize_box(box, image_size)
        # Convert to (x, y, w, h) format
        xywh_box = xyxy_to_xywh(norm_box)
        cx, cy, bw, bh = xywh_box
        # Find the grid cell for the box
        cell_x, cell_y = get_grid_cell(xywh_box, grid_size)
        # Assign to the best matching anchor
        anchor_idx= assign_box_to_anchor(xywh_box, anchors)
        anchor_w, anchor_h = anchors[anchor_idx]
        label = int(label)
        if not (0 <= label < num_classes):
            continue
        # Fill in the target tensor
        tx = (cx * grid_w) - cell_x
        ty = (cy * grid_h) - cell_y
        eps = 1e-6
        tw = np.log(max(bw, eps) / anchor_w)
        th = np.log(max(bh, eps) / anchor_h)
        target[cell_y, cell_x, anchor_idx, 0] = 1.0
        target[cell_y, cell_x, anchor_idx, 1] = tx
        target[cell_y, cell_x, anchor_idx, 2] = ty
        target[cell_y, cell_x, anchor_idx, 3] = tw
        target[cell_y, cell_x, anchor_idx, 4] = th
        target[cell_y, cell_x, anchor_idx, 5 + label] = 1.0
    return tf.convert_to_tensor(target)