import tensorflow as tf
import numpy as np
from itertools import combinations, permutations

from src.constants import (
    GRID_SIZE,
    IMAGE_SIZE,
    NUM_CLASSES,
    NUM_ANCHORS,
    ANCHORS,
)


def normalize_box(box, image_size):
    """
    Normalize bounding boxes to [0, 1] range based on image dimensions.
    """
    h, w = image_size
    x1, y1, x2, y2 = box

    return (x1 / w, y1 / h, x2 / w, y2 / h)


def xyxy_to_xywh(box):
    """
    Convert (x1, y1, x2, y2) to (cx, cy, w, h).
    """
    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    return (cx, cy, w, h)


def get_grid_cell(box, grid_size):
    """
    Find the grid cell for a bounding box based on its center.
    """
    x, y, _, _ = box
    grid_h, grid_w = grid_size

    cell_x = min(int(x * grid_w), grid_w - 1)
    cell_y = min(int(y * grid_h), grid_h - 1)

    return cell_x, cell_y


def compute_anchor_iou(box, anchors):
    """
    Compute IoU between a bounding box and anchor boxes,
    assuming both are centered at the same point.
    """
    _, _, w, h = box

    box_area = w * h
    ious = []

    for anc_w, anc_h in anchors:
        inter_w = min(w, anc_w)
        inter_h = min(h, anc_h)

        inter_area = max(0.0, inter_w) * max(0.0, inter_h)

        union_area = (
            box_area
            + anc_w * anc_h
            - inter_area
        )

        iou = (
            inter_area / union_area
            if union_area > 0
            else 0.0
        )

        ious.append(iou)

    return np.array(ious, dtype=np.float32)


def assign_objects_to_anchors(objects, anchors):
    """
    Assign objects in one grid cell to distinct anchors.

    Each object is:
        (xywh_box, label)

    Returns:
        list of (object_index, anchor_index)

    For more objects than anchors, only the best
    representable subset is assigned.
    """
    num_objects = len(objects)
    num_anchors = len(anchors)

    if num_objects == 0:
        return []

    max_assigned = min(num_objects, num_anchors)

    # IoU matrix:
    # rows    = objects
    # columns = anchors
    iou_matrix = np.array([
        compute_anchor_iou(obj[0], anchors)
        for obj in objects
    ])

    best_score = -np.inf
    best_assignment = None

    # Choose which objects will actually be represented.
    for object_indices in combinations(range(num_objects), max_assigned):

        # Assign distinct anchors to those objects.
        for anchor_indices in permutations(
            range(num_anchors),
            max_assigned
        ):
            score = sum(
                iou_matrix[obj_idx, anchor_idx]
                for obj_idx, anchor_idx
                in zip(object_indices, anchor_indices)
            )

            if score > best_score:
                best_score = score

                best_assignment = list(
                    zip(object_indices, anchor_indices)
                )

    return best_assignment


def generate_yolo_target(
    boxes,
    labels,
    image_size=IMAGE_SIZE,
    grid_size=GRID_SIZE,
    anchors=ANCHORS,
    num_anchors=NUM_ANCHORS,
    num_classes=NUM_CLASSES,
):
    """
    Generate a collision-aware YOLO-style target tensor with shape:

        (grid_height, grid_width, num_anchors, 5 + num_classes)
    """

    grid_h, grid_w = grid_size

    target = np.zeros(
        (
            grid_h,
            grid_w,
            num_anchors,
            5 + num_classes,
        ),
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # First: convert objects and group them by grid cell
    # ---------------------------------------------------------

    cell_objects = {}

    for box, label in zip(boxes, labels):

        if box is None or len(box) != 4:
            continue

        label = int(label)

        if not (0 <= label < num_classes):
            continue

        # Normalize
        norm_box = normalize_box(
            box,
            image_size
        )

        # Convert to cx, cy, w, h
        xywh_box = xyxy_to_xywh(norm_box)

        # Find cell
        cell_x, cell_y = get_grid_cell(
            xywh_box,
            grid_size
        )

        cell_key = (cell_y, cell_x)

        cell_objects.setdefault(
            cell_key,
            []
        ).append(
            (xywh_box, label)
        )

    # ---------------------------------------------------------
    # Second: assign distinct anchors within each cell
    # ---------------------------------------------------------

    for (cell_y, cell_x), objects in cell_objects.items():

        assignments = assign_objects_to_anchors(
            objects,
            anchors
        )

        for object_idx, anchor_idx in assignments:

            xywh_box, label = objects[object_idx]

            cx, cy, bw, bh = xywh_box

            anchor_w, anchor_h = anchors[anchor_idx]

            # Encode offsets
            tx = (cx * grid_w) - cell_x
            ty = (cy * grid_h) - cell_y

            eps = 1e-6

            tw = np.log(
                max(bw, eps) / anchor_w
            )

            th = np.log(
                max(bh, eps) / anchor_h
            )

            # Fill target
            target[
                cell_y,
                cell_x,
                anchor_idx,
                0
            ] = 1.0

            target[
                cell_y,
                cell_x,
                anchor_idx,
                1
            ] = tx

            target[
                cell_y,
                cell_x,
                anchor_idx,
                2
            ] = ty

            target[
                cell_y,
                cell_x,
                anchor_idx,
                3
            ] = tw

            target[
                cell_y,
                cell_x,
                anchor_idx,
                4
            ] = th

            target[
                cell_y,
                cell_x,
                anchor_idx,
                5 + label
            ] = 1.0

    return tf.convert_to_tensor(target)