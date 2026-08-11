import os
import tensorflow as tf
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from src.constants import BATCH_SIZE, GRID_SIZE, IMAGE_SIZE, NUM_CLASSES

VOC_ROOT = os.getenv(
    "VOC_ROOT",
    "E:/datasets/pascal-voc-2012/VOC2012_train_val"
)
IMAGE_DIR = os.path.join(VOC_ROOT, "JPEGImages")
ANNOTATION_DIR = os.path.join(VOC_ROOT, "Annotations")
TRAIN_FILE = os.path.join(VOC_ROOT, "ImageSets", "Main", "train.txt")
VAL_FILE = os.path.join(VOC_ROOT, "ImageSets", "Main", "val.txt")

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
CLASS_TO_ID = {
    name: i
    for i, name in enumerate(VOC_CLASSES)
}

ID_TO_CLASS = {i: name for i, name in enumerate(VOC_CLASSES)}

with open(TRAIN_FILE, "r") as f:
    train_ids = f.read().splitlines()

with open(VAL_FILE, "r") as f:
    val_ids = f.read().splitlines()


def load_image(img_id):
    img_path = os.path.join(IMAGE_DIR, img_id + ".jpg")
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def parse_annotation(img_id):
    ann_path = os.path.join(ANNOTATION_DIR, img_id + ".xml")
    tree = ET.parse(ann_path)
    root = tree.getroot()

    boxes, labels = [], []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_TO_ID[cls])

    return tf.convert_to_tensor(boxes, dtype=tf.float32), labels


def load_example(img_id):
    def _load_example_py(img_id_py):
        img_id_py = img_id_py.numpy().decode("utf-8")

        img = load_image(img_id_py)
        boxes, labels = parse_annotation(img_id_py)

        return img, boxes, labels

    img, boxes, labels = tf.py_function(
        func=_load_example_py,
        inp=[img_id],
        Tout=[tf.uint8, tf.float32, tf.int64]
    )
    img.set_shape([None, None, 3])
    boxes.set_shape([None, 4])
    labels.set_shape([None])
    return {"image": img, "boxes": boxes, "labels": labels}


def preprocess(dataset):
    def _preprocess(example):
        img = example["image"]
        boxes = example["boxes"]
        labels = example["labels"]

        img = tf.image.convert_image_dtype(img, tf.float32)

        original_height = tf.cast(tf.shape(img)[0], tf.float32)
        original_width = tf.cast(tf.shape(img)[1], tf.float32)
        img = tf.image.resize(img, IMAGE_SIZE, method = 'bilinear')

        scale = tf.stack([
            IMAGE_SIZE[1] / original_width,
            IMAGE_SIZE[0] / original_height,
            IMAGE_SIZE[1] / original_width,
            IMAGE_SIZE[0] / original_height,
        ])

        boxes = boxes * scale

        return {"image": img, "boxes": boxes, "labels": labels}

    return dataset.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)


def add_targets(dataset, target_fn, target_shape, target_kwargs=None):
    def _add_targets(example):
        boxes = example["boxes"]
        labels = example["labels"]

        def _generate_targets_py(boxes_py, labels_py):
            boxes_py = boxes_py.numpy()
            labels_py = labels_py.numpy()
            return target_fn(boxes_py, labels_py, **(target_kwargs or {})).numpy()

        targets = tf.py_function(
            func=_generate_targets_py,
            inp=[boxes, labels],
            Tout=[tf.float32],
        )[0]
        targets.set_shape(target_shape)

        return {
            "image": example["image"],
            "boxes": boxes,
            "labels": labels,
            "targets": targets,
        }

    return dataset.map(_add_targets, num_parallel_calls=tf.data.AUTOTUNE)


def build_dataset(batch_size=BATCH_SIZE, include_targets=False, target_fn=None, target_shape=None, target_kwargs=None, split="train", shuffle=True):
    if split == "train":
        dataset = tf.data.Dataset.from_tensor_slices(train_ids)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(train_ids), reshuffle_each_iteration=True)
    elif split == "val":
        dataset = tf.data.Dataset.from_tensor_slices(val_ids)
    else:
        raise ValueError("split must be 'train' or 'val'")
    dataset = dataset.map(lambda x: load_example(x), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = preprocess(dataset)

    if include_targets:
        if target_fn is None or target_shape is None:
            raise ValueError("target_fn and target_shape are required when include_targets=True")
        dataset = add_targets(dataset, target_fn, target_shape, target_kwargs)

    padded_shapes = {
        "image": list(IMAGE_SIZE) + [3],
        "boxes": [None, 4],
        "labels": [None],
    }
    if include_targets:
        padded_shapes["targets"] = target_shape

    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def build_yolo_dataset(batch_size=BATCH_SIZE, split="train", shuffle=True):
    from src.targets import ANCHORS, generate_yolo_target

    return build_dataset(
        batch_size=batch_size,
        split=split,
        shuffle=shuffle,
        include_targets=True,
        target_fn=generate_yolo_target,
        target_shape=(GRID_SIZE[0], GRID_SIZE[1], len(ANCHORS), 5 + NUM_CLASSES),
        target_kwargs={
            "image_size": IMAGE_SIZE,
            "grid_size": GRID_SIZE,
            "anchors": ANCHORS,
        },
    )


# Generic dataset for non-YOLO algorithms.
#dataset = build_dataset()

# Optional YOLO-specific dataset.
#yolo_dataset = build_yolo_dataset()


def visualize_example(dataset):
    batch = next(iter(dataset))

    images = batch["image"]
    boxes_batch = batch["boxes"]
    labels_batch = batch["labels"]

    image = images[0].numpy()
    boxes = boxes_batch[0].numpy()
    labels = labels_batch[0].numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    ax = plt.gca()

    for box, label in zip(boxes, labels):
        if tf.reduce_all(box == 0):
            continue

        xmin, ymin, xmax, ymax = box

        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            color="red",
            linewidth=2
        )

        ax.add_patch(rect)

        ax.text(
            xmin,
            ymin - 5,
            ID_TO_CLASS[int(label)],
            color="yellow",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.5)
        )

    plt.axis("off")
    plt.show()