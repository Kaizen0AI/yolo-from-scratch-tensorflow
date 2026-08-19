# YOLO From Scratch (TensorFlow)

This project implements a simplified YOLO-style object detector from scratch using TensorFlow/Keras, trained on the Pascal VOC 2012 dataset.

The goal was not to reproduce a production YOLO implementation, but to understand the major components of an object detector by implementing the data pipeline, target generation, model, loss, training loop, decoding, NMS, and evaluation myself.

## Features

* Pascal VOC 2012 dataset
* Custom TensorFlow data pipeline
* Pascal VOC XML annotation parsing
* YOLO-style target generation
* Anchor-based bounding-box encoding/decoding
* ResNet-inspired convolutional backbone
* Custom YOLO loss
* Custom `GradientTape` training loop
* Adam optimizer
* Learning-rate scheduling
* Early stopping
* Best-model checkpointing with `tf.train.CheckpointManager`
* Google Colab / Google Drive training workflow
* Bounding-box decoding
* Non-Maximum Suppression
* Detection-level evaluation with TP / FP / FN
* Precision and recall evaluation

## Project Structure

```text
src/
    dataset.py
    loss.py
    model.py
    targets.py
    constants.py

train_model.py
```

## Model

The project started with a simplified **7×7 grid and 2 anchors**.

During development, the detector was changed to:

```text
Grid:          14 × 14
Anchors:       3
Classes:       20
Output:        14 × 14 × 3 × 25
```

The backbone is a small ResNet-inspired CNN built from residual blocks with convolution, batch normalization, ReLU activations, and downsampling.

## Data Pipeline

The pipeline performs:

1. Loading Pascal VOC images and XML annotations
2. Resizing images to `224 × 224`
3. Scaling bounding boxes to the resized image
4. Training-time data augmentation
5. YOLO target generation
6. Padded batching
7. Prefetching with `tf.data.AUTOTUNE`

A major issue discovered during development was that augmentation was initially performed **inside the model**, while the target boxes were generated from the unaugmented image.

This caused a mismatch between the image seen by the model and its target.

The augmentation was therefore moved into the data pipeline so that geometric transformations are applied to both the image and its bounding boxes before YOLO targets are generated.

The final pipeline includes box-aware horizontal flipping, with brightness and contrast augmentation also experimented with later.

## Target Generation

Bounding boxes are converted from:

```text
(xmin, ymin, xmax, ymax)
```

to:

```text
(cx, cy, width, height)
```

and encoded relative to the assigned grid cell and anchor.

The width and height are encoded using:

```text
tw = log(width / anchor_width)
th = log(height / anchor_height)
```

and decoded using the corresponding exponential transformation.

### Target collisions

The original 7×7/2-anchor representation produced a large number of target collisions. Many objects were assigned to the same grid-cell/anchor slot, causing later objects to overwrite earlier targets and, in some cases, produce invalid multi-hot class targets.

The grid was therefore increased to 14×14 and the number of anchors increased to 3.

A collision-aware anchor assignment strategy was also implemented so that multiple objects in the same grid cell could use different anchors when possible.

This reduced the number of unrepresentable objects substantially and eliminated the multi-hot target problem. Some objects can still be unrepresentable when more than three objects occupy the same cell, which is a limitation of the chosen representation.

## Loss

A custom YOLO-style loss was implemented with separate components for:

* Bounding-box localization
* Objectness
* No-objectness
* Classification

The loss uses configurable weights:

```text
lambda_box    = 5.0
lambda_obj    = 1.0
lambda_noobj  = 0.5
lambda_class  = 1.0
```

During development, the positive and negative objectness losses were normalized separately rather than simply summing all positive and negative BCE terms. This reduced the influence of the much larger number of background predictions.

## Training

Training was implemented manually with `tf.GradientTape`.

The training loop includes:

* Forward pass
* Custom loss calculation
* Gradient computation
* Optimizer update
* Training-loss tracking
* Validation-loss tracking
* Learning-rate reduction
* Early stopping
* Best-model checkpointing

`tf.train.CheckpointManager` was used to retain only the best checkpoint.

The project also used Google Colab with Google Drive for longer training runs and GitHub for source-code version control.

## Debugging and Experiments

Several important experiments were used to validate the implementation.

### Overfitting sanity checks

The model was successfully overfit on:

* 1 image
* 8 images
* 16 images

This established that the model, loss, gradient computation, optimizer, and basic target representation were capable of learning the training examples.

### Full-dataset training

The original full-dataset model produced poor detection results despite decreasing training loss.

A major discovery was that the original augmentation implementation changed the image without changing the target boxes.

After moving augmentation into the data pipeline and transforming image/boxes together, training behavior and detection quality improved substantially.

### Evaluation

The project eventually moved from relying primarily on loss and manually inspecting individual predictions to detection-level evaluation using:

* True positives
* False positives
* False negatives
* Precision
* Recall

This revealed that **objectness was a major contributor to false positives**, even after classification and localization had improved.

## Inference

A prediction decoder was implemented to convert raw model outputs into:

```text
bounding boxes
objectness probability
class probabilities
class prediction
confidence
```

with:

```text
confidence = objectness × class probability
```

Non-Maximum Suppression was also implemented using TensorFlow's `tf.image.non_max_suppression`.

The NMS stage was later identified as requiring class-aware handling for a more complete detector implementation.

## Current Status

The project is considered **complete as a learning project**, but it was not taken to production-level detector performance.

### Completed

* [x] Pascal VOC dataset pipeline
* [x] Annotation parsing
* [x] Image preprocessing
* [x] Box-aware data augmentation
* [x] YOLO target generation
* [x] Anchor assignment
* [x] Collision-aware target assignment
* [x] ResNet-inspired backbone
* [x] Custom YOLO loss
* [x] Custom `GradientTape` training loop
* [x] Learning-rate scheduling
* [x] Early stopping
* [x] Checkpointing
* [x] Google Colab training setup
* [x] Prediction decoder
* [x] Non-Maximum Suppression
* [x] TP / FP / FN evaluation
* [x] Precision / recall evaluation

### Not fully completed

* [ ] Production-quality mAP evaluation
* [ ] Further optimization of objectness behavior
* [ ] Final model tuning for strong validation performance

## What I Learned

### 1. Keep an experiment log from the beginning

The project became much harder to reason about because changes and the reasons behind them were not recorded during development.

For a longer ML project, every meaningful experiment should record:

```text
Change
Reason
Configuration
Result
Conclusion
```

This makes it possible to understand why the current version looks the way it does without relying on memory.

### 2. Build the complete evaluation backbone before deep debugging

A major mistake was starting detailed diagnosis before having a complete end-to-end evaluation system.

A better workflow is:

```text
Data
→ preprocessing
→ model
→ loss
→ training
→ inference
→ evaluation metrics
→ diagnosis
```

For an object detector, having TP / FP / FN, precision, recall, IoU, and eventually mAP available early makes later debugging much more systematic.

### 3. Small sanity checks are extremely valuable

Before full training, verify that the model can overfit a tiny dataset.

If a model cannot overfit a few examples, the problem is probably fundamental.

If it can, the investigation can move toward optimization, generalization, or dataset-scale issues.

### 4. Detection augmentation must transform the targets too

Image augmentation cannot be treated like ordinary image-classification augmentation when bounding boxes are involved.

If the image geometry changes, the boxes must change with it before targets are generated.

### 5. Building everything from scratch has diminishing returns

Implementing the entire detector from scratch was valuable for understanding how the pieces interact, but it also made the project substantially larger and harder to debug.

For future projects, it makes more sense to build selected components from scratch while using established components where appropriate.

## Final Takeaway

This project started as an attempt to build a simplified YOLO detector from scratch and became a much larger exercise in machine-learning engineering and debugging.

The final detector is not a highly optimized object detector, but the project provided practical experience with:

* custom data pipelines
* object-detection target representations
* anchor boxes
* custom losses
* gradient-based training
* augmentation
* checkpointing
* inference
* NMS
* precision/recall
* debugging complex ML systems

The project is intentionally being stopped here rather than continuously expanded. The main goal was learning, and the project achieved that goal.
