# YOLO From Scratch (TensorFlow)

This project implements a simplified YOLO-style object detector from scratch using TensorFlow/Keras.

## Features

- Pascal VOC 2012 dataset
- Custom data pipeline
- YOLO target generation
- ResNet-inspired backbone
- Two-anchor detection head
- Custom YOLO loss
- Custom GradientTape training loop
- Early stopping
- Learning rate scheduling
- Model checkpointing

## Project Structure

```
src/
    dataset.py
    loss.py
    model.py
    targets.py
    constants.py

train_model.py
```

## Current Status

- [x] Dataset pipeline
- [x] Target generation
- [x] Model implementation
- [x] Loss function
- [x] Training loop
- [x] Overfit sanity check
- [ ] Full Pascal VOC training
- [ ] Prediction decoder
- [ ] Non-Maximum Suppression
- [ ] mAP evaluation

## Future Work

- Decode YOLO predictions
- Non-Maximum Suppression
- Inference pipeline
- Evaluation metrics