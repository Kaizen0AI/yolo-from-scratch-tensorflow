IMAGE_SIZE = (224,224)
BATCH_SIZE = 16      # or 8 if Colab RAM/GPU becomes an issue
GRID_SIZE = (14, 14)
NUM_CLASSES = 20
NUM_ANCHORS = 3
ANCHORS = [
    (0.1295, 0.1652),
    (0.4286, 0.5446),
    (1.3036, 1.2098)
    ]