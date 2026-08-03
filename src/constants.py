IMAGE_SIZE = (224,224)
BATCH_SIZE = 16      # or 8 if Colab RAM/GPU becomes an issue
GRID_SIZE = (7, 7)
NUM_CLASSES = 20
NUM_ANCHORS = 2
ANCHORS = [
    (0.10, 0.10),
    (0.20, 0.15),
]