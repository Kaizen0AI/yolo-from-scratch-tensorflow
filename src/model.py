import tensorflow as tf
from src.constants import GRID_SIZE, IMAGE_SIZE, NUM_CLASSES, NUM_ANCHORS

def res_block(x, filters, downsample= False):
    skip= x
    stride = 2 if downsample else 1
        
    if downsample or x.shape[-1] != filters:
        skip= tf.keras.layers.Conv2D(filters, (1,1), strides= stride, padding='valid', use_bias= False)(skip)
        skip= tf.keras.layers.BatchNormalization()(skip)
        
    x= tf.keras.layers.Conv2D(filters, (3,3), strides=stride, padding= 'same', use_bias= False)(x)
    x= tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.ReLU()(x)
    
    x= tf.keras.layers.Conv2D(filters, (3,3), strides=1, padding= 'same', use_bias= False)(x)
    x= tf.keras.layers.BatchNormalization()(x)     
    
    x= tf.keras.layers.Add()([x, skip])
    x= tf.keras.layers.ReLU()(x)

    return x

def stem_block(x):
    x= tf.keras.layers.Conv2D(32, (3,3), strides= 2, padding= 'same', use_bias= False)(x) 
    x= tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.ReLU()(x)
    x= tf.keras.layers.MaxPooling2D((3,3), strides= 2)(x)
    
    return x

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])

OUTPUT_CHANNELS = NUM_ANCHORS * (NUM_CLASSES + 5)

def build_model(input_shape = IMAGE_SIZE + (3,)):
    inputs= tf.keras.Input(shape= input_shape)

    x= data_augmentation(inputs)
    
    x= stem_block(x)
    
    x= res_block(x, 32)
    x= res_block(x, 64, downsample= True)
    
    x= res_block(x, 64)
    x= res_block(x, 128, downsample= True)
    
    x= res_block(x, 128)
    x= res_block(x, 256, downsample= True)

    #x= res_block(x, 256)
    #x= res_block(x, 512, downsample= True) #use this during training
    
    x= tf.keras.layers.Conv2D(512, (3,3), strides= 1, padding= 'same', use_bias= False)(x) # increaese filters to 512 during traning
    x= tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.ReLU()(x)

    x= tf.keras.layers.Conv2D(OUTPUT_CHANNELS, (1,1), strides= 1, padding= 'same')(x)
    outputs = tf.keras.layers.Reshape(
    (
        GRID_SIZE[0],
        GRID_SIZE[1],
        NUM_ANCHORS,
        NUM_CLASSES + 5))(x)
    
    
    model= tf.keras.Model(inputs, outputs)
    
    return model