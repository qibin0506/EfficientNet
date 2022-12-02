from tensorflow import keras
from keras import layers, models

inputs = layers.Input(shape=[100, 100, 3])
layer = layers.Conv2D(
    filters=32,
    strides=1,
    kernel_size=1,
    padding='same'
)(inputs)

layer = layers.GlobalAveragePooling2D()(layer)
layer = layers.Dense(2)(layer)

model = models.Model(inputs=inputs, outputs=layer)
model.summary()
