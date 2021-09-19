import pandas as pd
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
# import os
# import shutil
# import cv2 as cv
import matplotlib.pyplot as plt


print(tf.test.is_built_with_cuda())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
    
mirrored_strategy = tf.distribute.MirroredStrategy()

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])


# Reference for preprocessing: https://keras.io/examples/vision/learnable_resizer/
BATCH_SIZE = 16
INP_SIZE = (1080, 1920)
TARGET_SIZE = (360, 640) # 1/3 size
# TARGET_SIZE = (540, 960) # 1/2 size
EPOCHS = 150

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'DataDirectory/',
    validation_split=0.2,
    subset="training",
    labels='inferred',
    image_size=(TARGET_SIZE[0], TARGET_SIZE[1]),
    interpolation='bilinear',
    shuffle=True,
    batch_size=BATCH_SIZE,
    seed=2)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'DataDirectory/',
    validation_split=0.2,
    subset="validation",
    labels='inferred',
    image_size=(TARGET_SIZE[0], TARGET_SIZE[1]),
    interpolation='bilinear',
    shuffle=True,
    batch_size=BATCH_SIZE,
    seed=2)

with mirrored_strategy.scope():

    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)),
        layers.Conv2D(16, 3, padding='same'),
    #     layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.65),
        layers.Conv2D(32, 3, padding='same'),
    #     layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.65),
        layers.Conv2D(64, 3, padding='same'),
    #     layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.65),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.75),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=EPOCHS
    )
    
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.savefig(f'graph_acc_{EPOCHS}.png')
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig(f'graph_loss_{EPOCHS}.png')
        
