import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_train = pd.read_csv("./synimg/train/data.csv")

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                             shear_range=0.2,
                                                             zoom_range=0.2,
                                                             horizontal_flip=True)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train[:80000],
    x_col="filepath",
    y_col="style_name",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="other",
    target_size=(64, 32))

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)

validation_generator = test_datagen.flow_from_dataframe(
    dataframe=df_train[80000:],
    x_col="filepath",
    y_col="style_name",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="other",
    target_size=(64, 32))

df_test = pd.read_csv("./synimg/test/data_nostyle.csv")

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    x_col="filepath",
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(64, 32))

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(
    32, (3, 3), padding='same', input_shape=(64, 32, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(32, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='sigmoid'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer="sgd", metrics=["accuracy"])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=800)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

model.save("shokunin-july.h5")
