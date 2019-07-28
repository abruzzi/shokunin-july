import os

import numpy as np
import pandas as pd

from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.optimizers import RMSprop

from keras import layers as layers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.losses import categorical_crossentropy
from efficientnet.keras import EfficientNetB3

from constants import *


def create_train_data_sets():
    """
    define the generator for loading training data sets
    """

    df_train = pd.read_csv("./synimg/train/data.csv")
    df_train = df_train.sample(frac=1.0)

    train_data = df_train[:SPLIT_AT]
    validation_data = df_train[SPLIT_AT:]

    train_gen = image.ImageDataGenerator(
        rescale=1./255.,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2)

    validation_gen = image.ImageDataGenerator(rescale=1./255.)

    train_generator = train_gen.flow_from_dataframe(
        dataframe=train_data,
        directory="./",
        x_col="filepath",
        y_col="style_name",
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode="categorical",
        target_size=(32, 64))

    validation_generator = validation_gen.flow_from_dataframe(
        dataframe=validation_data,
        directory="./",
        x_col="filepath",
        y_col="style_name",
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode="categorical",
        target_size=(32, 64))

    return (train_generator, validation_generator)


def build_model():
    input_shape = (32, 64, 3)

    efficient_net = EfficientNetB3(
        weights='imagenet', include_top=False, input_shape=input_shape)
    efficient_net.trainable = False

    x = efficient_net.output
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(len(LABELS), activation="softmax")(x)

    model = Model(inputs=efficient_net.input, outputs=predictions)

    model.compile(optimizer=RMSprop(lr=0.0001, decay=1e-6),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])

    return model


def training():
    model = build_model()

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2,
                                  verbose=1, mode='max', min_lr=0.000001)

    checkpoint = ModelCheckpoint(MODEL, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    train_generator, validation_generator = create_train_data_sets()

    steps_per_epoch_train = train_generator.n//train_generator.batch_size
    steps_per_epoch_validation = validation_generator.n//validation_generator.batch_size

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch_train,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=steps_per_epoch_validation,
        callbacks=[checkpoint, reduce_lr])

    return history, model


history, model = training()
