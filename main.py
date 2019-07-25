import tensorflow as tf
from tensorflow import keras

import pandas as pd

LABELS = ['Luanda', 'HongKong', 'Zurich', 'Singapore', 'Geneva',
          'Beijing', 'Seoul', 'Sydney', 'Melbourne', 'Brisbane']


def create_model():
    """
    define the model
    """

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                                  activation='relu', input_shape=(64, 32, 3)))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(len(LABELS), activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer="adam", metrics=["accuracy"])

    return model


def create_train_data_sets():
    """
    define the generator for loading training data sets
    """

    df_train = pd.read_csv("./synimg/train/data.csv")

    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=12,
        brightness_range=[0.5, 1.0],
        zoom_range=0.1,
        width_shift_range=0.2,
        height_shift_range=0.2,
        validation_split=0.25)

    train_generator = datagen.flow_from_dataframe(
        dataframe=df_train,
        x_col="filepath",
        y_col="style_name",
        subset="training",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(64, 32))

    validation_generator = datagen.flow_from_dataframe(
        dataframe=df_train,
        x_col="filepath",
        y_col="style_name",
        batch_size=32,
        seed=42,
        shuffle=True,
        subset="validation",
        class_mode="categorical",
        target_size=(64, 32))

    return (train_generator, validation_generator)


def train(filename):
    """
    do the actual training processing
    """

    train_generator, validation_generator = create_train_data_sets()

    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

    model = create_model()
    # fit
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=STEP_SIZE_VALID)

    # save
    model.save(filename)


train("shokunin-july-32.h5")
