import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import InceptionResNetV2

import pandas as pd

LABELS = ['Luanda', 'HongKong', 'Zurich', 'Singapore', 'Geneva',
          'Beijing', 'Seoul', 'Sydney', 'Melbourne', 'Brisbane']


def create_model():
    """
    define the model
    """

    conv_base = InceptionResNetV2(
        weights="imagenet", include_top=False, input_shape=(256, 128, 3))

    model = keras.models.Sequential()
    model.add(conv_base)

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(len(LABELS), activation='softmax'))

    model.summary()
    conv_base.trainable = False

    opt = keras.optimizers.RMSprop(lr=2e-5)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt, metrics=["accuracy"])

    return model


def create_train_data_sets():
    """
    define the generator for loading training data sets
    """

    df_train = pd.read_csv("./synimg/train/data.csv")

    train_gen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255.,
        validation_split=0.25)

    train_generator = train_gen.flow_from_dataframe(
        dataframe=df_train,
        x_col="filepath",
        y_col="style_name",
        subset="training",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(256, 128))

    validation_generator = train_gen.flow_from_dataframe(
        dataframe=df_train,
        x_col="filepath",
        y_col="style_name",
        batch_size=32,
        seed=42,
        shuffle=True,
        subset="validation",
        class_mode="categorical",
        target_size=(256, 128))

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


train("shokunin-july-incption.h5")
