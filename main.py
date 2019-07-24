import tensorflow as tf
from tensorflow import keras

import pandas as pd

labels = ['Luanda', 'HongKong', 'Zurich', 'Singapore', 'Geneva',
          'Beijing', 'Seoul', 'Sydney', 'Melbourne', 'Brisbane']

# define the model
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                              activation='relu', input_shape=(64, 32, 3)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(labels), activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer="adam", metrics=["accuracy"])

# prepare data
df_train = pd.read_csv("./synimg/train/data.csv")

datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255., validation_split=0.2)

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

df_test = pd.read_csv("./synimg/test/data_nostyle.csv")
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    x_col="filepath",
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(64, 32))

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

# fit
history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VALID)

print(history.history)

# save
model.save("shokunin-july.h5")
