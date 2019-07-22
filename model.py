from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# Init
classifier = Sequential()

# Step 1 convolution
classifier.add(Convolution2D(
    32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Step 2 pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 flattening
classifier.add(Flatten())

# Step 4 Full connection
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.compile(loss=keras.losses.sparse_categorical_crossentropy,
# optimizer="sgd", metrics=["accuracy"])

#------------data.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

#------------data.py

#-------------train.py
# from IPython.display import display
from PIL import Image

classifier.fit_generator(
    training_set,
    steps_per_epoch=8000,
    epochs=10,
    validation_data=test_set,
    validation_steps=800
)
#-------------train.py

classifier.save("catdog_classifier.h5")