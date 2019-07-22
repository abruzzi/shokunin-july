from tensorflow.keras.preprocessing.image import ImageDataGenerator

def test_data(dir):
    gen = ImageDataGenerator(rescale=1./255)
    return gen.flow_from_directory(
            "{}}/test_set" % dir,
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary'
        )

def train_data(dir):
    gen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
    return gen.flow_from_directory(
        "%s/training_set" % dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )