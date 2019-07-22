import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from PIL import Image
import dataset

# class Classifier:        
#     def train(self, name, dataset_provider):
#          # Init
#         classifier = Sequential()

#         # Step 1 convolution
#         classifier.add(Convolution2D(
#             32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

#         # Step 2 pooling
#         classifier.add(MaxPooling2D(pool_size=(2, 2)))

#         # Step 3 flattening
#         classifier.add(Flatten())

#         # Step 4 Full connection
#         classifier.add(Dense(128, activation='relu'))
#         classifier.add(Dense(1, activation='sigmoid'))

#         classifier.compile(
#             optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#         classifier.fit_generator(
#             dataset_provider.training(),
#             steps_per_epoch=8000,
#             epochs=10,
#             validation_data=dataset_provider.test(),
#             validation_steps=800
#         )
#         model_path = "{}-classifier.h5" % name
#         classifier.save(model_path)
        
#         return model_path

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img-dir', type=str, default='dataset')
parser.add_argument('--model-input', type=str, default=None,
                    help='if set, load weights from this model file')

opts = parser.parse_args()
print(opts)
test_data = dataset.train(opts.img_dir)
train_data = dataset.test(opts.img_dir)