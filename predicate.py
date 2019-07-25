from tensorflow import keras

import pandas as pd
from keras.preprocessing import image
import numpy as np
from tqdm import tqdm

HEADER = "id,style_name"
RESULT_FILE = "submission.csv"

model = keras.models.load_model('models/shokunin-july-32.h5')


def load_test_data():
    test_input_csv = pd.read_csv("./synimg/test/data_nostyle.csv")
    test_image = []
    for i in tqdm(range(test_input_csv.shape[0])):
        img = image.load_img(test_input_csv['filepath'][i], target_size=(64, 32, 3), grayscale=False)
        img = image.img_to_array(img)
        img = img / 255
        test_image.append(img)
    return np.expand_dims(np.array(test_image), 1)


def style_of(img):
    labels = ['Luanda', 'HongKong', 'Zurich', 'Singapore', 'Geneva',
              'Beijing', 'Seoul', 'Sydney', 'Melbourne', 'Brisbane']

    def possibility_of(item):
        return item.get('possibility')

    prediction = model.predict(img)
    predict_results = np.round(prediction[0], 2)
    labeled = [{'label': labels[index], 'possibility': possibility} for (index, possibility) in
               enumerate(predict_results)]
    return max(labeled, key=possibility_of)


test_input = load_test_data()

with open(RESULT_FILE, "w") as f:
    f.write(HEADER)
    for (file_id, img) in enumerate(test_input, start=9000000):
        style = style_of(img)['label']
        f.write("\n")
        f.write(f"{file_id},{style}")