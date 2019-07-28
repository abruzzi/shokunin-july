from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from constants import *

styles_encoder = LabelEncoder().fit(LABELS)


def predict(model):
    df_test = pd.read_csv("./synimg/test/data_nostyle.csv")

    test_datagen = ImageDataGenerator(rescale=1./255.)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df_test,
        directory="./",
        x_col="filepath",
        target_size=(32, 64),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False
    )

    test_generator.reset()
    return model.predict_generator(test_generator, verbose=1, steps=312)


def style_of(predication):
    def possibility_of(item):
        return item.get('possibility')

    labeled = [{'label': styles_encoder.inverse_transform([index])[0], 'possibility': possibility} for (index, possibility) in
               enumerate(predication)]

    return max(labeled, key=possibility_of)


def summarize_prediction(predications):
    labels = map(lambda x: style_of(x)['label'], predications)
    zipped = dict(zip(df_test.id, labels))

    return [{"id": k, "style_name": v} for k, v in zipped.items()]


def generate_submission():
    model = load_model(MODEL)
    predication = predict(model)
    submission = pd.DataFrame(summarize_prediction(predications))

    submission.style_name.value_counts().plot.bar()
    submission.to_csv(SUBMISSION, index=False)


generate_submission()
