import tensorflow as tf
import numpy as np
import cv2

def predict_tb(image_path: str, model_path="models/trained_model.h5"):
    model = tf.keras.models.load_model(model_path)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]

    return "Tuberculosis Detected" if prediction > 0.5 else "Normal"
