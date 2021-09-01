import sys
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
from tensorflow import keras
from utils import class_activation_mapping as cam

if __name__ == "__main__":
    img = np.resize(plt.imread(r"data\car_ims\000001.jpg"), new_shape=(128, 128, 3))
    model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    x = model.output
    x = keras.layers.GlobalAveragePooling2D()(x)

    # and a fully connected output/classification layer
    predictions = keras.layers.Dense(196, activation="softmax")(x)

    # create the full network so we can train on it
    model = keras.models.Model(inputs=model.input, outputs=predictions)

    cam.compute_and_plot_CAM(model, img, None, "top_activation", figsize=(10, 10))