import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(0)
from tensorflow import keras
import numpy as np
np.random.seed(0)
import itertools
# from tensorflow.keras.layers.experimental.preprocessing import Rescaling
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam

import cv2


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class AppleModel(object):
    def __init__(self, input_shape=(256, 256, 3), weights=None):
        self.input_shape = input_shape
        self.weights = weights 
        self.model = self.build_model()


    def build_model(self):
        # corn model final
        model = keras.Sequential()

        model.add(keras.layers.Conv2D(32,(3,3),activation="relu",padding="same",input_shape=(256,256,3)))
        model.add(keras.layers.Conv2D(32,(3,3),activation="relu",padding="same"))
        model.add(keras.layers.MaxPooling2D(3,3))

        model.add(keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"))
        model.add(keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"))
        model.add(keras.layers.MaxPooling2D(3,3))

        model.add(keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"))
        model.add(keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"))
        model.add(keras.layers.MaxPooling2D(3,3))

        model.add(keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"))
        model.add(keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"))

        model.add(keras.layers.Conv2D(512,(5,5),activation="relu",padding="same"))
        model.add(keras.layers.Conv2D(512,(5,5),activation="relu",padding="same"))

        model.add(keras.layers.Flatten())

        model.add(keras.layers.Dense(1568,activation="relu"))
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.Dense(3,activation="softmax"))

        opt = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=opt,loss="sparse_categorical_crossentropy",metrics=['accuracy'])
        
        if self.weights:
            model.load_weights(self.weights)
        return model

    def predict_image(self, image):
        image = cv2.resize(image, (256, 256))
        image = image.reshape((1, 256, 256, 3))
        prediction = np.argmax(self.model.predict(image))
        return prediction
