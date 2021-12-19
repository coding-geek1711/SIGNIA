import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import cv2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class LoadModel(object):
    def __init__(self, input_shape=(256, 256, 3), weights = None):
        self.input_shape = input_shape
        self.weights = weights
        self.model = self.build_model()

    def build_model(self):
        resnet50 = ResNet50(include_top=False, input_shape=(256, 256, 3), weights=None)
        x = Flatten()(resnet50.layers[-1].output)
        x = Dense(units=16, activation='relu')(x)
        x = Dense(units=32, activation='relu')(x)
        last = Dense(units=2, activation='softmax')(x)

        model = Model(resnet50.inputs, last)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if self.weights:
            model.load_weights(self.weights)

        return model

    def predict_image(self, image):
        image = cv2.resize(image, (256, 256))
        image = image.reshape((1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        prediction = np.argmax(self.model.predict(image))
        return str(prediction)