from flask import Flask, request, send_file, jsonify
from PIL import Image
import os 
import numpy as np
import models.LoadModel_1 as load_model
import models.CornModel as corn_model
import tensorflow as tf


from numba import cuda

device = cuda.get_current_device()

app = Flask(__name__)

classifier_weights_path = ''

Classifier_Model = load_model.LoadModel((256, 256, 3))

@app.route("/")
def home():
    return "home"

@app.route("/img_in", methods=['POST'])
def recieve_img():
    file = request.files['image']
    img = np.array(Image.open(file.stream))
    
    prediction_image_type = Classifier_Model.predict_image(img)

    print("+++++++++++++++++++++++++++++++++++++++++++++++++")
    print(prediction_image_type)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++")

    tf.keras.backend.clear_session()
    
    if prediction_image_type == 1:
        
        CornModel = corn_model.CornModel()
        prediction = CornModel.predict_image(img)
        return prediction        
    else:
        # do apple
        pass

if __name__ == '__main__':
    app.run(debug=True)