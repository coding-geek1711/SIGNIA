from flask import Flask, request, send_file, jsonify
from PIL import Image
import os 

from models import LoadModel_1

app = Flask(__name__)

classifier_weights_path = ''


Classifier_Model = LoadModel_1((256, 256, 3), classifier_weights_path)

@app.route("/")
def home():
    return "home"

@app.route("/img_in", methods=['POST'])
def recieve_img():
    file = request.files['image']
    img = Image.open(file)
    prediction = Classifier_Model.predict_image(img)
    return prediction

