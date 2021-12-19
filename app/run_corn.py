import models.CornModel as corn_model
import cv2

print("Started run corn")

img_path = f'data/img.jpg'
img = cv2.imread(img_path)

CornModel = corn_model.CornModel(weights='weights/corn_disease.h5')

preds = CornModel.predict_image(img)

with open('final_pred.txt', 'w') as file:
    file.write(str(preds))