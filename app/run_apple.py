import models.AppleModel as apple_model
import cv2

print("Started run corn")

img_path = f'data/img.jpg'
img = cv2.imread(img_path)

AppleModel = apple_model.AppleModel(weights='weights/apple_disease.h5')

preds = AppleModel.predict_image(img)

with open('final_pred.txt', 'w') as file:
    file.write(str(preds))