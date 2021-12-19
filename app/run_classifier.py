import cv2
import models.LoadModel as load_model

img_path = f'data/img.jpg'
img = cv2.imread(img_path)


Classify = load_model.LoadModel()
preds = Classify.predict_image(img)

with open('file.txt', 'w') as file:
    file.write(preds)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")