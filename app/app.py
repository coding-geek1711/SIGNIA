import os 

os.system("python run_classifier.py")

with open("file.txt") as file:
    pred_classify = file.read()

os.system("nvidia-smi")


if pred_classify == 1:
    os.system("python run_corn.py")
else:
    os.system("python run_apple.py")
