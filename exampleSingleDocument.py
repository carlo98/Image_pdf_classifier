from Classifier.docClassifier import docClassifier as doC
import os
#in windows \\
#download python3.6;download pip; forse terminale python3.6
#python3.6/python exampleMain.py 
#pip install tensorflow keras pytesseract numpy os imutils opencv-python ctypes 

my_doC = doC(model=os.path.join(".","models","model_1.h5"))

folder = os.path.join(".","image_example","PRC")

for my_file in os.listdir(folder):
    print(my_doC.predict(input_file=os.path.join(folder,my_file)))
