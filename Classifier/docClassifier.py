'''
Created on Oct 21, 2019

@author: Carlo Cena
'''
import tensorflow as tf #Not supported for windows 32bit
from keras.preprocessing import image #It needs tensorflow
import pytesseract
import numpy as np
import os
import imutils
import cv2
from wand.image import Image as wi
from wand.api import library
import ctypes
# Tell wand about C-API method
library.MagickNextImage.argtypes = [ctypes.c_void_p]
library.MagickNextImage.restype = ctypes.c_int

class docClassifier(object):
    '''
     
    '''

    def __init__(self, model):
        '''
        Constructor, it defines:
        - the words that should define each type of document
        - configuration for pytesseract
        - CNN model
        - angles of rotation
        WARNING: IMPORTANT PARAMETERS, MODIFYING THEM CAN BREAK THE PROGRAM
        @param model: path of the model (example: "./model_1.h5")
        '''
        self.config = ('-l eng --oem 1 --psm 3')
        self.PRC = ["REVISIONE", "CONTROLLO", "CHECK", "PLAN", "PIANO", "MANUFACTORING"]
        self.FR = ["FAULT", "REPORT", "AULT", "FAUL", "EPORT", "REPOR"]
        self.CF = ["DISTINTA", "MATRICOLE", "ISTINTA", "DISTINT", "ATRICOLE", "MATRICOL"]
        self.model = tf.keras.models.load_model(model)
        self.angles = [0, 180, 90, -90]
        
    def predict(self, input_file):
        '''
        @param input_file: single pdf file to be classified
        @return: string representing type of document (PRC, FR, CF or ST)
        '''
        input_file = input_file[:-4]
    
        #Creating image for CNN
        with wi(filename=input_file+".pdf", resolution=150) as pdf:
            lenght = len(pdf.sequence)
            pdfimage = pdf.convert("jpg")
            img = wi(image=pdfimage)
            img.save(filename=input_file+".jpg")
    
        if lenght > 1:
            img = cv2.imread(input_file+"-0.jpg")
        else:
            img = cv2.imread(input_file+".jpg")
        x = cv2.resize(img, (800, 800))
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
    
        #Rotating image by predicted angle (using CNN)
        angle = self.angles[np.argmax(self.model.predict([x]))]
        img = imutils.rotate(img, angle)
    
        #Extracting text from image using tesseract API
        text = pytesseract.image_to_string(img, config=self.config)
       
        flag = 0
        cont_prc = 0  #Used to lower the probability of incorrect classifications
        cont_fr = 0
        cont_cf = 0
        already_found = []
        to_be_returned = ""
        for word in text.split(' '):
            type_int = 0  #Type_int keeps track of document type
            if flag == 0:
                for type_doc in [self.PRC, self.FR, self.CF]:
                    for word_dict in type_doc:
                        if word.find(word_dict) > -1 and flag == 0 and word_dict not in already_found:
                
                            if type_int == 0:
                                cont_prc += 1
                            elif type_int == 1:
                                cont_fr += 1
                            elif type_int == 2:
                                cont_cf += 1
                            
                            if cont_prc > 1:
                                to_be_returned = "PRC"
                                flag = 1
                                break
                            elif cont_fr > 2:
                                to_be_returned = "FR"
                                flag = 1
                                break
                            elif cont_cf > 2:
                                to_be_returned = "CF"
                                flag = 1
                                break
                            
                            already_found.append(word_dict)
                        
                    type_int += 1
                
                if flag == 1:
                    break
        if flag == 0:  #Checking for ST folder
            to_be_returned = "ST"

        #Removing images
        if lenght > 1:
            for i in range(lenght):
                os.remove(input_file+"-"+str(i)+".jpg")
        else:
            os.remove(input_file+".jpg")
        
        return to_be_returned