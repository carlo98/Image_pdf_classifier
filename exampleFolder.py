import time
import os
from Classifier.docClassifier import docClassifier as doC

path_folder = os.path.join(".","image_example")

my_doC = doC(model=os.path.join(".","models","model_1.h5"))

start_time = time.time()
for folder in os.listdir(path_folder):
    tot = 0
    correct = 0
    for my_file in os.listdir(os.path.join(path_folder,folder)):
        tot += 1
        output = my_doC.predict(os.path.join(path_folder,folder,my_file))
        if output == folder:
            correct += 1
        
    print("For folder ",folder," success ",correct/tot*100,"%")
print("Elapsed time: ",time.time()-start_time)
