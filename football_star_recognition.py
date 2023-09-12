import cv2
import matplotlib
import numpy as np

choice=0
while choice!=3:
    print("Football Player Face Recognition")
    print("1. Train and Test Model")
    print("2. Predict")
    print("3. Exit")
    choice = int(input(">> "))

    if choice==1:
        #Train and test model here
        print("Train and test")
    elif choice==2:
        print("Predict")
    elif choice==3:
        print("Program terminated successfully")
        break
      


