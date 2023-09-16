import cv2
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt

train=[]
test=[]

'''
#Read folders here
DATA_DIR='Dataset'
labels=[folder for folder in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,folder))]

for label in labels:
    label_path=os.path.join(DATA_DIR,label)

    #Count the number of images in each label
    images=len([file for file in os.listdir(label_path) if file.lower().endswith(('.jpg','jpeg','.png'))])
    print(f'Label: {label} {images} images')
'''

choice=0
while choice!=3:
    print("Football Player Face Recognition")
    print("1. Train and Test Model")
    print("2. Predict")
    print("3. Exit")
    choice = int(input(">> "))

    if choice==1:
        #Train and test model here
        face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        sample_image='C:/Users/admin/OneDrive - Bina Nusantara/Binus/Semester 5/Computer Vision/LAB/Project/Dataset/test/cristiano_ronaldo/16.jpg'
        processed_image=cv2.imread(sample_image)
        gray_image=cv2.cvtColor(processed_image,cv2.COLOR_BGR2GRAY)
        faces=face_classifier.detectMultiScale(gray_image)
        print(faces)
        for face in faces:
            x,y,w,h=face
            cv2.rectangle(processed_image,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("Test",processed_image)
        cv2.waitKey(0)
    elif choice==2:
        print("Predict")
    elif choice==3:
        print("Program terminated successfully")
        break
      


