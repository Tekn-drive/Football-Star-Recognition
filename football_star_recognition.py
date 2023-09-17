import cv2
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data_dir='C:/Users/admin/OneDrive - Bina Nusantara/Binus/Semester 5/Computer Vision/LAB/Project/Dataset/'
choice=0

def train_and_test(data_dir):
    images=[]
    Ls=[]
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    labels=os.listdir(data_dir)

    #Getting images from the dataset folder
    for label in labels:
        if os.path.isdir(os.path.join(data_dir,label)):
            temporary_folder=os.path.join(data_dir,label)
            print(temporary_folder)

            scanned_images=os.listdir(temporary_folder)

            if scanned_images:
                for image in scanned_images:
                    image_path=os.path.join(temporary_folder,image)
                    images.append(cv2.imread(image_path))
                    Ls.append(label)
            else:
                print("No image detected")
    
    #Processes all image inside the list
    i=1
    for im in images:
        gray_image=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        face=face_classifier.detectMultiScale(gray_image)
                
        for f in face:
            x,y,w,h=f
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow("Sample",im)
        cv2.waitKey(0)

        print(f'Image no.{i}')
        i+=1

    '''
    TrainLabel=np.array(TrainLabel)
    TrainLabel=cv2.UMat(TrainLabel)
    testL=np.array(testL)
    face_classifier=cv2.face.LBPHFaceRecognizer_create()
    face_classifier.train(train,TrainLabel)
    print("Model successfully trained")
    '''

while choice!=3:
    print("Football Player Face Recognition")
    print("1. Train and Test Model")
    print("2. Predict")
    print("3. Exit")
    choice = int(input(">> "))

    if choice==1:
        train_and_test(data_dir)
        
    elif choice==2:
        print("Predict")
    elif choice==3:
        print("Program terminated successfully")
        break
      


