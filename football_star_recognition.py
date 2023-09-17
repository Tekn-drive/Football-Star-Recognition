import cv2
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt

train_dir='C:/Users/admin/OneDrive - Bina Nusantara/Binus/Semester 5/Computer Vision/LAB/Project/Dataset/train'
test_dir='C:/Users/admin/OneDrive - Bina Nusantara/Binus/Semester 5/Computer Vision/LAB/Project/Dataset/test'
choice=0

def train_and_test(train_dir,test_dir):
    train=[]
    test=[]
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    train_labels=os.listdir(train_dir)
    test_labels=os.listdir(test_dir)

    #Getting images from the train folder
    for label in train_labels:
        if os.path.isdir(os.path.join(train_dir,label)):
            temporary_folder=os.path.join(train_dir,label)
            images=os.listdir(temporary_folder)
            print(temporary_folder)

            if images:
                print("Images detected")
                for image in images:
                    train_image_path=os.path.join(temporary_folder,image)
                    train.append(cv2.imread(train_image_path))
            else:
                print("No image detected")
    
    for label in test_labels:
        if os.path.isdir(os.path.join(test_dir,label)):
            temporary_folder=os.path.join(test_dir,label)
            images=os.listdir(temporary_folder)
            print(temporary_folder)

        if images:
            print("Images detected")
            for image in images:
                test_image_path=os.path.join(temporary_folder,image)
                test.append(cv2.imread(test_image_path))
        else:
            print("No image detected")
    
    #Processes all image inside the train list
    i=1
    for t in train:
        gray_image=cv2.cvtColor(t,cv2.COLOR_BGR2GRAY)
        face=face_classifier.detectMultiScale(gray_image)
                
        for f in face:
            x,y,w,h=f
            cv2.rectangle(t,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow("Train",t)
        cv2.waitKey(0)

        print(f'Train Image no.{i}')
        i+=1

    #Processes all image inside the test list
    i=1
    for t in test:
        gray_image=cv2.cvtColor(t,cv2.COLOR_BGR2GRAY)
        face=face_classifier.detectMultiScale(gray_image)
            
        for f in face:
            x,y,w,h=f
            cv2.rectangle(t,(x,y),(x+w,y+h),(0,255,0),2)
        
        cv2.imshow("Test",t)
        cv2.waitKey(0)
        
        print(f'Test Image no.{i}')
        i+=1

while choice!=3:
    print("Football Player Face Recognition")
    print("1. Train and Test Model")
    print("2. Predict")
    print("3. Exit")
    choice = int(input(">> "))

    if choice==1:
        train_and_test(train_dir,test_dir)
        
    elif choice==2:
        print("Predict")
    elif choice==3:
        print("Program terminated successfully")
        break
      


