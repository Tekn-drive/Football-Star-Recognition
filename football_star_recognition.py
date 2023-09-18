import cv2
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

data_dir='C:/Users/admin/OneDrive - Bina Nusantara/Binus/Semester 5/Computer Vision/LAB/Project/Dataset/'
choice=0

def train_and_test(data_dir):
    print("Training and Testing")
    images=[]
    Ls=[]
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    labels=os.listdir(data_dir)

    #Getting images from the dataset folder
    for label in labels:
        if os.path.isdir(os.path.join(data_dir,label)):
            temporary_folder=os.path.join(data_dir,label)
            scanned_images=os.listdir(temporary_folder)

            if scanned_images:
                for image in scanned_images:
                    image_path=os.path.join(temporary_folder,image)
                    image=cv2.imread(image_path,0)
                    faces=face_classifier.detectMultiScale(image,minNeighbors=30,scaleFactor=1.1)
                    
                    if(len(faces)<1):
                        continue
                    for (x,y,w,h) in faces:
                        face=image[y:y+h,x:x+w]
                        images.append(face)
                        Ls.append(label)
            else:
                print("No image detected")
    
    #Processes all image inside the list
    i=1

    Ls=np.array(Ls)
    label_encoder=LabelEncoder()
    Ls=label_encoder.fit_transform(Ls)
    X_train,X_test,Y_train,Y_test=train_test_split(images,Ls,test_size=0.25,random_state=69)
    X_train_resized=[cv2.resize(image,(48,48),interpolation=cv2.INTER_AREA)for image in X_train]
    X_test_resized=[cv2.resize(image,(48,48),interpolation=cv2.INTER_AREA)for image in X_test]

    face_classifier=cv2.face.LBPHFaceRecognizer_create()
    face_classifier.train(X_train_resized,Y_train)

    correct_images=0

    for image,label in zip(X_test_resized,Y_test):

        pred = face_classifier.predict(image)
        pred = pred[0]

        if pred == label:
            correct_images+=1

    print(f'Average Accuracy: {correct_images/len(X_test_resized)*100}%')
    print(f'{correct_images} images correct out of {len(X_test_resized)} images')

    face_classifier.save("FootBallStar.xml")
    print("Training and Testing Finished")

def test():
    model = cv2.CascadeClassifier("FootBallStar.xml")
    absolute_file_path=input("Input absolute path for image to predict >> ")
    image=cv2.imread(absolute_file_path)
    pred = model.predict(image)
    print(pred[0])

while choice!=3:
    print("Football Player Face Recognition")
    print("1. Train and Test Model")
    print("2. Predict")
    print("3. Exit")
    choice = int(input(">> "))

    if choice==1:
        train_and_test(data_dir)
        
    elif choice==2:
        test()
    elif choice==3:
        print("Program terminated successfully")
        break