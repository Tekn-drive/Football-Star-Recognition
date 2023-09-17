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
                    images.append(cv2.imread(image_path))
                    Ls.append(label)
            else:
                print("No image detected")
    
    #Processes all image inside the list
    i=1

    gray_images=[]

    for im in images:
        im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        gray_images.append(im)

    '''
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

    Ls=np.array(Ls)
    label_encoder=LabelEncoder()
    Ls=label_encoder.fit_transform(Ls)
    X_train,X_test,Y_train,Y_test=train_test_split(gray_images,Ls,test_size=0.25,random_state=6)
    face_classifier=cv2.face.LBPHFaceRecognizer_create()
    face_classifier.train(X_train,Y_train)
    face_classifier.save("FootBallStar.xml")

    trues=0
    fps=0
    fns=0

    trained_model = cv2.CascadeClassifier("FootBallStar.xml")

    for i in range(len(X_test)):
        x = X_test[i]
        y = Y_test[i]

        #It crashes here
        detected_face=trained_model.detectMultiScale(x)

        if detected_face:
            if y in detected_face:
                trues+=1
            else:
                fps+=1
        else:
            fns+=1
    
    precision = trues/(trues+fps)
    recall = trues/ (trues + fns)
    f1_score = 2*(precision*recall)/(precision+recall)

    print(precision)
    print(recall)
    print(f1_score)

    print("Training and Testing Finished")
    
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
      


