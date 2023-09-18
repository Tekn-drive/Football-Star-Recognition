import cv2
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
    Ls=np.array(Ls)
    label_encoder=LabelEncoder()
    Ls=label_encoder.fit_transform(Ls)
    resized_images=[]

    for image in images:
        image = cv2.resize(image,(48,48),interpolation=cv2.INTER_AREA)
        resized_images.append(image)

    X_train,X_test,Y_train,Y_test=train_test_split(resized_images,Ls,test_size=0.25,random_state=69)
    face_classifier=cv2.face.LBPHFaceRecognizer_create()
    face_classifier.train(X_train,Y_train)

    correct_images=0

    for image,label in zip(X_test,Y_test):

        pred = face_classifier.predict(image)
        pred = pred[0]

        if pred == label:
            correct_images+=1

    print(f'Average Accuracy: {correct_images/len(X_test)*100}%')
    print(f'{correct_images} images correct out of {len(X_test)} images')

    face_classifier.save("FootBallStar.xml")
    print("Training and Testing Finished")

def test():
    labels={0:"christiano_ronaldo",
            1:"erling_haaland",
            2:"jorginho",
            3:"karim_benzema",
            4:"kylian_mbappe",
            5:"lionel_messi",
            6:"mohamed_salah",
            7:"neymar",
            8:"robert_lewandoski",
            9:"rumelu_lukaku"
            }
            
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read("FootBallStar.xml")
    absolute_file_path=input("Input absolute path for image to predict >> ")
    image=cv2.imread(absolute_file_path)
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray_image,minNeighbors=30,scaleFactor=1.1)

    for (x,y,w,h) in faces:
        face = gray_image[y:y+h,x:x+w]
        prediction=model.predict(face)
        predicted_label=labels[prediction[0]]
        confidence=round(prediction[1],2)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(image,f"{predicted_label} : {confidence}%",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        cv2.imshow("Predicted Image",image)
        cv2.waitKey(0)

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