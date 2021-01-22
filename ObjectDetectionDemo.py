import cv2
import os
import numpy as np

#Given an image below function returns rectangle for face detected alongwith gray scale image
def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)#convert color image to grayscale
    face_haar_cascade=cv2.CascadeClassifier('/Users/Rohan/PycharmProjects/Python_Basics/practiceSession/Object_Detection_demo/haarcascade/haarcascade_frontalface_default.xml')#Load haar classifier
    print(face_haar_cascade)
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)#detectMultiScale returns rectangles
    #scale factor means decrease the original image size by 32%
    #min neighbors means it should have 5 value to detect faces

    return faces,gray_img

def labels_for_trainingdata(directory):
    faces=[]
    face_ID=[]

    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")
                continue

            id = os.path.basename(path)
            img_path=os.path.join(path,filename)
            print("image Path:",img_path)
            print("id:",id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("Image not loaded properly..")
                continue
            faces_rect,gray_img=faceDetection(test_img)
            if len(faces_rect)!=1:  #assuming only single person images are being given to classifier
                continue
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            face_ID.append(int(id))

    return faces,face_ID

def train_classifier(faces,face_ID):
   # face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(face_ID))
    return face_recognizer

def draw_rect(test_img,face):
    (x,y,w,h)  =face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,167,201), thickness=3)

def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)

