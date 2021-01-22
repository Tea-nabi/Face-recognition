import cv2
import os
import numpy as np
import ObjectDetectionDemo as ob

test_img = cv2.imread('Test_images/IMG_20171223_191856_Bokeh.jpg')
faces_detected,gray_img = ob.faceDetection(test_img)
print("Face Detected:",faces_detected)

# for (x,y,w,h) in faces_detected:
#     cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,255,0), thickness=3)
#
# resized_img = cv2.resize(test_img,(500,500))
# cv2.imshow("face Detection tutorial",resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# faces,faceID = ob.labels_for_trainingdata('/Users/Rohan/PycharmProjects/Python_Basics/practiceSession/Object_Detection_demo/training')
# face_recognizer=ob.train_classifier(faces,faceID)
# face_recognizer.save('trainingData.yml')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('/Users/Rohan/PycharmProjects/Python_Basics/practiceSession/Object_Detection_demo/trainingData.yml')

name = {0:"Rohan",1:"Sonu"}

#cap=cv2.VideoCapture(0)

# while True:
#     ret,test_img=cap.read()# captures frame and returns boolean value and captured image
#     faces_detected,gray_img=ob.faceDetection(test_img)
#     for (x,y,w,h) in faces_detected:
#       cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)
#
#     resized_img = cv2.resize(test_img, (1000, 700))
#     cv2.imshow('face detection Tutorial ',resized_img)
#     cv2.waitKey(10)



# for face in faces_detected:
#     (x,y,w,h)=face
#     roi_gray = gray_img[y:y+h,x:x+w]
#     label,confidence = face_recognizer.predict(roi_gray)
#     print("confidence: ",confidence)
#     print("label: ",label)
#
#     if (confidence > 40):
#         continue
#     ob.draw_rect(test_img,face)
#     predicted_name = name[label]
#     ob.put_text(test_img,predicted_name,x,y)


for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        print("confidence:",confidence)
        print("label:",label)
        ob.draw_rect(test_img,face)
        predicted_name=name[label]
        if confidence < 100:#If confidence less than 37 then don't print predicted face text on screen
            ob.put_text(test_img,predicted_name,x,y)

resized_img = cv2.resize(test_img,(700,500))
cv2.imshow("face dtecetion tutorial",resized_img)
cv2.waitKey(0)#Waits indefinitely until a key is pressed
cv2.destroyAllWindows