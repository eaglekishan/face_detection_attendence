import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# creating path
path="img"
images=[]
className=[]
myList=os.listdir(path)
print(myList)

# reading the images one by one and appending into a list
for ci in myList:
    curImg=cv2.imread(path+'/'+ci)
    images.append(curImg)
    className.append(ci.split('.')[0])
print(className)

# encoding the images
def findEncoding(images):
    encode=[]
    for i in images:
        img=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        en=face_recognition.face_encodings(img)[0]
        encode.append(en)
    return encode

encodeList=findEncoding(images)
print("Encoding Completed")

# reading live image from video
cap=cv2.VideoCapture(0)
while True:
    succ,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    currFaceLoc=face_recognition.face_locations(imgS)[0]
    currFaceEncode=face_recognition.face_encodings(imgS)[0]

    for faceLoc,faceEncode in zip(currFaceLoc,currFaceEncode):
        match=face_recognition.compare_faces(encodeList,faceEncode)
        dist=face_recognition.face_distance(encodeList,faceEncode)
        matchIndex=np.argmin(dist)


        if match[matchIndex]:
            name=className[matchIndex].upper()

            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow('webcam',img)
    cv2.waitKey(1)
