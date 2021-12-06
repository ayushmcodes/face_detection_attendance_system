import cv2
import numpy as np
import face_recognition
from datetime import datetime
print("Running")

#OS module provides functionalities 
#It is used to work with current directory,make files,folders etc
import os

path='ImagesAttendance'
images=[]
classNames=[]

myList=os.listdir(path)


for cl in myList:
    currImg=cv2.imread(f'{path}/{cl}')
    images.append(currImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            #split name and time using comma(,)
            entry=line.split(',')
            nameList.append(entry[0])
        
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



#-------------------------------------

encodeListKnown=findEncodings(images)
print("Encoding of known faces completed")

#Now we have to find match with the encodings
#Now will use webcam to capture input images
cap=cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    cv2.imshow('Camera',img)
    if cv2.waitKey(1) & 0xFF==32:
        print("Exit Key Pressed")
        break

    imgS=cv2.resize(img,(0,0),None,.25,.25)
    imgS=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    #in webcam image we may find multiple faces
    facesCurrFrame=face_recognition.face_locations(imgS)
    encodeCurrFrame=face_recognition.face_encodings(imgS,facesCurrFrame)

    #now we will iterate through all the faces found in the current frame
    #and will compare with the encoding we have found before


    #In the below code faceLoc will store the facelocation
    #and encodeFace will store the corresponding encodings 
    for encodeFace,faceLoc in zip(encodeCurrFrame,facesCurrFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        print(matches)
        #Lowest value of faceDis will be our answer

        #argmin returns the index of the minvalue from the array
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
