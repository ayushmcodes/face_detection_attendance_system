#https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
#https://stackoverflow.com/questions/51721695/dlib-installation-error
import cv2
import numpy as np
import face_recognition
print("Running")

#Converting Elon Image from BGR scale to RGB scale
imgElon=face_recognition.load_image_file('elon.jfif')
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)


#Converting Elon Image used for testing purpose from BGR scale to RGB scale
imgTest=face_recognition.load_image_file('elontest.jfif')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)


#detecting faces in the image
#[0] means coordinates of first face in the image4
#(top, right, bottom, left) is the coordinate representation
faceLoc=face_recognition.face_locations(imgElon)[0]

#encoding the image i.e finding 128 measurements
#[0] means 128 measurements of the first face
encodeElon=face_recognition.face_encodings(imgElon)[0]

#cv2.rectangle(img,(starting point),(ending point),color,thickness)
#(point)->(x,y)
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)



#repeating the above three code for the test image
faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)



#comparing measurements of elon and elontest using SVM classifier
results=face_recognition.compare_faces([encodeElon],encodeTest)
print(results)


#face_distance is used to find how close the two images are
#closer the distance more similar the images
faceDis=face_recognition.face_distance([encodeElon],encodeTest)
print(faceDis)


cv2.putText(imgTest,f'{results}',(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)