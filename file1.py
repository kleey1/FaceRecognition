#pip install opencv-python in command prompt to install opencv
import cv2
import numpy as np
import face_recognition

imgSylvester = face_recognition.load_image_file('images/sylvesterstallone.jpg')
imgSylvester = cv2.cvtColor(imgSylvester, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('images/sylvesterstallonetest.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgSylvester)[0]
encodeSylvester = face_recognition.face_encodings(imgSylvester)[0]
cv2.rectangle(imgSylvester, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

#using linear SVM to compare the faces
results = face_recognition.compare_faces([encodeSylvester], encodeTest)
faceDis = face_recognition.face_distance([encodeSylvester], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Sylvester', imgSylvester)
cv2.imshow('Sylvestertest', imgTest)
cv2.waitKey(0)




