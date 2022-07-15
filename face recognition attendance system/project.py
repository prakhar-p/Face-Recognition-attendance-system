import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime


path = 'imagesList'
images = []
imageName = []

mylist = os.listdir(path)
for i in mylist:
    currentImg = cv2.imread(f'{path}/{i}')
    images.append(currentImg)
    imageName.append(os.path.splitext(i)[0])

print(imageName)

def encoding(images):
    encodeList = []
    for i in images:
        convert = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(convert)[0]
        encodeList.append(encode)
    return encodeList

def attendence(name):
    with open('Attendence.csv', 'r+') as f:
        myDataList = f.readlines()
        myNameList = []
        for line in myDataList:
            entry = line.split(',')
            myNameList.append(entry[0])
        if name not in myNameList:
            now = datetime.now()
            now1 = datetime.today()
            dstring = now1.strftime("%d/%m/%y")
            tstring = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{tstring},{dstring}')

knownFace = encoding(images)
print(len(knownFace))

camera = cv2.VideoCapture(0)
while True:
    success, img = camera.read()
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25, )
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    faceCurrentFrame = face_recognition.face_locations(imgSmall)
    encodeCurrentFrame = face_recognition.face_encodings(imgSmall, faceCurrentFrame)

    for encodeFace, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
        match = face_recognition.compare_faces(knownFace, encodeFace)
        faceDistance = face_recognition.face_distance(knownFace, encodeFace)
        print(faceDistance)
        matchIndex = np.argmin(faceDistance)

        if match[matchIndex]:
            name = imageName[matchIndex]
            print(name)
                # y1,x2,y2,x1=faceLoc
                # y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            x1, y1, x2, y2 = faceLoc
            x1, y1, x2, y2 = x1 * 4, y1 * 4, x2 * 4, y2 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
            attendence(name)
    cv2.imshow("Camera", img)
    cv2.waitKey(1)
