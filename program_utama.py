import numpy as np
import cv2
import face_recognition
import os
import datetime

path = 'D:\RoadMap Kuliah\Semester 7\Visi Komputer\Tugas\source-code-face-recognition\source code\images'
smile = cv2.CascadeClassifier('D:\RoadMap Kuliah\Semester 7\Visi Komputer\Tugas\source-code-face-recognition\source code\haarcascade_smile.xml')
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('D:\RoadMap Kuliah\Semester 7\Visi Komputer\Tugas\source-code-face-recognition\source code\Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            dtString = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.writelines(f'\n{name},{dtString}')
            


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)
    
    for encodeFace, faceloc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex=np.argmin(faceDis)
        
        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2), (255,255,255), 2)
            cv2.putText(img, name, (x1+6,y1-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
            the_face = img[y1:y2,x1:x2]
            face_grayscale = cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)
            smiles = smile.detectMultiScale(face_grayscale,scaleFactor=1.7, minNeighbors=20)
            for (x,y,w,h) in smiles:
                if len(smiles)>0:
                    cv2.putText(img, 'Senyum', (x1, y2+40),fontScale=1,fontFace=cv2.FONT_HERSHEY_COMPLEX,color=(255,255,0))
                    markAttendance(name)
    cv2.imshow('webcam',img)
    key = cv2.waitKey(1)    
    if key == 81  or key == 113:
        break
cap.release()
cv2.destroyAllWindows()
