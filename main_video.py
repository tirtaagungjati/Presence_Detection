import cv2
from simple_facerec import SimpleFacerec
import os
from datetime import datetime

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

## load some pre trained data on face data from https://github.com/opencv/opencv/tree/4.x/data/haarcascades
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
## load some pre trained data on face data from https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_smile.xml
smile = cv2.CascadeClassifier('haarcascade_smile.xml')

# Load Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
# cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def markAttendance(name):
    with open("Attendance.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.strip().split(",")
            nameList.append(entry[0])
        today = datetime.now().strftime("%Y-%m-%d")
        if name in nameList and today in myDataList[nameList.index(name)]:
            return
        dtString = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.writelines(f'\n{name},{dtString}')


while True:
    ret, frame = cap.read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        markAttendance(name)
        
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()