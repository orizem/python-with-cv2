import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# For static image:
#img = cv2.imread('../Resources/people.jpg')
#width, height, tmp = img.shape
#resizeVal = 3
#img = cv2.resize(img, (height // resizeVal, width // resizeVal))
#imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

#for (x, y, w, h) in faces:
    #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

#cv2.imshow("Result", img)
#cv2.waitKey(0)

# For live detection:
cap = cv2.VideoCapture(0)

while True:
    success, video_img = cap.read()
    video_imgGray = cv2.cvtColor(video_img, cv2.COLOR_BGR2GRAY)

    video_faces = faceCascade.detectMultiScale(video_imgGray, 1.1, 4)

    for (x, y, w, h) in video_faces:
        cv2.rectangle(video_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Webcam", video_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
