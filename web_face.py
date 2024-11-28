import cv2
import numpy as np
import pyfiglet
# import time

cap = cv2.VideoCapture(0)
figlet = pyfiglet.figlet_format('Hi, let is look for your face:')
figlet_hello = pyfiglet.figlet_format('GO!')
print(figlet)




print('enter scaleFactor(3): ')
n = np.double(input())
print('enter minNeighbors(1): ')
b = int(input())
print(figlet_hello)
while True:
    # time.sleep(0.1)
    success, img = cap.read()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.Canny(img, 50, 50)
    # kernel = np.ones((5, 3), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # img = cv2.erode(img, kernel, iterations=1)
    # cv2.rectangle(img, (10, 10), (100, 100), (255, 0, 0), thickness=3)
    faces = cv2.CascadeClassifier('_faces.xml')
    result = faces.detectMultiScale(img, scaleFactor=n, minNeighbors=b)
    for (x, y, w, h) in result:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness=3)
        


    # smiles = cv2.CascadeClassifier('eye.xml')
    # results1 = smiles.detectMultiScale(img, scaleFactor=4, minNeighbors=2)
    # for (x, y, w, h) in results1:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), thickness=3)
    cv2.imshow('Result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    