import numpy as np 
import cv2 as cv 

cap = cv.VideoCapture('C:/Users/hassa/Downloads/Avengers Secret Wars Main on End.mp4')
while cap.isOpened():
    ret,frame = cap.read() 
    if not ret:
        print("cant recieve")
        break 

    gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break 

cap.release() 
cv.destroyAllWindows()
    