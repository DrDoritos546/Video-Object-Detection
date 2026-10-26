#capture video from camera
import numpy as np 
import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("cannot open camera")
    exit()
while True:
    #capture frame by frame 
    ret, frame = cap.read()
    #if frame not read correctly ret is true 
    if not ret:
        print("can't recieve frame, exiting...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break 

cap.release()
cv.destroyAllWindows()
