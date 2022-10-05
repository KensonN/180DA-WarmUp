# Resources: 
# https://answers.opencv.org/question/200861/drawing-a-rectangle-around-a-color-as-shown/
# https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
# Added BGR image masking and threshold contour area condition for drawing

import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,0])
    upper_blue = np.array([140,255,255])
    #define blue range in BGR
    lower_blue2 = np.array([255,0,0])
    upper_blue2 = np.array([255,200,200])
    # Threshold HSV image for blue
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Threshold BGR image for blue
    # mask = cv.inRange(frame, upper_blue2, upper_blue2)

    contours, _ =  cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
    if len(contours)>0:
        blue_area = max(contours, key=cv.contourArea)
        if cv.contourArea(blue_area) > 1000: 
            (xg,yg,wg,hg) = cv.boundingRect(blue_area)
            cv.rectangle(frame,(xg,yg),(xg+wg, yg+hg),(0,255,0),2)

    cv.imshow('frame', frame)
    
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()