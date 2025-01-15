import ultralytics
from ultralytics import YOLO
import cv2
import cvzone
import math

#Running Real-time from Webcam
cap = cv2.VideoCapture('Pigs2.mp4')

#put in the trained weight
model = YOLO('best.pt')

#This is the name that will appear on the detection screen:
classnames = ['Pig']

while (cap.isOpened()):
    ret,frame = cap.read()
    #resize the display window a bit
    frame = cv2.resize(frame,(640,480))
    result = model(frame,stream=True)

    #Getting bbox, confidence and class names information to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            Class = int(box.cls[0])
            #We only show the boxes if the confidence is over 50%
            if confidence > 0.5:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                #limit the confidence to a number under 100, with 2 decimal places
                confidence = math.ceil(confidence*100)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),5)
                cvzone.putTextRect(frame,f'{classnames[Class]}: {confidence}%',[x1+8,y1+100],scale=1.5,thickness=1)
                #scale: scale of annotation font
                #thickness: thickness of border
    
    #If 'q' is pressed on keyboard, the video will be terminated
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    if ret == True:
        cv2.imshow('frame',frame)
    else:
        break

#All the frame captured will be released
cap.release()
cv2.destroyAllWindows()