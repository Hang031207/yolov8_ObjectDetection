from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

results = model.predict(source='0',show=True)

while True:
    print(results)

    #If 'q' is pressed on keyboard, the video will be terminated
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break