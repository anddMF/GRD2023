from ultralytics import YOLO
import cv2
import math
import numpy as np
from djitellopy import tello
import services.keypress as kp
import time

drone = tello.Tello()
drone.connect()
print(drone.get_battery())

w, h = 640, 480
fbSizeRange = [11000, 13000]
udPositionRange = [160, 180]
pid = [0.4, 0.4, 0]
pError = 0

global img

drone.takeoff()
drone.streamon()

# start webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, w)
# cap.set(4, h)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


def trackObject(drone, info, w, pid, pError):
    area = info[1]
    x, y = info[0]
    fb = 0
    ud = 0
    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))

    # stay stationary if is in the range
    # move foward (fb) if its too far, move backwards if its too close
    if area > fbSizeRange[0] and area < fbSizeRange[1]:
        fb = 0
    elif area > fbSizeRange[1]:
        fb = -10
    elif area < fbSizeRange[0] and area != 0:
        fb = 10

    if x == 0:
        speed = 0
        error = 0

    # y 170
    print("ALTURA", y)

    if y > udPositionRange[0] and y < udPositionRange[1]:
        ud = 0
    elif y > udPositionRange[1]:
        ud = -30
    elif y < udPositionRange[0] and ud != 0:
        ud = 30
        
    print("COMANDOS: FB ", fb, "UD ", ud, "SPEED ", speed)
    # TODO: add speed again, but its not working 
    drone.send_rc_control(0, fb, ud, 0)
    
    return error


while True:
    img = drone.get_frame_read().frame
    # success, img = cap.read()
    
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            if classNames[cls] == "cell phone":
                # bounding box
                f1, f2, w, h = box.xywh[0]
                w, h = int(w), int(h)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # create a box and a circle on the object detected
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cx = x1 + w // 2
                cy = y1 + h // 2
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->", confidence)

                area = (x2 - x1) * (y2 - y1)  # pra celular, entre 6k e 7k
                print("AREA AREA AREA", area)
                detectedInfo = [[cx, cy], area]
                print("AREA", detectedInfo[1], "CENTER", detectedInfo[0])

                # object details
                color = (0, 255, 0)
                cv2.putText(img, classNames[cls], [x1, y1],
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # send coord to drone
                pError = trackObject(drone, detectedInfo, w, pid, pError)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        drone.land()
        break

# cap.release()
cv2.destroyAllWindows()
