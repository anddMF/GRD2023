import cv2
import numpy as np
from djitellopy import tello
import services.keypress as kp
import cv2
import time

drone = tello.Tello()
drone.connect()
print(drone.get_battery())

drone.takeoff()

w, h = 640, 360
fbRange = [7200, 7600]
pid = [0.4, 0.4, 0]
pError = 0

global img

drone.streamon()

def findFace(img):
    faceCascade = cv2.CascadeClassifier(
        "assets/code/haarcascade_eye.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 8)

    myFaceListCenter = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListCenter.append([cx, cy])
        myFaceListArea.append(area)
    # check the size of area that the face is using, if its too big, the camera is too close
    # it also checks and returns the center value
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListCenter[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]


def trackFace(drone, info, w, pid, pError):
    area = info[1]
    x,y = info[0]
    fb = 0

    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))

    # stay stationary if is in the range
    # move foward (fb) if its too far, move backwards if its too close
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20
    
    if x == 0:
        speed = 0
        error = 0

    drone.send_rc_control(0, fb, 0, speed)
    return error


# cap = cv2.VideoCapture(0)
while True:
    # _, img = cap.read()
    kp.init()
    img = drone.get_frame_read().frame
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, info = findFace(img)
    pError = trackFace(drone, info, w, pid, pError)
    print("AREA", info[1], "CENTER", info[0])
    cv2.imshow("Output", img)

    # commands = kp.get_keyboard_press(drone, img)
    # drone.send_rc_control(commands[0], commands[1], commands[2], commands[3])
    # time.sleep(0.05)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        drone.land()
        break
    
