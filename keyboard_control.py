from djitellopy import tello
import services.keypress as kp
import time
import cv2

drone = tello.Tello()
drone.connect()
print(drone.get_battery())

global img

drone.streamon()

def get_keyboard_press():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 40

    if kp.get_key("SPACE"):
        drone.takeoff()

    if kp.get_key("l"):
        drone.land()

    if kp.get_key("UP"):
        fb = speed

    if kp.get_key("DOWN"):
        fb = -speed

    if kp.get_key("LEFT"):
        lr = -speed

    if kp.get_key("RIGHT"):
        lr = speed

    if kp.get_key("w"):
        ud = speed

    if kp.get_key("s"):
        ud = -speed

    if kp.get_key("d"):
        yv = speed

    if kp.get_key("a"):
        yv = -speed

    if kp.get_key("z"):
        cv2.imwrite(f'assets/images/pic_{time.time()}.jpg', img)
        time.sleep(0.3)

    if (kp.get_key("p")):
        lr = 0
        fb = 0
        ud = 0
        yv = 0
    
    return [lr, fb, ud, yv]



while True:
    kp.init()
    img = drone.get_frame_read().frame
    img = cv2.resize(img, (640, 360))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

    commands = get_keyboard_press()
    drone.send_rc_control(commands[0], commands[1], commands[2], commands[3])
    time.sleep(0.05)


