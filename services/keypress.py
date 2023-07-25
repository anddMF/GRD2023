import pygame
import cv2
import time


def init():
    pygame.init()
    win = pygame.display.set_mode((500, 500))


def get_key(key_name):
    ans = False
    for eve in pygame.event.get():
        pass
    key_input = pygame.key.get_pressed()
    my_key = getattr(pygame, 'K_{}'.format(key_name))
    if key_input[my_key]:
        ans = True
    pygame.display.update()
    return ans

def get_keyboard_press(drone, img):
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 40

    if get_key("SPACE"):
        drone.takeoff()

    if get_key("l"):
        drone.land()

    if get_key("UP"):
        fb = speed

    if get_key("DOWN"):
        fb = -speed

    if get_key("LEFT"):
        lr = -speed

    if get_key("RIGHT"):
        lr = speed

    if get_key("w"):
        ud = speed

    if get_key("s"):
        ud = -speed

    if get_key("d"):
        yv = speed + 15

    if get_key("a"):
        yv = -speed - 15

    if get_key("z"):
        cv2.imwrite(f'assets/images/pic_{time.time()}.jpg', img)
        time.sleep(0.3)

    if (get_key("p")):
        lr = 0
        fb = 0
        ud = 0
        yv = 0
    
    return [lr, fb, ud, yv]


def main():
    if get_key("LEFT"):
        print('left left')
    if get_key("RIGHT"):
        print('RIGHT RIGHT')


if __name__ == '__main__':
    init()
    while True:
        main()
