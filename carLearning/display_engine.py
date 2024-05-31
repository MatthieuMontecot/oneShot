import numpy as np
import pygame

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255) 

def get_screen():
    windows_w=700
    windows_h=800
    center=np.array((windows_w/2,windows_h/2)).reshape(1,2,1)
    screen = pygame.display.set_mode((windows_w, windows_h))
    screen.fill(WHITE)
    return screen

def get_clock():
    clock = pygame.time.Clock()
    return clock

def quit():
    pygame.quit()