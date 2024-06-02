import numpy as np
import pygame

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)


def get_screen():
    windows_w = 700
    windows_h = 800
    screen = pygame.display.set_mode((windows_w, windows_h))
    screen.fill(WHITE)
    return screen


def get_clock():
    clock = pygame.time.Clock()
    return clock


def quit_display():
    pygame.quit()
