import numpy as np
import math_utils as mu
import pygame


class Line:
    def __init__(self, x0, y0, x1, y1):
        self.A = np.array([x0, y0])
        self.B = np.array([x1, y1])
        self.AB = self.B - self.A
        self.min_x = min([x0, x1])
        self.min_y = min([y0, y1])
        self.max_x = max([x0, x1])
        self.max_y = max([y0, y1])
        self.v = mu.normalize_v(self.AB)
        self.v_orthogonal = np.array([- self.v[1], self.v[0]])
        self.C = mu.orthogonal_dot(self.v, self.A)

    def display(self, screen):
        pygame.draw.line(screen, (0, 0, 0), self.A, self.B)
