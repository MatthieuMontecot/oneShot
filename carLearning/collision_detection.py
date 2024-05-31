import numpy as np
import math_utils as mu
import car_model
import line_class

class BoundingBox():
    def __init__(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        
def BB_collision_detection(object_A, object_B):
    if object_A.min_x > object_B.max_x:
        return False
    if object_B.min_x > object_A.max_x:
        return False
    if object_A.min_y > object_B.max_y:
        return False
    if object_B.min_y > object_A.max_y:
        return False
    return True

def collide_car_line(car, line):
    assert(isinstance(car, car_model.Car))
    assert(isinstance(line, line_class.Line))
    if not BB_collision_detection(car, line):
        return
    elif mu.distance(line.A, car.position) < car.radius:
        car.collided = True
        car.score -= 100
    elif mu.distance(line.B, car.position) < car.radius:
        car.collided = True
        car.score -= 100
    elif np.abs(np.dot(line.v_orth, car.position) - line.C) > car.radius:
        return
    else:
        car.collided = True
        car.score -= 100

def collide_vector_line(vector_point, e_vector, depth, line):
    '''we admit e_vector is normalized, we apply e_vector on vector point,
    thus we check if the line from vector_point to vector_point + e_vector cross'''
    assert (isinstance(line, line_class.Line))
    vector_constant = mu.orthogonal_dot(e_vector, vector_point)
    if np.sign(mu.orthogonal_dot(e_vector, line.A) - vector_constant) == np.sign(mu.orthogonal_dot(e_vector, line.B) - vector_constant):
        return False
    if np.sign(mu.orthogonal_dot(line.v, vector_point) - line.C) == np.sign(mu.orthogonal_dot(line.v, depth * e_vector + vector_point) - line.C):
        return False
    return True