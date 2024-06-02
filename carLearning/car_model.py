import numpy as np
from scipy.special import expit
import pygame
import math_utils as mu
import collision_detection
import physics_engine

PI = np.pi
PI_OVER_TWO = np.pi / 2
THREE_PI_OVER_TWO = 3 * np.pi / 2
TWO_PI = 2 * np.pi
BLUE = np.array((0, 0, 255))


class Car:
    """a car is a very simple neural network with 2 hidden layers of size 4,3, the input size is 5 and the output
    size is 2 the input are 5 values corresponding to the distance of sight in a given direction (in front of the car),
     and 4 other directions (2 at right and 2 at left) the output are 2 values, one in the range (0,1) corresponding to
     the engine power, I don't want cars to move backward because if let them hack the perf function the other output
     value is a float number in [-1,1] corresponding to the derivative of the angle of the car i.e. the angular speed of
     the car. This number is multiplied later by some coefficient to be in an expected natural range. (a variation of 1
      gradient in 0.1 second is not natural and allows models to do loops)"""
    color = BLUE

    def __init__(self):
        self.h1 = None
        self.h2 = None
        self.h3 = None
        self.theta_prime = None
        self.max_x = None
        self.max_y = None
        self.min_y = None
        self.min_x = None
        self.sin = None
        self.score = None
        self.collided = None
        self.e_theta = None
        self.sensor_bounding_boxes = None
        self.sensor_thetas = None
        self.theta = None
        self.acceleration = None
        self.speed = None
        self.position = None
        self.X = None
        self.sensor_projection = None
        self.sensor_e_thetas = None
        self.cos = None
        self.mutation_scale_factor = 0.05
        self.W1 = np.random.rand(4, 5) - 0.5
        self.b1 = np.random.rand(4, 1) - 0.5
        self.W2 = np.random.rand(3, 4) - 0.5
        self.b2 = np.random.rand(3, 1) - 0.5
        self.W3 = np.random.rand(2, 3) - 0.5
        self.b3 = np.random.rand(2, 1) - 0.5
        self.radius = physics_engine.universal_scale_factor * 20
        self.sensor_depth = physics_engine.universal_scale_factor * 100
        self.number_of_sensor = 5
        self.theta_angle = 30 * TWO_PI / 360
        self.reset()
        self.sensor_colors = self.number_of_sensor * [(125, 60, 10)]

    def reset(self):
        """resets the car state for next generation"""
        self.sensor_e_thetas = np.zeros((5, 2))
        self.sensor_projection = np.zeros((5, 2))
        self.X = np.zeros((5, 1))
        self.X.fill(self.sensor_depth)
        self.position = physics_engine.universal_scale_factor * np.array([100., 100.])
        self.speed = np.array([0., 0.])
        self.acceleration = np.array([0., 0.])
        self.theta = 0.
        self.sensor_thetas = np.zeros(5)
        self.update_sensor_thetas()
        self.sensor_bounding_boxes = [collision_detection.BoundingBox(0, 0, 0, 0) for _ in range(self.number_of_sensor)]
        self.update_sensor_thetas()
        self.e_theta = mu.get_e_theta(self.theta)
        self.collided = False
        self.score = 0.

    def update_sensor_projection(self):
        """updates the car sensors"""
        for i in range(self.number_of_sensor):
            self.sensor_projection[i] = self.position + self.X[i] * self.sensor_e_thetas[i]

    def display_sensor(self, screen):
        for i in range(self.number_of_sensor):
            if self.X[i] < self.sensor_depth:
                pygame.draw.circle(screen, (0, 150, 75), self.sensor_projection[i], 5)
            else:
                pygame.draw.circle(screen, self.sensor_colors[i], self.sensor_projection[i], 3)

    def update_sensor_thetas(self):
        for i in range(self.number_of_sensor):
            self.sensor_thetas[i] = (self.theta + (i - (self.number_of_sensor - 1) / 2) * self.theta_angle) % TWO_PI
            self.sensor_e_thetas[i: i + 1] = mu.get_e_theta(self.sensor_thetas[i], self.sensor_e_thetas[i])

    def get_line_of_sight(self, lines):
        self.X.fill(self.sensor_depth)
        for line in lines:
            for sensorS_n in range(self.number_of_sensor):
                if collision_detection.bounding_box_collision_detection(self.sensor_bounding_boxes[sensorS_n], line):
                    if collision_detection.collide_vector_line(self.position, self.sensor_e_thetas[sensorS_n],
                                                               self.X[sensorS_n],
                                                               line):
                        intersection = mu.cross_lines(self.position, self.sensor_e_thetas[sensorS_n], line.A, line.v)
                        distance = mu.distance(intersection, self.position)
                        assert (distance <= self.X[sensorS_n] + 1)
                        self.X[sensorS_n] = distance
        self.update_sensor_projection()

    def update_score(self):
        if not self.collided:
            self.score += physics_engine.deltaT * mu.norm(self.speed) + 0.5

    def feed(self):
        """this function allows to compute the output of the neuron using the self.X array (input) which should be
        set before """
        self.h1 = np.tanh(np.dot(self.W1, self.X / self.sensor_depth) - self.b1)
        self.h2 = np.tanh(np.dot(self.W2, self.h1) - self.b2)
        self.h3 = np.tanh(np.dot(self.W3, self.h2) - self.b3)
        self.acceleration = expit(self.h3[1][0])
        self.theta_prime = np.tanh(self.h3[0][0])

    def clone(self):
        """clones the car instance"""
        clone = Car()
        clone.W1 = self.W1.copy()
        clone.W2 = self.W2.copy()
        clone.W3 = self.W3.copy()
        return clone

    def mutate(self):
        """mutation, replace a given weight per another"""
        i = np.random.randint(3)
        j = np.random.randint(2)
        if i == 0 and j == 0:
            layer = self.W1
            self.W1[np.random.randint(layer.shape[0]), np.random.randint(layer.shape[1])] = np.random.uniform(-1, 1, 1)
        if i == 0 and j == 1:
            layer = self.b1
            self.b1[np.random.randint(layer.shape[0]), np.random.randint(layer.shape[1])] = np.random.uniform(-1, 1, 1)
        if i == 1 and j == 0:
            layer = self.W2
            self.W2[np.random.randint(layer.shape[0]), np.random.randint(layer.shape[1])] = np.random.uniform(-1, 1, 1)
        if i == 1 and j == 1:
            layer = self.b2
            self.b2[np.random.randint(layer.shape[0]), np.random.randint(layer.shape[1])] = np.random.uniform(-1, 1, 1)
        if i == 2 and j == 0:
            layer = self.W3
            self.W3[np.random.randint(layer.shape[0]), np.random.randint(layer.shape[1])] = np.random.uniform(-1, 1, 1)
        if i == 2 and j == 1:
            layer = self.b3
            self.b3[np.random.randint(layer.shape[0]), np.random.randint(layer.shape[1])] = np.random.uniform(-1, 1, 1)

    def tiny_mutate(self):
        """mutation, modify a weight"""
        i = np.random.randint(3)
        j = np.random.randint(2)
        if i == 0 and j == 0:
            layer = self.W1
            self.W1[np.random.randint(layer.shape[0]), np.random.randint(
                layer.shape[1])] += self.mutation_scale_factor * np.random.uniform(-1, 1, 1)
        if i == 0 and j == 1:
            layer = self.b1
            self.b1[np.random.randint(layer.shape[0]), np.random.randint(
                layer.shape[1])] += self.mutation_scale_factor * np.random.uniform(-1, 1, 1)
        if i == 1 and j == 0:
            layer = self.W2
            self.W2[np.random.randint(layer.shape[0]), np.random.randint(
                layer.shape[1])] += self.mutation_scale_factor * np.random.uniform(-1, 1, 1)
        if i == 1 and j == 1:
            layer = self.b2
            self.b2[np.random.randint(layer.shape[0]), np.random.randint(
                layer.shape[1])] += self.mutation_scale_factor * np.random.uniform(-1, 1, 1)
        if i == 2 and j == 0:
            layer = self.W3
            self.W3[np.random.randint(layer.shape[0]), np.random.randint(
                layer.shape[1])] += self.mutation_scale_factor * np.random.uniform(-1, 1, 1)
        if i == 2 and j == 1:
            layer = self.b3
            self.b3[np.random.randint(layer.shape[0]), np.random.randint(
                layer.shape[1])] += self.mutation_scale_factor * np.random.uniform(-1, 1, 1)

    def crossover(self, mate):
        """returns an offspring for this instance and his mate"""
        child = Car()
        crossover_point = np.random.randint(1, 3)
        child.W1 = 1 * self.W1
        child.b1 = 1 * self.b1
        if crossover_point == 1:
            child.W2 = 1 * mate.W2
            child.b2 = 1 * mate.b2
        else:
            child.W2 = 1 * mate.W2
            child.b2 = 1 * mate.b2
        child.W3 = 1 * mate.W3
        child.b3 = 1 * mate.b3
        return child

    def update_cos_sin(self):
        self.cos, self.sin = np.cos(self.theta), np.sin(self.theta)

    def get_bounding_box(self):
        """computes the corners of the vehicles given there center M and their orientations."""
        self.min_x, self.max_x = self.position[0] - self.radius, self.position[0] + self.radius
        self.min_y, self.max_y = self.position[1] - self.radius, self.position[1] + self.radius

    def display(self, screen):
        pygame.draw.circle(screen, self.color, self.position, self.radius)
        pygame.draw.line(screen, (0, 0, 0), self.position, self.radius * self.e_theta + self.position)

    def update_bounding_box(self):
        self.min_x = self.position[0] - self.radius
        self.max_x = self.position[0] + self.radius
        self.min_y = self.position[1] - self.radius
        self.max_y = self.position[1] + self.radius

    def update(self, lines):
        self.update_sensor_bounding_box()
        self.get_line_of_sight(lines)
        self.update_score()
        self.feed()

    def update_sensor_bounding_box(self):
        """update line of sight bounding_box"""
        for i, sensor_theta in enumerate(self.sensor_thetas):
            if sensor_theta < PI_OVER_TWO or sensor_theta > THREE_PI_OVER_TWO:
                self.sensor_bounding_boxes[i].min_x = self.position[0]
                self.sensor_bounding_boxes[i].max_x = self.position[0] + np.cos(sensor_theta) * self.sensor_depth
            else:
                self.sensor_bounding_boxes[i].min_x = self.position[0] + np.cos(sensor_theta) * self.sensor_depth
                self.sensor_bounding_boxes[i].max_x = self.position[0]
            if sensor_theta < PI:
                self.sensor_bounding_boxes[i].min_y = self.position[1]
                self.sensor_bounding_boxes[i].max_y = self.position[1] + np.sin(sensor_theta) * self.sensor_depth
            else:
                self.sensor_bounding_boxes[i].min_y = self.position[1] + np.sin(sensor_theta) * self.sensor_depth
                self.sensor_bounding_boxes[i].max_y = self.position[1]
