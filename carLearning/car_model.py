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

class Car():
    '''a car is a very simple neural network with 2 hidden layers of size 4,3, the input size is 5 and the output
    size is 2 the input are 5 values corresponding to the distance of sight in a given direction. (in front of the car,
     and 4 other directions (2 at right and 2 at left) the output are 2 values, one in the range (0,1) corresponding to
     the engine power (i don't want cars to move backward because if let's them hack the perf function the other output
     value is a float number in [-1,1] corresponding to the derivative of the angle of the car i.e the angular speed of
     the car. This number is multiplied later by some coefficient to be in an expected natural range. (a variation of 1
      gradient in 0.1 second is not natural and allows models to do loops)'''
    color = BLUE
    def __init__(self):
        self.mutation_scaler = 0.05
        self.W1=np.random.rand(4, 5) - 0.5
        self.b1=np.random.rand(4, 1) - 0.5
        self.W2=np.random.rand(3, 4) - 0.5
        self.b2=np.random.rand(3, 1) - 0.5
        self.W3=np.random.rand(2, 3) - 0.5
        self.b3=np.random.rand(2, 1) - 0.5
        self.radius = physics_engine.universal_scaler * 20
        self.LOS_depth = physics_engine.universal_scaler * 100
        self.number_of_LOS = 5
        self.theta_angle =  30 * TWO_PI / 360
        self.reset()
        self.LOS_colors = self.number_of_LOS * [(125, 60, 10)]
    def reset(self):
        self.LOS_e_thetas = np.zeros((5,2))
        self.LOS_projection = np.zeros((5,2))
        self.X = np.zeros((5, 1))
        self.X.fill(self.LOS_depth)
        self.position = physics_engine.universal_scaler * np.array([100., 100.])
        self.speed = np.array([0., 0.])
        self.acceleration = np.array([0., 0.])
        self.theta = 0.
        self.LOS_thetas = np.zeros((5))
        self.update_LOS_thetas()
        self.LOS_BBs = [collision_detection.BoundingBox(0, 0, 0, 0) for i in range(self.number_of_LOS)]
        self.update_LOS_thetas()
        self.e_theta = mu.get_e_theta(self.theta)
        self.collided = False
        self.score = 0.
    def update_LOS_projection(self):
        for i in range(self.number_of_LOS):
            self.LOS_projection[i] = self.position + self.X[i] * self.LOS_e_thetas[i]
    def display_LOS(self, screen):
        for i in range(self.number_of_LOS):
            if self.X[i] < self.LOS_depth:
                pygame.draw.circle(screen,(0, 150, 75) , self.LOS_projection[i] , 5)
            else:
                pygame.draw.circle(screen, self.LOS_colors[i], self.LOS_projection[i] , 3)
    def update_LOS_thetas(self):
        for i in range(self.number_of_LOS):
            self.LOS_thetas[i] = (self.theta + (i - (self.number_of_LOS - 1) / 2) * self.theta_angle) % TWO_PI
            self.LOS_e_thetas[i : i + 1] = mu.get_e_theta(self.LOS_thetas[i], self.LOS_e_thetas[i])
    def get_line_of_sight(self, lines):
        self.X.fill(self.LOS_depth)
        for line in lines:
            for LOSS_n in range(self.number_of_LOS):
                if collision_detection.BB_collision_detection(self.LOS_BBs[LOSS_n], line):
                    if collision_detection.collide_vector_line(self.position, self.LOS_e_thetas[LOSS_n], self.X[LOSS_n], line):
                        M = mu.cross_lines(self.position, self.LOS_e_thetas[LOSS_n], line.A, line.v)
                        distance = mu.distance(M, self.position)
                        assert(distance <= self.X[LOSS_n] + 1)
                        self.X[LOSS_n] = distance
        self.update_LOS_projection()
    def update_score(self):
        if not self.collided:
            self.score += physics_engine.deltaT * mu.norm(self.speed) + 0.5
    def feed(self): 
        '''this function allows to compute the output of the neuron using the self.X array (input) which should be set before'''
        self.h1 = np.tanh(np.dot(self.W1, self.X / self.LOS_depth) - self.b1)
        self.h2 = np.tanh(np.dot(self.W2, self.h1) - self.b2)
        self.h3 = np.tanh(np.dot(self.W3, self.h2) - self.b3)
        self.acceleration = expit(self.h3[1][0])
        self.theta_prime = np.tanh(self.h3[0][0])
    def copy(self):
        clone = Car()
        clone.W1 = self.W1.copy()
        clone.W2 = self.W2.copy()
        clone.W3 = self.W3.copy()
        return clone
    def mutate(self):
        '''mutation, replace a given weight per another'''
        i=np.random.randint(3)
        j=np.random.randint(2)
        if i==0 and j==0:
            layer=self.W1
            self.W1[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])]=np.random.uniform(-1,1,1)
        if i==0 and j==1:
            layer=self.b1
            self.b1[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])]=np.random.uniform(-1,1,1)
        if i==1 and j==0:
            layer=self.W2
            self.W2[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])]=np.random.uniform(-1,1,1)
        if i==1 and j==1:
            layer=self.b2
            self.b2[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])]=np.random.uniform(-1,1,1)
        if i==2 and j==0:
            layer=self.W3
            self.W3[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])]=np.random.uniform(-1,1,1)
        if i==2 and j==1:
            layer=self.b3
            self.b3[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])]=np.random.uniform(-1,1,1)
    def tiny_mutate(self):
        '''mutation, modify a weight'''
        i=np.random.randint(3)
        j=np.random.randint(2)
        if i==0 and j==0:
            layer=self.W1
            self.W1[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])] += self.mutation_scaler * np.random.uniform(-1,1,1)
        if i==0 and j==1:
            layer=self.b1
            self.b1[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])] += self.mutation_scaler * np.random.uniform(-1,1,1)
        if i==1 and j==0:
            layer=self.W2
            self.W2[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])] += self.mutation_scaler * np.random.uniform(-1,1,1)
        if i==1 and j==1:
            layer=self.b2
            self.b2[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])] += self.mutation_scaler * np.random.uniform(-1,1,1)
        if i==2 and j==0:
            layer=self.W3
            self.W3[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])] += self.mutation_scaler * np.random.uniform(-1,1,1)
        if i==2 and j==1:
            layer=self.b3
            self.b3[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])] += self.mutation_scaler * np.random.uniform(-1,1,1)
    def crossover(parent1, parent2):
        '''returns an offspring for parrent1 and parent2'''
        child=Car()
        crossOverPoint=np.random.randint(1,3)
        child.W1 = 1 * parent1.W1
        child.b1 = 1 * parent1.b1
        if crossOverPoint==1:
            child.W2 = 1 * parent2.W2
            child.b2 = 1 * parent2.b2
        else:
            child.W2 = 1 * parent2.W2
            child.b2 = 1 * parent2.b2
        child.W3 = 1 * parent2.W3
        child.b3 = 1 * parent2.b3
        return child
    def update_cos_sin(self):
        self.cos, self.sin = np.cos(self.theta), np.sin(self.theta)
    def get_BB(self):
        '''computes the corners of the vehicles given there center M and there orientations.'''
        self.min_x, self.max_x = self.position[0] - self.radius, self.position[0] + self.radius
        self.min_y, self.max_y = self.position[1] - self.radius, self.position[1] + self.radius
    def display(self, screen):
        pygame.draw.circle(screen,self.color , self.position, self.radius)
        pygame.draw.line(screen, (0, 0, 0), self.position, self.radius * self.e_theta + self.position)
    def update_BB(self):
        self.min_x = self.position[0] - self.radius
        self.max_x = self.position[0] + self.radius
        self.min_y = self.position[1] - self.radius
        self.max_y = self.position[1] + self.radius
    def update(self, lines):
        self.update_LOS_BB()
        self.get_line_of_sight(lines)
        self.update_score()
        self.feed()
    def update_LOS_BB(self):
        '''update line of sight BB'''
        for i, LOS_theta in enumerate(self.LOS_thetas):
            if LOS_theta < PI_OVER_TWO or LOS_theta > THREE_PI_OVER_TWO:
                self.LOS_BBs[i].min_x = self.position[0]
                self.LOS_BBs[i].max_x = self.position[0] + np.cos(LOS_theta) * self.LOS_depth
            else:
                self.LOS_BBs[i].min_x = self.position[0] + np.cos(LOS_theta) * self.LOS_depth
                self.LOS_BBs[i].max_x = self.position[0]
            if LOS_theta < PI:
                self.LOS_BBs[i].min_y = self.position[1]
                self.LOS_BBs[i].max_y = self.position[1] + np.sin(LOS_theta) * self.LOS_depth
            else:
                self.LOS_BBs[i].min_y = self.position[1] + np.sin(LOS_theta) * self.LOS_depth
                self.LOS_BBs[i].max_y = self.position[1]
