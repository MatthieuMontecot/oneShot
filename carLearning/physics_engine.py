import numpy as np
import car_model
import math_utils as mu
deltaT = 1
friction_coefficient = 0.1
theta_scaler = 0.1
TWO_PI = 2 * np.pi
universal_scaler = 0.6
def update(object, thetaPmax=0.1):
    '''this updates the physics of the gave i.e the position, speed and orientations of the vehicles given there acceleration and angular momentums,
    end previous positions speed and orientations. Also, we use the friction parameter which compensate the acceleration proportionaly to the speed and prevent
    to have a higher speed that a certain limit. Also, we update the performance of all vehicles by the distance they drove at this step, minus a constant at each iteration to penalize slow solutions.'''
    object.theta = (object.theta + deltaT * object.theta_prime * theta_scaler) % TWO_PI
    object.update_LOS_thetas()
    object.e_theta = mu.get_e_theta(object.theta, object.e_theta)
    #object.speed += deltaT * object.acceleration * object.e_theta - friction_coefficient * object.speed
    object.speed = 10 * deltaT * object.acceleration * object.e_theta
    object.position += universal_scaler * deltaT * object.speed