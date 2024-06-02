import numpy as np
import car_model
import math_utils as mu

deltaT = 1
friction_coefficient = 0.1
theta_scale_factor = 0.1
TWO_PI = 2 * np.pi
universal_scale_factor = 0.6


def update(instance):
    """this updates the physics of the gave i.e. the position, speed and orientations of the vehicles given there
    acceleration and angular momentum's, end previous positions speed and orientations. Also, we use the friction
    parameter which compensate the acceleration proportionally to the speed and prevent to have a higher speed that
    a certain limit. Also, we update the performance of all vehicles by the distance they drove at this step, minus
    a constant at each iteration to penalize slow solutions."""
    instance.theta = (instance.theta + deltaT * instance.theta_prime * theta_scale_factor) % TWO_PI
    instance.update_sensor_thetas()
    instance.e_theta = mu.get_e_theta(instance.theta, instance.e_theta)
    # instance.speed += deltaT * instance.acceleration * instance.e_theta - friction_coefficient * instance.speed
    instance.speed = 10 * deltaT * instance.acceleration * instance.e_theta
    instance.position += universal_scale_factor * deltaT * instance.speed
