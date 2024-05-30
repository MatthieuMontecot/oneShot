# carLearning presentation
This project consist of generating a population of self driving cars in some course and make this population evolve to make offsprings
able to do the complete course efficiently. A demo can be viewed on a typical generation after 30 generations.

/!\ I coded this in 2020 during quarantine, I didn't maintain it and it lacks a lot of optimization
Everythin from genetic algorithm, neural network representation, collision detection and physics engine is implemented from scratch
using numpy operation, no more.
This implementation lacks tons of opimizations especially for collision detection, which didn't use bounding boxes and thus lead to a slow solution.
I recommend just having a look at the video I uploaded as a demo

## the demo explanation:
   You will see rectangles representing cars, with crosses representing each car's line of sight. The goal is for the generation to escape the circuit,
   again, due to the absence of optimized collision detection, I didn't use a long circuit, leading to what could be considered as a proof of concept.

# approach explanation

## the genetic algorithm implementation explained

The inputs of each cars NN is 5 numbers corresponding to the distance with the first wall on a given direction (the distance of sight).
The idea is to use simple feed forward neural networks with entry 5, 2 hidden layers of size 4 and 3 and the output layer of size 2,
encoding the acceleration of the vehicle and his angular acceleration.
The selection process consists of taking m parents with a probability proportional to their performances at the power p with p the selection 
pressure (a common approach is to increase the pressure selection with time to let potentially good solutions with bad starts improve).
The offspring generation consists of select a random hidden layer and take weights of parent1 on the left of it and weights from parent2 at the right.
Also, in order to increase the exploration of possibilities, I also added random new models at each generation.
The mutation is done with a probability and replace a random weight with a brand new one.

## collision detection

To solve collision detection, I used a convex polygons property which is '2 convex polygons collide if and only if all parallel projection to the polygons edges
on a 1 dimentional line makes the projections of the 2 polygons overlapping on each other.
I used basic geometry of coordonate systems to solve it by hand. Also, numpy is fast at computation but I chose to not fasten the computation excluding cases when
the cars already crashed. This would for sure improve computational efficiency but would prevent me for grouping calculus in 4 dimentional matrices which is efficient 
in numpy, or this would have forced me to create new matrices at each step when some car crashed. Which wouldn't be efficient. The same is true for preselection of
non colliding cars and lines du to to high distance.
Also, the complexity of the approach is proportional to both the number of lines and the number of vehicles. But it still fast on the examples I worked on, and can be
fasten increasing the deltaT parameter, which correspond to the timeSteps used to update postition speeds and rotation using their derivatives.

The line of sight calculus is done calculating the coordinates of one of the line point and the director vector of this line in the coordonate system of the phi direction
( we move to the coordinate system where the angle of sight is horizontal, then use the director vector of the line to project that point of the line on the horizontal
axis, then, if this point is on the border, is positive and not further than a given treshold, we use his x coordinate as the distance D, else we set it to the treshold.

To solve the evolution of the system, I used the classic Euler method to solve the differential equations. This is pretty basic but ther is no need to use more advance EDO
solving solutions for such a simple system.
So, at each timeStep, 
the position M(t+1)=M(t)+deltaT*Vm(t) (Vm(t) is the speed vector of M)
the speed Vm(t)=Vm(t)+(1-lambda)*deltaT*Am(t)*eThetaV(t)-lambda*Vm(t)
with Vm(t) the speed vector of M, Am his scalar acceleration outputed by the neural network and eThetaV(t) the normalized director vector
of the vehicle direction. Lambda is a scalar friction parameter which encodes friction and prevent cars to reach arbitrary high speed and
also provides a more natural system, avoiding quadratic speeds and setting a natural limit to this speed as for real cars.
The angle theta(t)=theta(t)+deltaT*maxAllowedRotation*thetaPrime(t)
with thetaPrime a scalar outputed per the NN and theta the direction of the car.
 

