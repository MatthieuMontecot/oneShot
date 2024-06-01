# carLearning presentation
This project consist of generating a population of self driving cars in some course and make this population evolve to make offsprings
able to do the complete course efficiently. <br>
A demo can be viewed on a typical generation after 30 generations. /!\ The demo available is from 2020 when I implemented this. I rewrote it almost entirely since to clean the code. Also I took 2 days to implement it in 2020 and two others in 2024 since the goal of this repository is to do quick projects. <br>

Everything from genetic algorithm, neural network representation, collision detection and physics engine is implemented from scratch
using basically numpy operations, no more. Pygame is used only for display even though I'm aware the collision detected system can be implemented with pygame, I just wanted to do the math.

## the demo explanation:
   You will see rectangles representing cars, with crosses representing each car's line of sight. The goal is for the generation to escape the circuit,
   again, due to the absence of optimized collision detection, I didn't use a long circuit, leading to what could be considered as a proof of concept.

# How to use?
first of all please clone the repository

## to load a pretrained population
To use a pretrained population which just started succeeding the circuit, please use <br>
python main.py -l population.pkl <br>
This will allow you to load a pretrained population from the population.pkl file.
## to train a brand new population
To evolve a brand new population, please use <br>
python main.py 

## to save a trained population
To save a population you trained, please use the -s destination_file as follow: <br>
python main.py -s my_population.pkl <br>
Please adapt the filename. Note that it will save the population if you close the pygame window.

# approach explanation
## the genetic algorithm implementation explained
### the model:
The inputs of each cars NN is 5 numbers corresponding to the distance with the first wall on a given direction (the distance of sight).
The idea is to use simple feed forward neural networks with entry 5, 2 hidden layers of size 4 and 3 and the output layer of size 2,
encoding the speed of the vehicle and his angular acceleration. <br>
### the evolving
The selection process consists of taking m parents with a probability proportional to their performances at the power p with p the selection 
pressure (a common approach is to increase the pressure selection with time to let potentially good solutions with bad starts improve).
The offspring generation consists of select a random hidden layer and take weights of parent1 on the left of it and weights from parent2 at the right.
Also, in order to increase the exploration of possibilities, I also added the possibility to add random new models at each generation.
The mutation is done with a probability and replace a random weight with a brand new one.
### the score function
To evolve the population, we want to select the best members as parents. To do so, we need a scoring function. I simply computed the cumulative length traveled by each instance with a malus for crash and a bonus for each survived frame. This approach isn't optimal since if a solution managed to spin inside the circuit it could have the system. This could be the next step.
## collision detection
The collision detection has been implemented from scratch, I used primary boundingbox system to filter out most of the potential collisions, then I used simple linear algebra to detect collisions.
# further improvements
If I improve this project, I will implement a better way to draw circuits, using Bezier Curves and adapt the collision detection system.
