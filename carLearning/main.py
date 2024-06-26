import numpy as np
import pygame
import sys
import time
import pickle
import argparse
import level_design as ld
import display_engine as de
import car_model
import genetic_algorithm as ga
import physics_engine
import collision_detection


parser = argparse.ArgumentParser()
parser.add_argument("-e", help="number of epochs")
parser.add_argument("-hide", help="do you want to avoid displaying?", action="store_true")
parser.add_argument("-v", help="verbose", action="store_true")
parser.add_argument("-t", help="maximum time allowed, this values is increased at each epoch")
parser.add_argument("-l", help="loading_file to load a population")
parser.add_argument("-s", help="saving_file to save a population")
args = parser.parse_args()


frame = 0
t0 = time.time()

if args.e is None:
    args.e = 100
else:
    args.e = int(args.e)
if args.t is None:
    args.t = 1000
FPS = 80
if not args.hide:
    screen = de.get_screen()
    clock = de.get_clock()
else:
    screen = None
    clock = None

if __name__ == '__main__':
    lines = ld.level1()
    if args.l is not None:
        with open(args.l, 'rb') as input_file:
            population = pickle.load(input_file)
    else:
        population = ga.Population(car_model.Car)
    for e in range(args.e):
        if args.v and e > 0:
            print(f"{frame / (time.time() - t0)} frames per second with {population.n} cars and {len(lines)} lines")
            if frame > args.t:
                score_scale_factor = (1 + ((e - 1) / 20))
            else:
                score_scale_factor = 1
            print(
                f"""mean score (divided by the increasing epoch duration): 
                    {population.average_score / score_scale_factor}""")
            print(
                f"""max score (divided by the increasing epoch duration):
                    {population.scores.max() / score_scale_factor}""")
            population.reset()
            population.selection_pressure += 0.1
        print(f'generation {e}')
        if args.hide is None:
            clock.tick(FPS)
        frame = 0
        t0 = time.time()
        while frame < args.t * (1 + (e / 20)) and False in [pop_element.collided for pop_element in
                                                            population.population]:
            if not args.hide:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if args.s:
                            with open(args.s, 'wb') as output_file:
                                pickle.dump(population, output_file, pickle.HIGHEST_PROTOCOL)
                        de.quit_display()
                        sys.exit()
            frame += 1
            for pop_element in population.population:
                if not pop_element.collided:
                    pop_element.update_bounding_box()
                    for line_element in lines:
                        collision_detection.collide_car_line(pop_element, line_element)
                if not pop_element.collided:
                    pop_element.update_sensor_bounding_box()
                    pop_element.get_line_of_sight(lines)
                    pop_element.update_score()
                    pop_element.feed()
                    physics_engine.update(pop_element)
            if not args.hide:
                clock.tick(FPS)
                pygame.display.update()
                screen.fill(de.WHITE)
                for pop_element in population.population:
                    pop_element.display(screen)
                    pop_element.display_sensor(screen)
                for line_element in lines:
                    line_element.display(screen)
        population.update_scores()
        population.update_average_score()
        population.evolve()

if args.s:
    with open(args.s, 'wb') as output_file:
        pickle.dump(population, output_file, pickle.HIGHEST_PROTOCOL)
