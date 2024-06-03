import argparse
import numpy as np
from exercices import get_mask
import exercices as ex
from time import time
import sudoku_lib as sl
import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("-o", help="origin_csv")
parser.add_argument("-d", help="destination_csv")
parser.add_argument("-f", help="friendly input")
parser.add_argument("-s", help="string option, gives the string you want to solve")
parser.add_argument("-v", help="do you want to display every solution", action="store_true")
parser.add_argument("-evaluate", help="do you want an evaluation", action="store_true")
args = parser.parse_args()


def evaluator(successes, fails):
    if fails + successes == 0:
        return "no grid solved yet"
    else:
        return successes / (fails + successes)


failed_grids = 0
succeeded_grids = 0

if __name__ == "__main__":
    if args.s is not None:
        try:
            assert (len(args.s) == 81)
        except:
            raise Exception(f"input string length should be 81, received {len(args.s)}")
        grid = np.zeros((9, 9), np.uint32)
        sl.fill_grid_with_s(grid, args.s)
        grid = ex.createM(np.transpose(grid))
        sl.plot(grid)
        worked, grid = sl.roll(grid, 5)
        sl.plot(grid)
    if args.f is not None:
        grid = sl.load_friendly(args.f)
        sl.plot(grid)
        worked, grid = sl.roll(grid, 5)
        sl.plot(grid)
    if args.d is not None and args.o is not None:
        N = np.zeros((9, 9), np.uint32)
        try:
            assert not os.path.exists(args.d)
        except:
            raise Exception("destination file already exists, please chose a non existing destination path")
        with open(args.d, 'w') as destination_file:
            with open(args.o, "r") as origin_file:
                destination_file.write("quizzes, solution" + '\n')
                for i, line in tqdm.tqdm(enumerate(origin_file)):
                    if i == 0:
                        continue
                    s = line[:81]
                    sl.fill_grid_with_s(N, s)
                    grid = ex.createM(np.transpose(N))
                    if args.v:
                        sl.plot(grid)
                    worked, grid = sl.roll(grid, 5)
                    if args.v:
                        sl.plot(grid)
                        input('please press enter for next grid')
                    dest_file.write(s + ',' + sl.from_grid_to_s(grid) + '\n')
    else:
        if args.o is not None:
            N = np.zeros((9, 9), np.uint32)
            with open(args.o, "r") as origin_file:
                for i, line in tqdm.tqdm(enumerate(origin_file)):
                    if i == 0:
                        continue
                    s = line[:81]
                    sl.fill_grid_with_s(N, s)
                    grid = ex.create_grid(np.transpose(N))
                    if args.v:
                        sl.plot(grid)
                    worked, grid = sl.roll(grid, 5)
                    readable_grid = ex.from_grid_to_array(grid)
                    if ex.check(readable_grid):
                        succeeded_grids += 1
                    else:
                        failed_grids += 1
                    if args.v:
                        sl.plot(grid)
                        input('please press enter for next grid')
            print(f"success rate:{evaluator(succeeded_grids, failed_grids)}")
