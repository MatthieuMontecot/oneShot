import argparse
import numpy as np
from exercices import getMask
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
args = parser.parse_args()

if __name__ == "__main__":
    if not args.s is None:
        try:
            assert(len(args.s) == 81)
        except:
            raise Exception(f"input string length should be 81, received {len(args.s)}" )
        M = np.zeros((9,9), np.uint32)
        sl.fill_M_with_s(M, args.s)
        M = ex.createM(np.transpose(M))
        sl.plot(M)
        worked, M = sl.roll(M,5)
        sl.plot(M)
    if not args.f is None:
        M = sl.load_friendly(args.f)
        sl.plot(M)
        worked, M = sl.roll(M,5)
        sl.plot(M)
    if not args.d is None and not args.o is None:
        N = np.zeros((9, 9), np.uint32)
        try:
            assert not os.path.exists(args.d)
        except:
            raise Exception("destination file already exists, please chose a non existing destination path")
        with open(args.d, 'w') as dest_file:
            with open(args.o, "r") as origin_file:
                dest_file.write("quizzes, solution" + '\n')
                for i, line in tqdm.tqdm(enumerate(origin_file)):
                    if i == 0:
                        continue
                    s = line[:81]
                    sl.fill_M_with_s(N, s)
                    M = ex.createM(np.transpose(N))
                    if args.v:
                        sl.plot(M)
                    worked, M = sl.roll(M,5)
                    if args.v:
                        sl.plot(M)
                        input('please press enter for next grid')
                    dest_file.write(s + ',' + sl.from_M_to_s(M) + '\n')
    else:
        if not args.o is None:
            N = np.zeros((9, 9), np.uint32)
            with open(args.o, "r") as origin_file:
                for i, line in enumerate(origin_file):
                    if i == 0:
                        continue
                    s = line[:81]
                    sl.fill_M_with_s(N, s)
                    M = ex.createM(np.transpose(N))
                    if args.v:
                        sl.plot(M)
                    worked, M = sl.roll(M,5)
                    if args.v:
                        sl.plot(M)
                        input('please press enter for next grid')
            
