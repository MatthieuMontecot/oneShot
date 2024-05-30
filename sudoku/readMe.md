# sudoku project sum up:
This is a project about sudoku solving I made in one night. I took sudoku unsolved and solutions
from the https://www.kaggle.com/bryanpark/sudoku?select=sudoku.csv page which consist of 1,000,000 
grids. Solving the entire database took me 3H30 with the solver.py scipt with 100% success rate. 

# how to use
## solve a given sudoku grid as a string:

python solver.py -s 040100050107003960520008000000000017000906800803050620090060543600080700250097100 <br>

please adapt the string to the problem

## solve a given sudoku stored in a file which looks like this:

0,0,0,0,0,0,2,0,0<br>
0,8,0,0,0,7,0,9,0<br>
6,0,2,0,0,0,5,0,0<br>
0,7,0,0,6,0,0,0,0<br>
0,0,0,9,0,1,0,0,0<br>
0,0,0,0,2,0,0,4,0<br>
0,0,5,0,0,0,6,0,3<br>
0,9,0,4,0,0,0,7,0<br>
0,0,6,0,0,0,0,0,0<br>

use the -f option (for friendly file) with the corresponding path: <br>

python solver.py -f friendly_input.txt

## solve the bryanpark database:

first download the database on https://www.kaggle.com/bryanpark/sudoku?select=sudoku.csv and unzip it,

### if you want to print each grid without saving it:

python solver.py -o sudoku.csv <br>

(replace sudoku.csv by the corresponding path)

### if you want to solve the database and print the outputs in a destination file:

python solver.py -o sudoku.csv -d solution.csv

if you want to check by looking at it a solution in this output file, please use:<br>
python solver.py -s 040100050107003960520008000000000017000906800803050620090060543600080700250097100 <br>
which will consider the string as a grid to solve and display it in a better way<br>
(please adapte the given string)

### if you want to see the solutions one by one while saving them in a file:

python solver.py -v -o sudoku.csv -d solution.csv

# additional comments:

The way a sudoku grid is encoded before solving is a numpy array of dimension (3,3,3,3,9) named M and containing either zeros or ones. In facts, a sudoku grid can be seen as a collection of 3x3 boxes arranges in a 3x3 maner. The first 4 dimensions corresponding to (box abscisse in the overall grid, box ordinate of the grid in the overall grid, slot abscisse in the overall grid, slot ordinate of the grid in the overall grid, corresponding number). The last dimension correspond to numbers in the grid, first layer correspond to ones, until the last to 9. A one in the last layer means that the 9 is possible. This means that if the very top left slot of the grid can, with current knowledge be a 3 a 4 or a 9, the vector M[0,0,0,0] (top left slot) will be [0,0,1,1,0,0,0,0,1].
