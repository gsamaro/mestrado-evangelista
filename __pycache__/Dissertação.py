import pandas as pd
import numpy as np
from dataclasses import dataclass
from docplex.mp.model import Model
from read_file import LerDados

myInput =[[8, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 3, 6, 0, 0, 0, 0, 0],
 [0, 7, 0, 0, 9, 0, 2, 0, 0],
 [0, 5, 0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 4, 5, 7, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 3, 0],
 [0, 0, 1, 0, 0, 0, 0, 6, 8],
 [0, 0, 8, 5, 0, 0, 0, 1, 0],
 [0, 9, 0, 0, 0, 0, 4, 0, 0]]

model = Model("sudoku")
R = range(1, 10)
idx = [(i, j, k) for i in R for j in R for k in R]

x = model.binary_var_dict(idx, "X")

for i in R:
    for j in R:
        if myInput[i - 1][j - 1] != 0:
            model.add_constraint(x[i, j, myInput[i - 1][j - 1]] == 1)

for i in R:
    for j in R:
        model.add_constraint(model.sum(x[i, j, k] for k in R) == 1)
for j in R:
    for k in R:
        model.add_constraint(model.sum(x[i, j, k] for i in R) == 1)
for i in R:
    for k in R:
        model.add_constraint(model.sum(x[i, j, k] for j in R) == 1)

solution = model.solve()
solution.print_information()


