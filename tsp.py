# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:30:39 2021

@author: Pjoter
"""

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.problems.single.traveling_salesman import create_random_tsp_problem
from pymoo.util.termination.default import SingleObjectiveDefaultTermination

import numpy as np
import pandas as pd
from pymoo.core.repair import Repair

num_cities = 20
pop_size = 50
prob_mut = 0.3
prob_crossover = 0.7
gen_lim = 10000//pop_size

# Import data
cities_coordinates = pd.read_csv("CitiesXY.csv", sep=',').iloc[0:num_cities,1:3]
print("Cities:\n", cities_coordinates.head(), "\n")

cost_matrix = pd.read_csv("CityDistCar.csv", sep = ',').iloc[0:num_cities, 1:num_cities+1]
print("Cost matrix on form:\n", cost_matrix.head(), "\n")

class StartFromZeroRepair(Repair):

    def _do(self, problem, pop, **kwargs):
        X = pop.get("X")
        I = np.where(X == 0)[1]

        for k in range(len(X)):
            i = I[k]
            x = X[k]
            _x = np.concatenate([x[i:], x[:i]])
            pop[k].set("X", _x)

        return pop

problem = create_random_tsp_problem(30, 100, seed=1)

algorithm = GA(
    pop_size=20,
    sampling=get_sampling("perm_random"),
    crossover=get_crossover("perm_erx"),
    mutation=get_mutation("perm_inv"),
    repair=StartFromZeroRepair(),
    eliminate_duplicates=True
)

# if the algorithm did not improve the last 200 generations then it will terminate (and disable the max generations)
termination = SingleObjectiveDefaultTermination(n_last=200, n_max_gen=np.inf)

res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
)

print("Traveling Time:", np.round(res.F[0], 3))
print("Function Evaluations:", res.algorithm.evaluator.n_eval)

from pymoo.problems.single.traveling_salesman import visualize
visualize(problem, res.X)