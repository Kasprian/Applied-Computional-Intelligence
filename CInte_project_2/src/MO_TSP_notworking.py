# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:30:39 2021

@author: Pjoter
"""

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.factory import  get_crossover, get_mutation, get_sampling, get_termination

import numpy as np
import pandas as pd
import random
from pymoo.core.repair import Repair
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
import autograd.numpy as anp

# Declaring global variables
num_cities = 20
pop_size = 50
prob_mut = 0.3
prob_crossover = 0.7
gen_lim = 10000//pop_size


#Definition of problem
class TravelingSalesman(ElementwiseProblem):

    def __init__(self, cities, discCar, costCar, discAirPlane, costAirPlane, **kwargs):

        self.cities = cities
        self.distCar = discCar
        self.costCar = costCar
        self.distAirplane = discAirPlane
        self.costAirplane = costAirPlane

        super().__init__(
            n_var=2,
            n_obj=2,
            xl=np.array([0,0]),
            xu=np.array([cities,1]),
            **kwargs
        )

#Evaluation function
    def _evaluate(self, x, out, *args, **kwargs):
        print(x)
        print(x.shape)
        out['Disc'], out['Cost'] = self.get_route_length(x)

#Counting length
    def get_route_length(self, x):
        n_cities = len(x)
        dist = 0
        cost = 0
        for k in range(n_cities - 1):
            i, j = x[0][k], x[0][k + 1]
            if x[1][k] == 0:
                dist += self.discCar[i, j]
                cost += self.costCar[i, j]
            else:
                dist += self.discAirPlane[i, j]
                cost += self.costAirplane[i, j]
        last, first = x[0][-1], x[0][0]
        if x[1][last] == 0:
            dist += self.discCar[last, first]
            cost += self.costCar[last, first]
        else:
            dist += self.discAirPlane[last, first]
            cost += self.costAirplane[last, first]
            
        return dist,cost
    
    
#Custom sampling
class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        form = [random.randint(0,1) for _ in range(num_cities)]
    
        individual = np.arange(num_cities)
        random.shuffle(individual)
        return np.vstack((individual, form))


# Read the data
cities_coordinates = pd.read_csv("data/CitiesXY.csv", sep=',').iloc[0:num_cities,1:3]
discCar = pd.read_csv("data/CityDistCar.csv", sep = ',').iloc[0:num_cities, 1:num_cities+1]
costCar = pd.read_csv("data/CityDistCar.csv", sep = ',').iloc[0:num_cities, 1:num_cities+1]
discAirPlane = pd.read_csv("data/CityDistCar.csv", sep = ',').iloc[0:num_cities, 1:num_cities+1]
costAirPlane = pd.read_csv("data/CityDistCar.csv", sep = ',').iloc[0:num_cities, 1:num_cities+1]


problem = TravelingSalesman(20, discCar, costCar, discAirPlane, costAirPlane)

algorithm = NSGA2(
    pop_size=20,
    sampling=get_sampling("int_random"),
    crossover=get_crossover("perm_erx"),
    mutation=get_mutation("perm_inv"),
#    repair=StartFromZeroRepair(),
 #   eliminate_duplicates=True
)

# if the algorithm did not improve the last 200 generations then it will terminate (and disable the max generations)
termination = get_termination("n_gen", 40)

res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    save_history=True,
    verbose=True
)


print(res)

