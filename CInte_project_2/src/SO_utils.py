import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List

def init_individual(num_cities):
    individual = np.arange(num_cities)
    random.shuffle(individual)
    return individual

def plot_sol(citiesXY: pd.DataFrame, solution: List):

    # Define a label_counter to keep track of labels
    label_counter = 0

    # Lists for the points of solution
    x, y = [], []

    # Add the points of the solution
    for city in solution:
        x.append(citiesXY.iloc[city,0])
        y.append(citiesXY.iloc[city,1])
    
    # Add the first points to the end to get the loop
    x.append(citiesXY.iloc[solution[0],0])
    y.append(citiesXY.iloc[solution[0],1])

    # Define a colormap to cover the interesting solutions    
    colors = iter(plt.cm.YlOrRd(np.linspace(0, 1, len(x))))

    # Iterate through each point, making lines between them to illustrate the 
    # progression of the solution
    x_prev, y_prev = np.NaN, np.NaN

    for x_elem, y_elem, c in zip(x, y, colors):
        # Increase the label
        label_counter += 1

        # Plot the points of the scatterplot
        # If on element 1 or at the end of the solutions, make legend for orientation
        if label_counter == 1: 
            plt.scatter(x_elem, y_elem, color=c, label = "First iteration")
        elif label_counter == len(solution)+1:
            plt.scatter(x_elem, y_elem, color=c, label = "Last iteration")
        else:
            plt.scatter(x_elem, y_elem, color=c)

        if x_prev != np.NaN and y_prev != np.NaN:
            plt.plot([x_prev, x_elem], [y_prev, y_elem], color = c)
        x_prev, y_prev = x_elem, y_elem

    # 
    plt.legend(loc=2, prop={'size': 4})
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Path of final solution")
    plt.grid()
    plt.show()

def init_heuristic_individual(citiesXY: pd.DataFrame, init_pop: List):
    # Define the x_value on which to split the dataframe
    split = 500
    
    # Clone the first individual of the population
    heuristic_ind = init_pop[0]

    # Generate the list of cities
    # Start by keeping tab of which cities are related to which coordinates
    citiesXY["cities"] = citiesXY.index

    # Split df into two dataframes of cities on left and right of the split-value
    cities_left = citiesXY.where(citiesXY["x"] <= split).dropna()
    cities_right = citiesXY.where(citiesXY["x"] > split).dropna()

    # Sort the left from lowest y to highest
    cities_left.sort_values(by=["y"], ascending=True, inplace=True)
    # Sort the right from highest y to lowest
    cities_right.sort_values(by=["y"], ascending=False, inplace=True)

    # Generate the new list by adding the cities from the left, then the right
    data = cities_left["cities"].tolist() + cities_right["cities"].tolist()
    data = [int(x) for x in data]

    # Generate and return the new population:
    heuristic_ind[0] = np.array(data, dtype='int32')

    init_pop[0] = heuristic_ind
    
    return init_pop

def evaluate_TSP(cost_matrix: pd.DataFrame, pop: List[List]):
    pop_eval = []

    for ind in pop:
        sum = cost_matrix.iloc[ind[0][0], ind[0][len(ind[0])-1]]
        for city in range(1, len(ind[0])):
            sum += cost_matrix.iloc[ind[0][city-1], ind[0][city]]

        pop_eval.append(sum)
    return pop_eval, 


def subplots_TSP(citiesXY: pd.DataFrame, sol_list: List, num_gen:int):
    # Define a colormap to cover the interesting solutions
    color = iter(plt.cm.YlOrRd(np.linspace(0, 1, len(sol_list))))

    # Define a counter to keep track of correct labels
    label_counter = 0

    # Plot the cities
    plt.scatter(x = citiesXY.iloc[:,0], y=citiesXY.iloc[:,1])
    
    # Iterate through the solutions
    for sol in sol_list:
        label_counter += 1

        # Lists for the points of solutions 
        x, y = [], []

        # Add the points of each solution
        for city in sol[0]:
            x.append(citiesXY.iloc[city,0])
            y.append(citiesXY.iloc[city,1])
        
        # Add the first points to the end to get the loop
        x.append(citiesXY.iloc[sol[0][0],0])
        y.append(citiesXY.iloc[sol[0][0],1])

        # Plot the solution with the correct color/label
        c = next(color)
        label = ('Gen ' + str(int(label_counter*(num_gen/len(sol_list)))) + 
                ' of ' + str(int(num_gen)))
        plt.plot(x,y, c = c, label = label)

    # Define legends and show figure
    plt.legend(loc=2, prop={'size': 4})
    plt.grid()
    plt.show()

def pop_metrics(pop: List, prints:bool = False, decimation:int = 3):
    # Extract length of pop and population fitnesses
    length_pop = len(pop)
    fitnesses = [ind.fitness.values[0] for ind in pop]
    
    # Find the mean fitness of the population
    mean = sum(fitnesses) / length_pop
    
    # Find the standard deviation of fitnesses within the population
    sum2 = sum(x*x for x in fitnesses)
    std = abs(sum2 / length_pop - mean**2)**0.5
    
    # Find the min/max values of the population
    max_pop = max(fitnesses)
    min_pop = min(fitnesses)

    # If the function is set to print, print the metrics
    if prints:
        print("  Min %s" % round(min_pop, decimation))
        print("  Max %s" % round(max_pop, decimation))
        print("  Avg %s" % round(mean, decimation))
        print("  Std %s" % round(std, decimation))

    # If not, return them
    else:
        return mean, std, max_pop, min_pop

def gen_convergence_curve(all_solutions: List, mean_list: List = None):
    # Get the fitnesses of all individuals
    fitnesses = []
    for ind in all_solutions:
        fitnesses.append(ind.fitness.values[0])
    # Get the number of generations
    indexes = np.arange(0,len(fitnesses),1).tolist()

    # Plot the curve
    plt.plot(indexes, fitnesses, label='Best individual')
    if mean_list is not None: 
        plt.plot(indexes, mean_list, label='Population mean')
        
    plt.xlabel("Generations")
    plt.ylabel("Fitness of optimal solution")
    plt.title("Convergence curve")
    plt.legend(loc=2, prop={'size': 4})
    plt.grid()
    plt.show()