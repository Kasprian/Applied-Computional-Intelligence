import time
import random

import numpy as np
import pandas as pd

from deap import base, creator, tools

from SO_utils import init_individual, evaluate_TSP, subplots_TSP, plot_sol, gen_convergence_curve, pop_metrics, init_heuristic_individual

# Declaring global variables
num_cities = 20 # max = 50
pop_size = 40   # size of the population
prob_crossover = 0.7   # probability of a crossover occuring
prob_mut = 0.15  # probability of a mutation occurring
prob_mut_individual_elements = 0.05   # Probablility of a mutation occurring on a specific element
tournament_size = 3     # Amount of individuals per tournament

use_heuristic_solution = True  # wether or not to initialize population with a heuristic solution
use_elite_selection = False     # wether or not to keep the best individual between generations

gen_lim = 10000//pop_size   # max generations, decided by max 10.000 evaluations

# Declaring list of seeds for the main_loop
num_seeds = 1
seed_list = np.arange(0,num_seeds,1)

# Import data
cities_coordinates = pd.read_csv("data\CitiesXY.csv", sep=',').iloc[0:num_cities,1:3]
print("Cities:\n", cities_coordinates.head(), "\n")

cost_matrix = pd.read_csv("data\CityDistCar.csv", sep = ',').iloc[0:num_cities, 1:num_cities+1]
print("Cost matrix on form:\n", cost_matrix.head(), "\n")

# Define creators for DEAP library
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Define the DEAP toolbox
toolbox = base.Toolbox()

# Declare the Individual and population
toolbox.register("ind_declaration", init_individual, num_cities)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.ind_declaration, 1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=pop_size)

# Register fitness-function, crossover operator and mutation operator
toolbox.register("evaluate", evaluate_TSP, cost_matrix)
toolbox.register("crossover", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes)
toolbox.register("select", tools.selTournament, tournsize=tournament_size)

def run(heuristic:bool, elitist:bool, print_runtime:bool = False):
    # Get the start time of the algorithm
    start = time.time()

    # Define lists to save individuals: 
    all_sol_list = []   # The best of every generation
    mean_list = []      # The mean of every generation
    
    # Declare the initial population (with heuristic individual) and evaluate its fitness
    pop = toolbox.population()
    if heuristic:
        pop = init_heuristic_individual(cities_coordinates, pop)

    fitnesses = toolbox.evaluate(pop=pop)
    for elem in range(0, len(pop)):
        ind = pop[elem]
        fit = fitnesses[0][elem]
        ind.fitness.values = fit,
    
    # Start evolving the solution through generations
    gen = 0

    while gen < gen_lim:
        gen += 1

        # Select parents through chosen selection_scheme and generate offspring
        # Initially, offspring are chosen to be their parents before crossover/mutations
        # are applied 
        parents = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, parents))

        # Make offspring differ from parents by applying crossovers and mutations
        # Applying crossover: 
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability prob_crossover
            if random.random() < prob_crossover:
                toolbox.crossover(child1[0], child2[0])

                # Delete existing fitnesses of the crossed offspring
                del child1.fitness.values
                del child2.fitness.values
        
        # Applying mutations: 
        for mutant in offspring:
            # mutate an individual with probability prob_mut
            if random.random() < prob_mut:
                toolbox.mutate(mutant[0], indpb = prob_mut_individual_elements)

                # Delete existing fitness of the mutated offspring
                del mutant.fitness.values

        # Evaluate offspring without assigned fitnesses
        # Offspring w.o fitness are crossed/mutated
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.evaluate(pop=invalid_ind)
        for elem in range(0, len(invalid_ind)):
            ind = invalid_ind[elem]
            fit = fitnesses[0][elem]
            ind.fitness.values = fit,

        # Replace the population with the evolved offspring
        pop[:] = offspring

        # Calculate fitnesses of the new population
        fitnesses = [ind.fitness.values for ind in pop]
        # Extract the best individual
        best_ind = tools.selBest(pop, 1)[0]

        # If elitist selection, keep the best individual for the next generation
        if elitist:
            pop[0] = best_ind
        
        # Save the best individual of the generation
        all_sol_list.append(best_ind)
        
        # Get and save the mean of the generation        
        mean, _, _, _ = pop_metrics(pop)
        mean_list.append(mean)

    # Get metrics of the last generation
    final_mean, final_std, _, _ = pop_metrics(pop)

    end = time.time()
    if print_runtime:
        print("===========================\nTotal runtime:"
                        , round(end-start, 4), "seconds")
    
    return final_mean, final_std, mean_list, all_sol_list

def main():
    # Generate lists of all means and standard deviations
    mean_list_all_seeds, std_list_all_seeds = [], []

    # Generate variables to keep the best results between all seeds
    best_mean = None
    best_mean_list, best_sol_list = [], []
    
    # Iterate through all randomizer seeds, running the GA for each of them and 
    # returning the mean, std, 
    for seed in seed_list:
        print("On seed ", seed)
        random.seed(seed)
        seed_final_mean, seed_final_std, seed_mean_list, seed_all_sol = run(heuristic=use_heuristic_solution, 
                                                                elitist=use_elite_selection,
                                                                print_runtime=False)

        if best_mean is None or seed_final_mean < best_mean:
            best_mean = seed_final_mean
            best_mean_list = seed_mean_list
            best_sol_list = seed_all_sol

        mean_list_all_seeds.append(seed_final_mean)
        std_list_all_seeds.append(seed_final_std)
    
    # Plot the convergence curve of the best mean among the seeds
    gen_convergence_curve(best_sol_list, best_mean_list)

    # Plot the best solution of the best mean among the seeds
    plot_sol(cities_coordinates, best_sol_list[-1][0])

    # Print the average mean and std over all the seeds
    tot_mean = np.mean(mean_list_all_seeds)
    tot_std = np.mean(std_list_all_seeds)
    print("For ", num_cities, " cities with ", num_seeds, "seeds:")
    print("average mean:", tot_mean, "- average STD:",  tot_std)

main()