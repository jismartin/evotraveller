import numpy as np
import pandas as pd
import random
import copy
# Be carefull, in Python lists are passed by reference; a list of list is a list of references to lists

# TSP hyperparameters
grid_size = 10

# Set seed
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Places to visit (return a list of locations of the TSP)
def places(grid_size,n_places):
    points = list(zip(np.random.randint(0,grid_size,n_places),np.random.randint(0,grid_size,n_places)))
    while (0,0) in points:
        points = list(zip(np.random.randint(0,grid_size,n_places),np.random.randint(0,grid_size,n_places)))
    return points

# Add the origin and destination to the sequence-list
def add_extremes(sequence):
    return [(0,0)] + sequence + [(0,0)]

# Fitness function (works for one sequence-list)
def distance(sequence):
    path=add_extremes(sequence)
    return sum(np.sqrt((path[i][0] - path[i+1][0])**2 + (path[i][1] - path[i+1][1])**2) 
               for i in range(len(path)-1))
def fitness(solution):
    return 1 / distance(solution) # shorter distence higher fitness

# Intial Population (return a list of lists!)
def initial_population(places,population_size):
    return [random.sample(places, len(places)) for _ in range(population_size)]

# Selection (random tournament)
def random_tournament(population, tournament_size):
    selected_for_next_round = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=fitness)
        selected_for_next_round.append(winner.copy()) # Be careful, a copy of a list to get a new solution
    return selected_for_next_round

# Crossover (ox crossover) returns two offspring solutions from two parents
# Ox ccopy
def ox_copy(parent1,parent2,start,end):
    size = min(len(parent1), len(parent2))
    child=parent1.copy() # Lists are always passed by reference in Phyton!
    tmp=parent2.copy()
    tmp.reverse()
    for place in child[start:end]:
        tmp.remove(place)
    for j in range(end,size):
        child[j] = tmp.pop()
    for j in range(0,start):
        child[j] = tmp.pop()
    return child

# Generate offsprings
def ox_crossover(parent1, parent2):
    size = min(len(parent1), len(parent2))
    start, end = sorted(random.sample(range(size), 2))
    child1=ox_copy(parent1,parent2,start,end)
    child2=ox_copy(parent2,parent1,start,end)
    return child1, child2

# Mutation
def mutate(solution, mutation_rate):
    size=len(solution)
    if random.random() < mutation_rate:
        # swap two random positions
        i,j = random.sample(range(size), 2)
        tmp=solution[i]
        solution[i]=solution[j]
        solution[j]=tmp  
    return solution

# Genetic algortihm
def genetic_algorithm(targets, population_size, generations, tournament_size, mutation_rate, elitism,crossover):
    
    # Initialize the population
    population = initial_population(targets,population_size)
    best_solution=max(population, key=fitness)
    fitness_evolution=[fitness(best_solution)]
    
    # Track simulating generations
    df_simulation = pd.DataFrame({'c%i'%i:[add_extremes(population[i])] for i in range(population_size)})

    e = max(1, int(elitism*population_size))
    e = e if (e % 2 == 0) else e + 1 # size of elitism: even number
    nc = int(crossover*population_size) 
    nc = nc if (nc % 2 == 0) else nc + 1 # size of crossover: even number

    for t in range(generations):
        
        # Sort population by fitness in descending order
        population = sorted(population, key=fitness, reverse=True)

        # Create next generation list
        
        # (1) elite (no mutation)
        next_generation = copy.deepcopy(population[:e]) # Keep elite individuals (best solutions)

        # Selection
        selection = random_tournament(population, tournament_size)
        parents=random.sample(selection, len(selection)) # shuffle the selected parents
        
        # (2) direct descendants with mutation
        for i in range(population_size - nc - e):
            parent = parents.pop()
            next_generation.append(mutate(parent,mutation_rate))

        # (3) descendants with croosover and mutation
        for i in range(int(nc/2)):
            parent1 = parents.pop()
            parent2 = parents.pop()
            child1, child2 = ox_crossover(parent1, parent2)
            next_generation.append(mutate(child1,mutation_rate))
            next_generation.append(mutate(child2,mutation_rate))
 
        # Update population
        population = copy.deepcopy(sorted(next_generation, key=fitness, reverse=True))
        fitness_evolution.append(fitness(max(population, key=fitness)))

        # Track simulation data        
        df_simulation=pd.concat([df_simulation,
                                 pd.DataFrame({'c%i'%i:[add_extremes(population[i])] for i in range(population_size)})],ignore_index=True)

    best_solution=max(population, key=fitness)

    return add_extremes(best_solution), fitness_evolution, df_simulation