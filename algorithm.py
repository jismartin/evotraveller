import numpy as np
import pandas as pd
import random

# TSP parameters
grid_size = 10
n_places = 9

# Places to visit
def places(grid_size,n_places):
    points = list(zip(np.random.randint(0,grid_size,n_places),np.random.randint(0,grid_size,n_places)))
    while (0,0) in points:
        points = list(zip(np.random.randint(0,grid_size,n_places),np.random.randint(0,grid_size,n_places)))
    return points

# Fitness function
def distance(sequence):
    path=[(0,0)] + sequence + [(0,0)]
    return sum(np.sqrt((path[i][0] - path[i+1][0])**2 + (path[i][1] - path[i+1][1])**2) 
               for i in range(len(path)-1))
def fitness(solution):
    return 1 / distance(solution) # shorter distence higher fitness

# Intial Population
def initial_population(places,population_size):
    return [random.sample(places, len(places)) for _ in range(population_size)]

# Selection (random tournament)
def random_tournament(population, tournament_size):
    selected_for_next_round = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=fitness)
        selected_for_next_round.append(winner)
    return selected_for_next_round

# Crossover (ox crossover)
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

def add_extremes(sequence):
    return [(0,0)] + sequence + [(0,0)]

# Genetic algortihm
def genetic_algorithm(targets, population_size, generations, tournament_size, mutation_rate, elitism,crossover):
    population = initial_population(targets,population_size)
    best_solution=max(population, key=fitness)
    fitness_evolution=[]
    fitness_evolution.append(fitness(max(population, key=fitness)))
    
    # Keep simulating generations
    df_simulation = pd.DataFrame({'c%i'%i:[add_extremes(population[i])] for i in range(population_size)})

    e = int(elitism*population_size)
    e = e if (e % 2 == 0) else e+1 # size of elitism: even number
    nc = int(crossover*population_size) 
    nc = nc if (nc % 2 == 0) else nc+1 # size of crossover: even number

    for t in range(generations):
        population = sorted(population,key=fitness)
        population.reverse()

        # (1) elite (no mutation)
        next_generation = population[:e]

        # Selection
        selection = random_tournament(population, tournament_size)
        parents=random.sample(selection, len(selection))
        
        # (2) direct descendants with mutation
        for i in range(population_size-nc-e):
            parent = parents.pop()
            next_generation.append(mutate(parent,mutation_rate))

        # (3) descendants with croosover and mutation
        for i in range(int(nc/2)):
            parent1 = parents.pop()
            parent2 = parents.pop()
            child1, child2 = ox_crossover(parent1, parent2)
            next_generation.append(mutate(child1,mutation_rate))
            next_generation.append(mutate(child2,mutation_rate))
 
        population = next_generation

        if best_solution not in population:
            print('Best solution lost!')

        fitness_evolution.append(fitness(max(population, key=fitness)))
        if fitness(max(population, key=fitness)) > fitness(best_solution):
            best_solution=max(population, key=fitness) 

        population = sorted(population,key=fitness)
        population.reverse()
        df_simulation=pd.concat([df_simulation,
                                 pd.DataFrame({'c%i'%i:[add_extremes(population[i])] for i in range(population_size)})],ignore_index=True)

    return add_extremes(best_solution), fitness_evolution, df_simulation