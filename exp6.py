import random
import math
import matplotlib.pyplot as plt
import numpy as np

# Define the fitness function (sine function with two inputs)
def fitness_function(x, y):
    return math.sin(x) + math.sin(y)

# Initialize the population
def initialize_population(pop_size, lower_bound, upper_bound):
    return [(random.uniform(lower_bound, upper_bound),
             random.uniform(lower_bound, upper_bound)) for _ in range(pop_size)]

# Select parents based on their fitness
def select_parents(population):
    total_fitness = sum(fitness_function(x, y) for x, y in population)
    roulette_wheel = [fitness_function(x, y) / total_fitness for x, y in population]
    parent1 = random.choices(population, weights=roulette_wheel)[0]
    parent2 = random.choices(population, weights=roulette_wheel)[0]
    return parent1, parent2

# Perform crossover to create a new generation
def crossover(parent1, parent2, crossover_prob=0.7):
    if random.random() < crossover_prob:
        child1 = (parent1[0], parent2[1])
        child2 = (parent2[0], parent1[1])
        return child1, child2
    else:
        return parent1, parent2

# Perform mutation in the population
def mutate(individual, mutation_prob=0.01):
    x, y = individual
    if random.random() < mutation_prob:
        x += random.uniform(-0.1, 0.1)
    if random.random() < mutation_prob:
        y += random.uniform(-0.1, 0.1)
    return x, y

# Genetic Algorithm
def genetic_algorithm(generations, pop_size, lower_bound, upper_bound):
    population = initialize_population(pop_size, lower_bound, upper_bound)
    best_fitness_history = []
    for gen in range(generations):
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = new_population[:pop_size]  # keep size fixed
        best_individual = max(population, key=lambda ind: fitness_function(*ind))
        best_fitness = fitness_function(*best_individual)
        best_fitness_history.append(best_fitness)
        print(f"Generation {gen+1}: Best individual = {best_individual}, "
              f"Fitness = {best_fitness}")
    return max(population, key=lambda ind: fitness_function(*ind)), best_fitness_history

# Run the algorithm
if __name__ == "__main__":
    generations = 50
    pop_size = 100
    lower_bound = -2 * math.pi
    upper_bound = 2 * math.pi

    best_solution, fitness_history = genetic_algorithm(generations, pop_size,
                                                       lower_bound, upper_bound)
    print(f"\nBest solution found: {best_solution}, "
          f"Fitness = {fitness_function(*best_solution)}")

    # -------- Plot the results --------
    # Plot fitness history (convergence curve)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, generations+1), fitness_history, marker='o', color='blue')
    plt.title("Genetic Algorithm Convergence on f(x,y) = sin(x) + sin(y)")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    plt.show()
