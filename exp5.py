import random
import matplotlib.pyplot as plt

# Define the fitness function (objective function to maximize)
def fitness_function(x):
    return -x**2 + 6*x + 9

# Initialize the population
def initialize_population(pop_size, lower_bound, upper_bound):
    return [random.uniform(lower_bound, upper_bound) for _ in range(pop_size)]

# Select parents using roulette wheel selection
def select_parents(population):
    total_fitness = sum(fitness_function(ind) for ind in population)
    roulette_wheel = [fitness_function(ind) / total_fitness for ind in population]
    parent1 = random.choices(population, weights=roulette_wheel)[0]
    parent2 = random.choices(population, weights=roulette_wheel)[0]
    return parent1, parent2

# Perform crossover to create children
def crossover(parent1, parent2, crossover_prob=0.7):
    if random.random() < crossover_prob:
        child1 = (parent1 + parent2) / 2
        child2 = (parent1 + parent2) / 2
        return child1, child2
    else:
        return parent1, parent2

# Perform mutation
def mutate(individual, mutation_prob=0.01):
    if random.random() < mutation_prob:
        individual += random.uniform(-1, 1)
    return individual

# Genetic Algorithm main function
def genetic_algorithm(generations, pop_size, lower_bound, upper_bound):
    population = initialize_population(pop_size, lower_bound, upper_bound)
    best_fitness_per_gen = []

    for gen in range(generations):
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population[:pop_size]  # Keep population size fixed
        best = max(population, key=fitness_function)
        best_fitness = fitness_function(best)
        best_fitness_per_gen.append(best_fitness)

        print(f"Generation {gen+1}: Best individual = {round(best, 4)}, Fitness = {round(best_fitness, 4)}")

    return best, best_fitness_per_gen

# Run the algorithm
if __name__ == "__main__":
    generations = 50
    pop_size = 100
    lower_bound = -10
    upper_bound = 10

    best_solution, fitness_history = genetic_algorithm(generations, pop_size, lower_bound, upper_bound)

    print(f"\nBest solution found: x = {round(best_solution, 4)}, Fitness = {round(fitness_function(best_solution), 4)}")

    # Plotting the fitness over generations
    plt.plot(fitness_history, marker='o', color='blue', label='Best Fitness')
    plt.title('Genetic Algorithm Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
