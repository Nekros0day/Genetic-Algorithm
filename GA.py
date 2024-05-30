import numpy as np
import matplotlib.pyplot as plt

def decode_chromosome(chromosome, number_of_variables, maximum_variable_value):
    n_genes = len(chromosome)
    k = n_genes // number_of_variables
    x = np.zeros(number_of_variables)

    for i in range(number_of_variables):
        gene_start = i * k
        gene_end = (i + 1) * k
        gene_section = chromosome[gene_start:gene_end]
        power_factors = 2.0 ** -np.arange(1, k + 1)
        
        x[i] = np.dot(gene_section, power_factors)
        x[i] = -maximum_variable_value + 2 * maximum_variable_value * x[i] / (1 - 2 ** (-k))
    
    return x

def cross(individual1, individual2):
    n_genes = len(individual1)
    crossover_point = 1 + np.random.randint(0, n_genes - 1)
    
    new_individual1 = np.zeros(n_genes, dtype=int)
    new_individual2 = np.zeros(n_genes, dtype=int)
    
    new_individual1[:crossover_point] = individual1[:crossover_point]
    new_individual2[:crossover_point] = individual2[:crossover_point]
    new_individual1[crossover_point:] = individual2[crossover_point:]
    new_individual2[crossover_point:] = individual1[crossover_point:]
    
    return new_individual1, new_individual2

def evaluate_individual(x):
    terms = [1.5, 2.25, 2.625] - x[0] + x[0] * (x[1] ** np.array([1, 2, 3]))
    g_x = np.sum(terms ** 2)
    fitness = 1 / (g_x + 1)
    return fitness

def initialize_population(population_size, number_of_genes):
    return np.random.rand(population_size, number_of_genes) > 0.5

def mutate(individual, mutation_probability):
    n_genes = len(individual)
    mutated_individual = np.copy(individual)
    
    mutate_indices = np.random.rand(n_genes) < mutation_probability
    mutated_individual[mutate_indices] = 1 - individual[mutate_indices]
    
    return mutated_individual

def run_function_optimization(population_size, number_of_genes, number_of_variables, maximum_variable_value, 
                              tournament_size, tournament_probability, crossover_probability, mutation_probability, number_of_generations):
    maximum_fitness = 0.0
    population = initialize_population(population_size, number_of_genes)
    best_variable_values = None

    for generation in range(number_of_generations):
        fitness_list = np.zeros(population_size)
        for i in range(population_size):
            chromosome = population[i]
            variable_values = decode_chromosome(chromosome, number_of_variables, maximum_variable_value)
            fitness = evaluate_individual(variable_values)
            fitness_list[i] = fitness
            if fitness > maximum_fitness:
                maximum_fitness = fitness
                i_best_individual = i
                best_variable_values = variable_values

        temporary_population = np.copy(population)
        for i in range(0, population_size, 2):
            i1 = tournament_select(fitness_list, tournament_probability, tournament_size)
            i2 = tournament_select(fitness_list, tournament_probability, tournament_size)
            if np.random.rand() < crossover_probability:
                individual1 = population[i1]
                individual2 = population[i2]
                new_individual_pair = cross(individual1, individual2)
                temporary_population[i] = new_individual_pair[0]
                temporary_population[i+1] = new_individual_pair[1]
            else:
                temporary_population[i] = population[i1]
                temporary_population[i+1] = population[i2]

        temporary_population[0] = population[i_best_individual]
        for i in range(1, population_size):
            temporary_population[i] = mutate(temporary_population[i], mutation_probability)
        
        population = temporary_population

    return maximum_fitness, best_variable_values

def tournament_select(fitness_list, tournament_probability, tournament_size):
    population_size = len(fitness_list)
    selected_indexes = np.random.choice(population_size, tournament_size, replace=False)
    sorted_fitnesses = np.argsort(fitness_list[selected_indexes])[::-1]
    
    selected_individual_index = selected_indexes[sorted_fitnesses[-1]]  # Default: worst individual is selected
    for i in range(tournament_size):
        if np.random.rand() < tournament_probability:
            selected_individual_index = selected_indexes[sorted_fitnesses[i]]  # Better individual is selected based on probability
            break  # Exit once a selection is made
    
    return selected_individual_index

# Parameter specifications
number_of_runs = 100                # Do NOT change
population_size = 100               # Do NOT change
maximum_variable_value = 5          # Do NOT change (x_i in [-a,a], where a = maximum_variable_value)
number_of_genes = 50                # Do NOT change
number_of_variables = 2             # Do NOT change
number_of_generations = 300         # Do NOT change
tournament_size = 2                 # Do NOT change
tournament_probability = 0.75       # Do NOT change
crossover_probability = 0.8         # Do NOT change

# Batch runs
mutation_probabilities = [0, 0.01, 0.02, 0.04, 0.08, 0.1, 0.3, 0.5, 0.7, 1.0]

# Initialize medians array
medians = np.zeros(len(mutation_probabilities))

for m_idx, mutation_probability in enumerate(mutation_probabilities):
    print(f'Mutation rate = {mutation_probability:.5f}')
    
    maximum_fitness_list = np.zeros(number_of_runs)
    
    for i in range(number_of_runs):
        maximum_fitness, best_variable_values = run_function_optimization(
            population_size, number_of_genes, number_of_variables, maximum_variable_value, tournament_size,
            tournament_probability, crossover_probability, mutation_probability, number_of_generations)
        
        print(f'Run: {i+1}, Score: {maximum_fitness:.10f}')
        
        maximum_fitness_list[i] = maximum_fitness
    
    # Compute statistics
    average = np.mean(maximum_fitness_list)
    median_val = np.median(maximum_fitness_list)
    std_val = np.sqrt(np.var(maximum_fitness_list))
    print(f'PMut = {mutation_probability:.2f}: Median: {median_val:.10f}, Average: {average:.10f}, STD: {std_val:.10f}')
    
    # Store the median value
    medians[m_idx] = median_val

# Plot results
plt.plot(mutation_probabilities, medians, '-ok', markerfacecolor='r')
plt.xlabel('pMut')
plt.ylabel('Median fitness')
plt.yscale('log')
plt.xlim([0, 1.2])
plt.ylim([0, 1.001])
plt.title('Median performance as a function of pMut')
plt.show()

