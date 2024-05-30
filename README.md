# Genetic Algorithm for Function Optimization

This repository contains an implementation of a Genetic Algorithm (GA) for optimizing a specific function. The GA is used to minimize the function g(x) by finding the optimal values of the variables x₁ and x₂.

## Overview

The goal of this project is to demonstrate the application of a Genetic Algorithm to optimize a given mathematical function. The function to be optimized is defined as:


g(x) = ∑ ([1.5, 2.25, 2.625] - x₁ + x₁ * (x₂^[1, 2, 3]))²

The fitness of an individual is calculated as:

fitness(x) = 1 / (g(x) + 1)


The GA aims to find the values of x₁ and x₂ that minimize the function g(x), thereby maximizing the fitness.

## Implementation Details

The code is structured as follows:

- **decode_chromosome**: Decodes a binary chromosome into real-valued variables.
- **cross**: Performs crossover between two parent individuals to produce offspring.
- **evaluate_individual**: Evaluates the fitness of an individual.
- **initialize_population**: Initializes a population of individuals randomly.
- **mutate**: Performs mutation on an individual.
- **run_function_optimization**: Main function that runs the genetic algorithm.
- **tournament_select**: Implements tournament selection to select an individual based on fitness.

## Parameters

The parameters for the genetic algorithm are specified in the script and include:

- `number_of_runs`: Number of runs for the batch experiments.
- `population_size`: Number of individuals in the population.
- `maximum_variable_value`: Maximum absolute value for the variables.
- `number_of_genes`: Number of genes in each chromosome.
- `number_of_variables`: Number of variables to be optimized.
- `number_of_generations`: Number of generations to run the algorithm.
- `tournament_size`: Size of the tournament for selection.
- `tournament_probability`: Probability of selecting the better individual in the tournament.
- `crossover_probability`: Probability of performing crossover.
- `mutation_probabilities`: List of mutation probabilities to be tested.

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/genetic-algorithm-optimization.git
    cd genetic-algorithm-optimization
    ```

2. **Run the script**:
    ```bash
    python GA.py
    ```

3. **View the results**:
    The script will output the fitness and variable values for each run, and plot the median fitness values as a function of mutation probability.


A plot will be generated showing the median performance as a function of mutation probability.

## Dependencies

- numpy
- matplotlib

Install the dependencies using:
```bash
pip install numpy matplotlib

