import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures

def generalized_rosenbrock(x):
    generalized_rosenbrock.counter += 1
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def whitley(x):
    whitley.counter += 1
    x = np.asarray(x)
    d = len(x)
    i = np.arange(1, d)
    term1 = 100 * (x[:-1]**2 - x[1:])**2
    term2 = (1 - x[:-1])**2
    term3 = np.cos(x[:-1] / np.sqrt(i))
    result = np.sum(term1 + term2 / (4000.0 - np.prod(term3)))
    return result

def salomon(x):
    salomon.counter += 1
    D = len(x)
    sum_term = np.sum(np.cos(2 * np.pi * x[:D-1] / np.sqrt(np.arange(1, D))) ** 6)
    return 1 - np.cos(2 * np.pi * np.sqrt(np.sum(x ** 2))) + 0.1 * np.sqrt(sum_term)

def initialize_population(population_size, dimension, lower_bound, upper_bound):
    return np.random.uniform(low=lower_bound, high=upper_bound, size=(population_size, dimension))

def selection(population, fitness_values, num_parents):
    selected_indices = np.argsort(fitness_values)[:num_parents]
    return population[selected_indices]

def crossover(parents, offspring_size):
    offspring = np.zeros(offspring_size)
    crossover_point = np.random.randint(1, offspring_size[1])
    for i in range(offspring_size[0]):
        parent1_idx = i % parents.shape[0]
        parent2_idx = (i+1) % parents.shape[0]
        offspring[i, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[i, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutate(offspring, mutation_rate):
    for i in range(offspring.shape[0]):
        for j in range(offspring.shape[1]):
            if np.random.rand() < mutation_rate:
                offspring[i, j] += np.random.uniform(-1, 1)
    return offspring

def replace_population(population, offspring, fitness_values, func):
    combined_population = np.vstack([population, offspring])
    combined_fitness = np.hstack([fitness_values, evaluate_fitness(offspring, func)])
    indices = np.argsort(combined_fitness)[:population.shape[0]]
    return combined_population[indices], combined_fitness[indices]

def evaluate_fitness(population, func):
    return np.array([func(ind) for ind in population])

def evolutionary_algorithm(population_size, dimension, num_generations, num_parents, mutation_rate, target_budget, quality_thresholds, lower_bound, upper_bound, func):
    population = initialize_population(population_size, dimension, lower_bound, upper_bound)
    func.counter = 0
    fitness_values = evaluate_fitness(population, func)
    steps = []
    budgets = []
    best_fitness_values = []
    steps_at_budget = np.zeros(41)
    budget_thresholds = 10.0 ** np.arange(0, 4.1, 0.1) * dimension

    for generation in range(num_generations):
        budget = int(func.counter)
        if budget >= target_budget:
            break 
        parents = selection(population, fitness_values, num_parents)
        offspring = crossover(parents, (population_size - num_parents, dimension))
        mutated_offspring = mutate(offspring, mutation_rate)
        population, fitness_values = replace_population(population, mutated_offspring, fitness_values, func)

        best_fitness = np.min(fitness_values)
        best_fitness_values.append(best_fitness)
        actual_step = np.argmax(best_fitness >= np.array(quality_thresholds))
        steps.append(actual_step)
        steps_at_budget[func.counter <= budget_thresholds] = actual_step
        print(f"budget {budget}, generation {generation + 1}")
        budgets.append(budget)
    
    max_step_crossed = max(steps)
    best_solution = population[np.argmin(fitness_values)]
    best_fitness = np.min(fitness_values)
    return steps, budgets, max_step_crossed, best_fitness, steps_at_budget


def perform_experiments(num_experiments, population_size, dimension, num_generations, num_parents, mutation_rate, target_budget, quality_thresholds, lower_bound, upper_bound, func):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                evolutionary_algorithm,
                population_size,
                dimension,
                num_generations,
                num_parents,
                mutation_rate,
                target_budget,
                quality_thresholds,
                lower_bound,
                upper_bound,
                func
            )
            for _ in range(num_experiments)
        ]
        normalized_steps = []
        max_steps_crossed = np.zeros(num_experiments)
        all_budgets = []
        best_fitnesses = np.zeros(num_experiments)
        final_steps_at_budget = np.zeros(41)
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            steps, budget, max_step_crossed, best_fitness, steps_at_budget = future.result()
            normalized_steps.append(steps)
            max_steps_crossed[i] = max_step_crossed
            all_budgets.append(budget)
            best_fitnesses[i] = best_fitness
            final_steps_at_budget += steps_at_budget
    final_steps_at_budget /= num_experiments
    max_budgets_length = max(len(budget) for budget in all_budgets)
    all_budgets = [budget + [0] * (max_budgets_length - len(budget)) for budget in all_budgets]
    budgets = np.mean(all_budgets, axis=0)
    max_steps_length = max(len(steps) for steps in normalized_steps)
    normalized_steps = [steps + [0] * (max_steps_length - len(steps)) for steps in normalized_steps]
    normalized_steps = np.mean(normalized_steps, axis=0, dtype=np.float64)  # średnia wyników
    normalized_steps = np.array(normalized_steps, dtype=np.float64)
    np.savetxt(f"{func.__name__}_{dimension}.csv", final_steps_at_budget, fmt='%.15f',delimiter=",")
    budgets = remove_zeros_after_half(budgets)
    max_steps = 51
    mapped_steps = [step / max_steps for step in normalized_steps]
    return mapped_steps, budgets, int(max(max_steps_crossed)), min(best_fitnesses)

def remove_zeros_after_half(arr):
    length = len(arr)
    half_index = length // 2

    index_to_remove = half_index
    while index_to_remove < length and arr[index_to_remove] == 0:
        index_to_remove += 1

    new_arr = arr[:index_to_remove]

    return new_arr

def plot_normalized_step_at_target_budget(num_experiments, population_size, dimension, num_generations, num_parents, mutation_rate, target_budget, quality_thresholds, lower_bound, upper_bound, func):
    normalized_steps, budgets, max_step_crossed, best_fitness = perform_experiments(num_experiments, population_size,
                                                                                   dimension, num_generations,
                                                                                   num_parents, mutation_rate,
                                                                                   target_budget, quality_thresholds,
                                                                                   lower_bound, upper_bound, func)

def plot_optimization_landscape(func, lower_bound, upper_bound):
    x_range = np.linspace(lower_bound, upper_bound, 100)
    y_range = np.linspace(lower_bound, upper_bound, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')
    ax.set_title(f'Optimization Landscape for {func.__name__}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')
    plt.show()

if __name__ == '__main__':
    #functions = [whitley, salomon, generalized_rosenbrock]
    functions = [whitley]
    #functions = [salomon]
    #functions = [generalized_rosenbrock]
    dimensions = [10, 30, 50]
    quality_thresholds = []
    x = 2
    while x >= -8.2:
        quality_thresholds.append(10**x)
        x -= 0.2
    for func in functions:
        for dim in dimensions:
            if func == generalized_rosenbrock:
                lower_bound = -30
                upper_bound = 30
                plot_normalized_step_at_target_budget(num_experiments=100,
                                                      population_size=1000, 
                                                      dimension=dim, 
                                                      num_generations=150 * (dim//10), 
                                                      num_parents=20, 
                                                      mutation_rate=0.1, 
                                                      target_budget=(10000 * dim), #np.log10(10**4 * dim)
                                                      quality_thresholds=quality_thresholds, 
                                                      lower_bound=lower_bound, 
                                                      upper_bound=upper_bound, 
                                                      func=func)
            elif func == whitley:
                lower_bound = -10.24
                upper_bound = 10.24
                plot_normalized_step_at_target_budget(num_experiments=1,
                                                      population_size=1000, 
                                                      dimension=dim, 
                                                      num_generations=150 * (dim//10), 
                                                      num_parents=20, 
                                                      mutation_rate=0.1, 
                                                      target_budget=(10000 * dim), 
                                                      quality_thresholds=quality_thresholds,
                                                      lower_bound=lower_bound,
                                                      upper_bound=upper_bound,
                                                      func=func)
            elif func == salomon:
                lower_bound = -100
                upper_bound = 100
                plot_normalized_step_at_target_budget(num_experiments=100,
                                                      population_size=1000,
                                                      dimension=dim, 
                                                      num_generations=150 * (dim//10), 
                                                      num_parents=20, 
                                                      mutation_rate=0.1,
                                                      target_budget=(10000 * dim), 
                                                      quality_thresholds=quality_thresholds,
                                                      lower_bound=lower_bound,
                                                      upper_bound=upper_bound,
                                                      func=func)