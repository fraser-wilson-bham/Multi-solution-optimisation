import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import bc_functions2 as bc_functions
from evo import Evolution, Optimization


# --- Derating Function ---
def derating_function(s, o, o_r, alpha=2):
    distance = np.linalg.norm(s - o)
    if distance < o_r:
        return (distance / o_r)**alpha
    else:
        return 1

# --- Modified Fitness Function ---
def modified_fitness(x, derating_functions, score_func):
    original_fitness = score_func(x)
    modified_score = -original_fitness

    for df in derating_functions:
        modified_score *= derating_function(x, df["centre"], df["radius"], alpha=2)
    return modified_score

# --- Custom Individual Class ---
class BCOptimization(Optimization):
    def __init__(self, value=None, init_params=None):
        super().__init__(value, init_params)

# --- EvolutionWithSequentialNiching Class ---
class EvolutionWithSequentialNiching(Evolution):
    def __init__(self, pool_size, fitness, individual_class, n_offsprings,
                 pair_params, mutate_params, init_params, niche_radius):
        super().__init__(pool_size, fitness, individual_class, n_offsprings,
                         pair_params, mutate_params, init_params)
        self.niche_radius = niche_radius
        self.derating_functions = []
        self.original_fitness = fitness

    def step(self):
        mothers, fathers = self.pool.get_parents(self.n_offsprings)
        offsprings = []

        for mother, father in zip(mothers, fathers):
            offspring = mother.pair(father, self.pair_params)
            offspring.mutate(self.mutate_params)
            offsprings.append(offspring)

        combined_individuals = self.pool.individuals + offsprings

        shared_population = []
        for ind in combined_individuals:
            modified_fit = modified_fitness(ind.value, self.derating_functions, lambda x: -self.original_fitness(BCOptimization(x)))
            shared_population.append((ind, modified_fit))

        shared_population.sort(key=lambda x: x[1], reverse=True)
        selected_individuals = [ind for ind, _ in shared_population[:len(self.pool.individuals)]]

        # --- Optima Identification ---
        for ind in combined_individuals:
            is_duplicate = False
            for df in self.derating_functions:
                if np.linalg.norm(ind.value - df["centre"]) < self.niche_radius * 0.5:
                    is_duplicate = True
                    if self.original_fitness(ind) < df["score_best"]:
                        df["centre"] = ind.value
                        df["score_best"] = self.original_fitness(ind)
                        df["score"] = modified_fitness(ind.value, self.derating_functions, lambda x: -self.original_fitness(BCOptimization(x)))

                    break

            if not is_duplicate:
                score_best = self.original_fitness(ind)
                self.derating_functions.append({
                    "centre": ind.value,
                    "radius": self.niche_radius,
                    "score_best": score_best,
                    "score": modified_fitness(ind.value, self.derating_functions, lambda x: -self.original_fitness(BCOptimization(x))),
                })

        self.pool.individuals = selected_individuals


    def get_found_optima(self):
        return self.derating_functions

# --- Plotting function ---
def plot_results_ea_sn(found_optima, bounds, bc_function, min_f, max_f, title_suffix=""):

    x = np.linspace(bounds[0][0], bounds[0][1], 50)
    y = np.linspace(bounds[1][0], bounds[1][1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.array([bc_function([xi, yi]) for xi, yi in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
    Z = (Z - min_f) / (max_f - min_f)

    local_minima = np.array([opt["centre"] for opt in found_optima])
    minima_z_values = np.array([opt["score_best"] for opt in found_optima])
    minima_z_values_normalized = 1 - ((minima_z_values - min_f) / (max_f - min_f))

    # --- 3D Plot ---
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    function_surface = ax_3d.plot_surface(X, Y, 1-Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none', alpha=0.3)
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z')
    ax_3d.view_init(60, 35)
    ax_3d.set_zlim(0, 1)
    ax_3d.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_3d.set_zticklabels(['1.0', '0.8', '0.6', '0.4', '0.2', '0.0'])
    if len(local_minima) > 0:
        ax_3d.scatter([loc[0] for loc in local_minima],
                      [loc[1] for loc in local_minima],
                      minima_z_values_normalized,
                      c='red', s=20, depthshade=False)
    ax_3d.set_title(f'3D Plot of Function and Minima of {title_suffix} using EA:SN')
    fig_3d.tight_layout()

    # --- Contour Plot ---
    fig_contour, ax_contour = plt.subplots(figsize=(6, 6))
    contour = ax_contour.contourf(X, Y, 1-Z, 20, cmap=cm.viridis)
    if len(local_minima) > 0:
        ax_contour.scatter(local_minima[:, 0], local_minima[:, 1], c='red', marker='o', s=50)
    ax_contour.set_xlabel('X')
    ax_contour.set_ylabel('Y')
    ax_contour.set_aspect('equal')
    ax_contour.set_title(f'Contour Plot of {title_suffix} using EA:SN')
    #fig_contour.colorbar(contour, ax=ax_contour)
    fig_contour.tight_layout()

    #plt.show()
    plt.close(fig_3d)
    plt.close(fig_contour)

# --- Run Evolution ---
def run_evolution_sn(bc_function_name, population_size, n_offsprings, generations, bounds, niche_radius):
    bc_func, normalize_func, scale_factor, original_bounds, n_dims = bc_functions.get_function(bc_function_name)
    if bc_func is None:
        print(f"Error: Function '{bc_function_name}' not found.")
        return

    min_f, max_f = normalize_func()

    def adapted_fitness(individual):
        return bc_func(individual.value)

    init_params = {'lower_bound': bounds[0][0], 'upper_bound': bounds[1][1], 'dim': len(bounds)}
    mutate_params = {'lower_bound': bounds[0][0], 'upper_bound': bounds[1][1], 'rate': 0.1, 'dim': len(bounds)}
    pair_params = {'alpha': 0.7}

    evolution = EvolutionWithSequentialNiching(
        pool_size=population_size,
        fitness=adapted_fitness,  # Original fitness function
        individual_class=BCOptimization,
        n_offsprings=n_offsprings,
        pair_params=pair_params,
        mutate_params=mutate_params,
        init_params=init_params,
        niche_radius=niche_radius
    )

    for generation in range(generations):
        evolution.step()
        if (generation + 1) % 50 == 0:
            found_optima = evolution.get_found_optima()
            if len(found_optima) > 0:
                scaled_minima = [
                    [opt['centre'][0] * scale_factor[0], opt['centre'][1] * scale_factor[1]]
                    for opt in found_optima
                ]
                original_minima_values = [bc_func(np.array([scaled_minima[i][0], scaled_minima[i][1]])) for i in range(len(scaled_minima))] # Calculate original function value at the minima

                print(f"Generation {generation + 1}, Function: {bc_function_name}, Original Local Minima Coordinates: {scaled_minima}, Minima values are: {original_minima_values}")

            else:
                print(f"Generation {generation + 1}, Function: {bc_function_name}, No Local Minima Found (niche={niche_radius})")

    found_optima = evolution.get_found_optima()
    plot_results_ea_sn(found_optima, bounds, bc_func, min_f, max_f, title_suffix=f"{bc_function_name}")

    return found_optima, bounds, bc_func, min_f, max_f