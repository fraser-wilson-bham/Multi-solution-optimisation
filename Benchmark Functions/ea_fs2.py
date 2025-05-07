import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import bc_functions2 as bc_functions
from evo import Evolution, Optimization

def fitness_sharing_vectorized(population, aptitudes, alpha, niche_radius):
    if population.ndim == 1:
        population = population.reshape(1, -1)
    if population.shape[0] == 0: 
        return aptitudes 

    diff = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    if niche_radius > 1e-9:
        sharing_matrix = np.where(distances < niche_radius, 1 - (distances / niche_radius) ** alpha, 0)
    else:
        sharing_matrix = np.zeros_like(distances) 

    np.fill_diagonal(sharing_matrix, 0) 
    sharing_effect = np.sum(sharing_matrix, axis=1)
    
    
    modified_aptitudes = aptitudes - sharing_effect 
    return modified_aptitudes

class BCOptimization(Optimization):
    def __init__(self, value=None, init_params=None):
        super().__init__(value, init_params)
def plot_results(population, fitnesses, bounds, bc_function, min_f, max_f, initial_niche_radius, title_suffix=""):
    
    x = np.linspace(bounds[0][0], bounds[0][1], 50)
    y = np.linspace(bounds[1][0], bounds[1][1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.array([bc_function([xi, yi]) for xi, yi in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
    range_f = max_f - min_f
    if abs(range_f) < 1e-14:
        Z_norm = np.zeros_like(Z) 
    else:
        Z_norm = (Z - min_f) / range_f
    def find_local_minima(niche_radius):
        local_minima_indices = []
        considered_indices = set()
        if not population or not hasattr(population[0], 'value'):
             return []
        if fitnesses is not None and len(fitnesses) == len(population):
            original_fitnesses = fitnesses
        else:
            
            original_fitnesses = np.array([bc_function(ind.value) for ind in population])

        sorted_indices = np.argsort(original_fitnesses)

        population_values = np.array([ind.value for ind in population]) 

        for i in sorted_indices:
            if i in considered_indices:
                continue

            is_local_minimum = True
            neighbors = []

            for j in range(len(population_values)):
                if i != j:
                    distance = np.linalg.norm(population_values[i] - population_values[j])
                    if distance < niche_radius:
                        neighbors.append(j)
                        if original_fitnesses[j] < original_fitnesses[i]:
                            is_local_minimum = False
                            break
            if is_local_minimum:
                local_minima_indices.append(i)
                considered_indices.add(i)
                considered_indices.update(neighbors)
        return local_minima_indices

    niche_radius = initial_niche_radius
    local_minima_indices = find_local_minima(niche_radius)
    while len(local_minima_indices) == 0 and niche_radius > 1e-6:
        niche_radius /= 2
        local_minima_indices = find_local_minima(niche_radius)
    if population and hasattr(population[0], 'value'):
        local_minima = [population[i].value for i in local_minima_indices]
        local_minima = np.array(local_minima) 
    else:
        local_minima = np.array([]) 
    if 'Z_norm' not in locals():
        print("Error: Z_norm not calculated due to normalization issue.")
        return 
    if local_minima.size > 0:
         minima_z_values = np.array([bc_function(ind) for ind in local_minima])
         if abs(range_f) < 1e-14:
             minima_z_values_normalized = np.zeros(len(minima_z_values))
         else:
             minima_z_values_normalized = (minima_z_values - min_f) / range_f
    else:
         minima_z_values_normalized = np.array([])
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    function_surface = ax_3d.plot_surface(X, Y, 1 - Z_norm, rstride=1, cstride=1, cmap='viridis', edgecolor='none', alpha=0.3)
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z (1-normalized score)') 
    ax_3d.view_init(60, 35)
    ax_3d.set_zlim(0, 1)
    ax_3d.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_3d.set_zticklabels(['1.0', '0.8', '0.6', '0.4', '0.2', '0.0']) 
    if local_minima.size > 0: 
        ax_3d.scatter([loc[0] for loc in local_minima],
                      [loc[1] for loc in local_minima],
                      1 - minima_z_values_normalized, 
                      c='red', s=20, depthshade=False)
    ax_3d.set_title(f'3D Plot: {title_suffix} (EA:FS Internal Plot)') 
    fig_3d.tight_layout()
    fig_contour, ax_contour = plt.subplots(figsize=(6, 6))
    contour = ax_contour.contourf(X, Y, 1 - Z_norm, 20, cmap=cm.viridis) 
    if local_minima.size > 0:
        ax_contour.scatter(local_minima[:, 0], local_minima[:, 1], c='red', marker='o', s=50)
    ax_contour.set_xlabel('X')
    ax_contour.set_ylabel('Y')
    ax_contour.set_aspect('equal')
    ax_contour.set_title(f'Contour Plot: {title_suffix} (EA:FS Internal Plot)') 
    fig_contour.colorbar(contour, ax=ax_contour, label='1 - Normalized Score')
    fig_contour.tight_layout()

    plt.show()
    plt.close(fig_3d)
    plt.close(fig_contour)
class EvolutionWithSharing(Evolution):
    def __init__(self, pool_size, fitness, individual_class, n_offsprings, pair_params, mutate_params, init_params, niche_radius, alpha):
        super().__init__(pool_size, fitness, individual_class, n_offsprings, pair_params, mutate_params, init_params)
        self.niche_radius = niche_radius
        self.alpha = alpha
        self.fitness_func = fitness  
    def _preserve_niche_elites(self, combined_individuals, selected_individuals):

        population_values = np.array([ind.value for ind in combined_individuals])
        current_fitnesses = np.array([self.fitness_func(ind) for ind in combined_individuals])
        def find_local_minima_in_population(population_vals, fitnesses, niche_radius):
            local_minima_indices = []
            considered_indices = set()
            fitnesses_np = np.asarray(fitnesses)
            sorted_indices = np.argsort(fitnesses_np) 

            for i in sorted_indices:
                if i in considered_indices:
                    continue

                is_local_minimum = True
                neighbors = []

                for j in range(len(population_vals)):
                    if i != j:
                        if not isinstance(population_vals[i], np.ndarray) or not isinstance(population_vals[j], np.ndarray):
                            continue 
                        distance = np.linalg.norm(population_vals[i] - population_vals[j])
                        if distance < niche_radius:
                            neighbors.append(j)
                            if fitnesses_np[j] < fitnesses_np[i]:
                                is_local_minimum = False
                                break 

                if is_local_minimum:
                    local_minima_indices.append(i)
                    considered_indices.add(i)
                    considered_indices.update(neighbors) 
            
            return [population_vals[idx] for idx in local_minima_indices]


        current_niche_radius = self.niche_radius
        local_minima = find_local_minima_in_population(population_values, current_fitnesses, current_niche_radius)
        while len(local_minima) == 0 and current_niche_radius > 1e-6:
            current_niche_radius /= 2
            local_minima = find_local_minima_in_population(population_values, current_fitnesses, current_niche_radius)
        elites = []
        for minimum_point in local_minima: 
            niche_individuals = []
            for ind in combined_individuals:
                if not isinstance(ind.value, np.ndarray) or not isinstance(minimum_point, np.ndarray):
                    continue 
                distance = np.linalg.norm(ind.value - minimum_point)
                if distance < current_niche_radius: 
                    niche_individuals.append(ind)
            if niche_individuals:
                best_in_niche = min(niche_individuals, key=lambda ind: self.fitness_func(ind))
                elites.append(best_in_niche) 
        
        selected_set = set(selected_individuals) 
        final_selected = list(selected_individuals) 
        for elite in elites:
            if elite not in selected_set:
                final_selected.append(elite)
                selected_set.add(elite) 
        if len(final_selected) > len(self.pool.individuals):
             final_selected.sort(key=lambda ind: self.fitness_func(ind)) 
             final_selected = final_selected[:len(self.pool.individuals)] 

        return final_selected


    def step(self):
        mothers, fathers = self.pool.get_parents(self.n_offsprings)
        offsprings = []

        for mother, father in zip(mothers, fathers):
            offspring = mother.pair(father, self.pair_params)
            offspring.mutate(self.mutate_params)
            offsprings.append(offspring)

        combined_individuals = self.pool.individuals + offsprings
        if not combined_individuals: 
             self.pool.individuals = []
             return
        original_fitnesses = np.array([self.fitness_func(ind) for ind in combined_individuals])
        population_array = np.array([ind.value for ind in combined_individuals])
        shared_fitnesses = fitness_sharing_vectorized(population_array, original_fitnesses, self.alpha, self.niche_radius)
        
        indexed_shared_pop = list(enumerate(shared_fitnesses))
        indexed_shared_pop.sort(key=lambda x: x[1])
        selected_indices = [idx for idx, _ in indexed_shared_pop[:len(self.pool.individuals)]]
        selected_individuals_shared = [combined_individuals[i] for i in selected_indices] 
        
        final_selected_individuals = self._preserve_niche_elites(combined_individuals, selected_individuals_shared)

        self.pool.individuals = final_selected_individuals
def run_evolution(bc_function_name, population_size, n_offsprings, generations, bounds, initial_niche_radius, alpha):
    try:
        bc_func, normalize_func, scale_factor, original_bounds, n_dims = bc_functions.get_function(bc_function_name)
    except ImportError:
        print("Error: Could not import bc_functions. Ensure it's accessible.")
        return None, None, None, None, None 
    except AttributeError:
         print(f"Error: Function '{bc_function_name}' not found in bc_functions.")
         return None, None, None, None, None
    except TypeError: 
        print(f"Error: bc_functions.get_function for '{bc_function_name}' does not return expected 5 values (including n_dims).")
        try:
            n_dims = len(bounds)
            print(f"Falling back to n_dims={n_dims} based on provided bounds.")
            
        except TypeError:
             print("Cannot determine n_dims from bounds either. Aborting.")
             return None, None, None, None, None


    if bc_func is None: 
        print(f"Error: Function '{bc_function_name}' loading failed.")
        return None, None, None, None, None
    if isinstance(scale_factor, (int, float)):
        scale_factor = tuple([scale_factor] * n_dims)
    elif not hasattr(scale_factor, '__len__') or len(scale_factor) != n_dims:
        print(f"Warning: scale_factor length mismatch or invalid type ({scale_factor}) vs n_dims ({n_dims}). Using first element if possible, or 1.0.")
        fallback_sf = 1.0
        if hasattr(scale_factor, '__len__') and len(scale_factor) > 0:
            fallback_sf = scale_factor[0]
        elif isinstance(scale_factor, (int, float)):
            fallback_sf = scale_factor 
        scale_factor = tuple([fallback_sf] * n_dims)


    min_f, max_f = normalize_func()
    
    def adapted_fitness(individual):
        try:
            point = np.asarray(individual.value)
            if point.shape != (n_dims,):
                if point.size == n_dims:
                    point = point.reshape((n_dims,))
                else:
                    raise ValueError(f"Dimension mismatch: Expected ({n_dims},), got {point.shape}")
            return bc_func(point)
        except Exception as e:
            return np.inf 
    
    try:
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]
        current_n_dims_bounds = len(bounds) 
    except (TypeError, IndexError):
        print(f"Error: Invalid bounds format: {bounds}. Expected list of tuples, e.g., [(-1, 1), (-1, 1)]")
        return None, None, None, None, None
    if current_n_dims_bounds != n_dims:
        print(f"ERROR: Dimension mismatch between function ({n_dims}) and bounds ({current_n_dims_bounds}). Aborting.")
        return None, None, None, None, None
    init_params = {'lower_bound': lower_bounds[0], 'upper_bound': upper_bounds[0], 'dim': n_dims} 
    mutate_params = {'lower_bound': lower_bounds[0], 'upper_bound': upper_bounds[0], 'rate': 0.1, 'dim': n_dims} 
    pair_params = {'alpha': 0.7}


    evolution = EvolutionWithSharing(
        pool_size=population_size,
        fitness=adapted_fitness,  
        individual_class=BCOptimization,
        n_offsprings=n_offsprings,
        pair_params=pair_params,
        mutate_params=mutate_params,
        init_params=init_params,
        niche_radius=initial_niche_radius,
        alpha=alpha
    )
    
    def find_local_minima_in_population(population_inds, niche_radius):
        local_minima_found_indices = []
        considered_indices = set()

        if not population_inds: return [] 
        pop_values = np.array([ind.value for ind in population_inds])
        pop_fitnesses = np.array([adapted_fitness(ind) for ind in population_inds]) 
        pop_fitnesses_np = np.asarray(pop_fitnesses)
        sorted_indices = np.argsort(pop_fitnesses_np) 

        for i in sorted_indices:
            if i in considered_indices:
                continue

            is_local_minimum = True
            neighbors = []

            for j in range(len(pop_values)):
                if i != j:
                    if not isinstance(pop_values[i], np.ndarray) or not isinstance(pop_values[j], np.ndarray):
                        continue 

                    distance = np.linalg.norm(pop_values[i] - pop_values[j])
                    if distance < niche_radius:
                        neighbors.append(j)
                        if pop_fitnesses_np[j] < pop_fitnesses_np[i]:
                            is_local_minimum = False
                            break 

            if is_local_minimum:
                local_minima_found_indices.append(i)
                considered_indices.add(i)
                considered_indices.update(neighbors) 
        final_local_minima_individuals = [population_inds[idx] for idx in local_minima_found_indices]
        return final_local_minima_individuals
    for generation in range(generations):
        evolution.step()
        if (generation + 1) % 50 == 0:
             current_population = evolution.pool.individuals
             current_niche_radius_report = initial_niche_radius 
             minima_individuals = find_local_minima_in_population(current_population, current_niche_radius_report)

             if minima_individuals:
                 minima_values_norm = [ind.value for ind in minima_individuals]
                 if len(scale_factor) == n_dims and n_dims > 0 and len(minima_values_norm) > 0 and len(minima_values_norm[0]) == n_dims :
                     try:
                         scaled_minima_coords = [
                             [coord * sf for coord, sf in zip(norm_point, scale_factor)]
                             for norm_point in minima_values_norm
                         ]
                         original_minima_values = [bc_func(np.array(scaled_point)) for scaled_point in scaled_minima_coords]

                         print(f"Gen {generation + 1} ({bc_function_name}): Found {len(scaled_minima_coords)} potential minima.")
                         
                     except Exception as scale_print_err:
                          print(f"Gen {generation + 1} ({bc_function_name}): Found {len(minima_individuals)} potential minima (error during scaling/printing: {scale_print_err}).")

                 else:
                     print(f"Gen {generation + 1} ({bc_function_name}): Found {len(minima_individuals)} potential minima (scale factor mismatch or invalid data, showing normalized).")
                     

             else:
                 print(f"Gen {generation + 1} ({bc_function_name}): No Local Minima Found (radius={current_niche_radius_report:.4f})")
    final_population = evolution.pool.individuals
    final_minima_individuals = find_local_minima_in_population(final_population, initial_niche_radius)
    final_minima_points_norm = [ind.value for ind in final_minima_individuals]
    final_minima_list = [list(point) for point in final_minima_points_norm]

    print(f"EA:FS Final: Identified {len(final_minima_list)} potential local minima for {bc_function_name} using final population and initial niche radius.")
    
    return final_minima_list, bounds, bc_func, min_f, max_f
