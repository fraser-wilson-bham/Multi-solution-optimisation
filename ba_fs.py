import numpy as np
from bees_algorithm import BeesAlgorithm
from ea_fs2 import fitness_sharing_vectorized

def bees_algorithm_fs(score_function, bounds, max_iter, niche_radius, alpha, num_sites, num_elite,
                      selected_bees, elite_bees, stlim):

    # Using the negated score function for initialisation, for maximisation within BA
    alg = BeesAlgorithm(lambda x: -score_function(x),
                        bounds[0], bounds[1], ns=num_sites, nb=num_sites, ne=num_elite,
                        nrb=selected_bees, nre=elite_bees, stlim=stlim,
                        useSimplifiedParameters=True)

    for _ in range(max_iter):
        alg.performSingleStep()

        population = np.array([site.values for site in alg.current_sites])
        original_fitnesses = np.array([-site.score for site in alg.current_sites])

        shared_fitnesses = fitness_sharing_vectorized(population, original_fitnesses, alpha, niche_radius)

        for i, site in enumerate(alg.current_sites):
            site.score = -shared_fitnesses[i]

    found_optima = []
    for site in alg.current_sites:
        is_duplicate = False
        for existing_optimum in found_optima:
            if np.linalg.norm(site.values - existing_optimum["centre"]) < niche_radius * 0.5:
                is_duplicate = True
                break
        if not is_duplicate:

            score_best = score_function(site.values)
            found_optima.append({"centre": site.values, "score_best": score_best, "score": site.score})

    return found_optima