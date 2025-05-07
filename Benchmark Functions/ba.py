import numpy as np
from bees_algorithm import BeesAlgorithm

def bees_algorithm_standard(score_function, bounds, max_iter, num_sites, num_elite,
                            selected_bees, elite_bees, stlim):
    
    # Negating the score function for maximisation within the BA library
    alg = BeesAlgorithm(lambda x: -score_function(x),
                        bounds[0], bounds[1], ns=0, nb=num_sites, ne=num_elite,
                        nrb=selected_bees, nre=elite_bees, stlim=stlim,
                        useSimplifiedParameters=True)

    alg.performFullOptimisation(max_iteration=max_iter)

    # This extracts and returns only the global best solution
    best_solution = alg.best_solution.values
    best_score = score_function(best_solution)

    return {"centre": best_solution, "score_best": best_score}