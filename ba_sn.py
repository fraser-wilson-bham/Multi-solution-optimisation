import numpy as np
from bees_algorithm import BeesAlgorithm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import bc_functions2 as bc_functions
from matplotlib import cm


# --- Pruning Function Classes (from BA-LORRE) ---
class _PruningFunction:
    def apply_pruning(self, derating_functions):
        raise NotImplementedError("Subclasses must implement apply_pruning")

class PruningNone(_PruningFunction):
    def apply_pruning(self, derating_functions):
        return derating_functions

class PruningAbsScoreCutoff(_PruningFunction):
    def __init__(self, cutoff):
        self.cutoff = cutoff
        super().__init__()

    def apply_pruning(self, found_optima):
        return [opt for opt in found_optima if opt["score"] >= self.cutoff]

class PruningPercScoreCutoff(_PruningFunction):
    def __init__(self, cutoff):
        assert 0 <= cutoff <= 1, f"Cutoff value must be in [0,1], found {cutoff}"
        self.cutoff = cutoff
        super().__init__()

    def apply_pruning(self, found_optima):
        optima_scores = sorted([opt["score"] for opt in found_optima], reverse=True)
        score_cutoff = optima_scores[int((len(optima_scores) - 1) * self.cutoff)]
        return [opt for opt in found_optima if opt["score"] >= score_cutoff]

# --- Derating Function (Proportional Derating) ---
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

# --- Bees Algorithm with Sequential Niching ---
def bees_algorithm_sn(score_function, bounds, max_iter, niching_radius, num_sites, num_elite, selected_bees, elite_bees, stlim, pruning_functions=[]):
    found_optima = []
    derating_functions = []

    alg = BeesAlgorithm(lambda x: modified_fitness(x, derating_functions, score_function),
                       bounds[0], bounds[1], ns=0, nb=num_sites, ne=num_elite,
                       nrb=selected_bees, nre=elite_bees, stlim=stlim,
                       useSimplifiedParameters=True)

    for _ in range(max_iter):
        alg.performSingleStep()
        for site in alg.current_sites:
            if site.ttl == 0:
                is_duplicate = False
                for existing_optimum in found_optima:
                    if np.linalg.norm(site.values - existing_optimum["centre"]) < niching_radius * 0.5:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    score_best = -score_function(site.values)  # Use original, negated for maximisation
                    found_optima.append({"centre": site.values, "score_best": score_best, "score": site.score})
                    derating_functions.append({"centre": site.values, "radius": niching_radius})

    for pruning_func in pruning_functions:
        found_optima = pruning_func.apply_pruning(found_optima)

    return found_optima