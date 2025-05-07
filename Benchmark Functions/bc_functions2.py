import numpy as np
from scipy.stats import multivariate_normal
import logging
import math 
def _validate_point_standalone(point, n_dimensions):
    if not isinstance(point, (tuple, list, np.ndarray)):
        raise ValueError(f"Functions can be evaluated only on tuple, list, or np.ndarray, found {type(point)}")
    if isinstance(point, np.ndarray) and point.ndim > 1 and point.shape[0] == n_dimensions:
         pass 
    elif len(point) != n_dimensions:
        raise ValueError(f"Function requires {n_dimensions} dimensions, asked to be evaluated on a point of {len(point)} dimensions")
    if not (isinstance(point, np.ndarray) and point.ndim > 1):
        if not all(isinstance(v, (float, int, np.number)) for v in point): 
            idx = None
            for i, v in enumerate(point):
                if not isinstance(v, (float, int, np.number)):
                    idx = i
                    break
            vs = [str(x) + "(" + str(type(x)) + ")" if i == idx else x for i, x in enumerate(point)]
            raise ValueError(f"Functions can only be evaluated on numeric values (float, int, numpy numbers), passed {vs}")
def test_local_minimum(func_to_test, potential_minimum, n_dimensions, bounds,
                       radius_factor=1e-10, score_threshold=1e-6, n_tests=int(1e5), strict=True):

    if score_threshold < 0.0:
        raise ValueError(f"Score threshold must be a non-negative number, passed {score_threshold}")
    if radius_factor <= 0.0:
        raise ValueError(f"Radius factor must be a positive number, passed {radius_factor}")
    if not isinstance(n_tests, int) or n_tests <= 0:
        raise ValueError(f"The number of tests must be a positive integer, passed {n_tests}")
    if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2 and
            isinstance(bounds[0], (list, tuple, np.ndarray)) and
            isinstance(bounds[1], (list, tuple, np.ndarray)) and
            len(bounds[0]) == n_dimensions and len(bounds[1]) == n_dimensions):
         raise ValueError(f"Bounds must be tuple/list of ([lb_0,...], [ub_0,...]) with correct dimensions. Passed: {bounds}")

    bounds_lower, bounds_upper = bounds
    _validate_point_standalone(potential_minimum, n_dimensions) 
    potential_minimum_np = np.asarray(potential_minimum, dtype=float)
    try:
        r = np.linalg.norm(np.array(bounds_upper, dtype=np.float64) - np.array(bounds_lower, dtype=np.float64)) * radius_factor
        if r == 0 or not np.isfinite(r): 
             logging.warning(f"Calculated radius is zero or invalid ({r}). Using a small default radius (1e-12). Check bounds and radius_factor.")
             r = 1e-12
    except Exception as e:
        raise ValueError(f"Error calculating radius from bounds {bounds}. Check bounds format and values. Original error: {e}")


    pms = func_to_test(potential_minimum_np) 
    if not np.isfinite(pms):
        return (False, potential_minimum_np.tolist(), f"Validation Failed: Candidate score is not finite ({pms}).")


    for _ in range(n_tests):
        a = np.random.normal(0, 1, size=n_dimensions)
        an = np.linalg.norm(a)
        if an == 0: continue 
        
        sampled_radius = np.random.uniform(np.float64(r) * 1e-2, np.float64(r))
        p_offset = a / an * sampled_radius
        p = potential_minimum_np + p_offset
        

        if np.allclose(p, potential_minimum_np, atol=radius_factor*1e-6, rtol=1e-12): 
             continue 

        ps = func_to_test(p) 
        if not np.isfinite(ps):
             continue 

        dist = np.linalg.norm(p - potential_minimum_np) 
        if strict:
            
            if ps <= pms + score_threshold:
                message = (f"Validation Failed (strict): Point {p.tolist()} (dist={dist:.2e}) "
                           f"has score {ps:.8g} which is <= candidate score {pms:.8g} "
                           f"(threshold={score_threshold:.1e}).")
                return (False, p.tolist(), message)
        else:
            
            if ps < pms - score_threshold:
                 message = (f"Validation Failed (non-strict): Point {p.tolist()} (dist={dist:.2e}) "
                           f"has score {ps:.8g} which is significantly < candidate score {pms:.8g} "
                           f"(threshold={score_threshold:.1e}).")
                 return (False, p.tolist(), message)

    message = f'Validation Passed: Point {potential_minimum_np.tolist()} (score={pms:.8g}) is likely a local minimum '
    if strict:
        message += f'(strict, threshold={score_threshold:.1e}).'
    else:
        message += f'(non-strict, threshold={score_threshold:.1e}).'
    return (True, None, message)

def BC_1(x):
    x = np.asarray(x)
    f = (1 / (1 + np.exp(-10 * (x[0] - 0.25)))) * (1 / (1 + np.exp(-10 * (x[1] - 0.25))))
    sineX = np.sin(np.pi * (2 + 4 * f) * x[0])
    sineY = np.sin(np.pi * (2 + 4 * f) * x[1])
    z = sineX * sineY
    return z

def normalize_BC_1():
    return -1.0, 1.0

def BC_2(x):
    x = np.asarray(x)
    xc = np.arange(-0.75, 1, 0.5)
    yc = np.arange(-0.75, 1, 0.5)
    centres = len(xc)
    XC, YC = np.meshgrid(xc, yc, indexing='ij') 
    IC, JC = np.meshgrid(np.arange(centres), np.arange(centres), indexing='ij') 

    sigma = 0.005 + (IC + JC) * 0.012 
    if x.ndim == 1: 
        x0, x1 = x[0], x[1]
        squared_sum = (x0 - XC)**2 + (x1 - YC)**2 
    else: 
        x0, x1 = x[0][..., np.newaxis, np.newaxis], x[1][..., np.newaxis, np.newaxis] 
        squared_sum = (x0 - XC)**2 + (x1 - YC)**2 

    gauss = np.exp(-squared_sum / sigma) 
    if x.ndim == 1:
        z = -np.sum(gauss) 
    else:
        z = -np.sum(gauss, axis=(-2, -1)) 

    return z

def normalize_BC_2():
    x_range = np.arange(-1, 1.001, 0.005) 
    y_range = np.arange(-1, 1.001, 0.005)
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    Z_grid = BC_2([X, Y]) 
    return np.min(Z_grid), np.max(Z_grid)

def BC_3(x):
    x = np.asarray(x)
    c1 = np.array([0, 0])
    c2 = np.array([0.2, 0])
    sigma1 = np.array([0.4, 0.4])
    sigma2 = np.array([0.03, 0.03])
    argument1 = ((x[0] - c1[0])**2 / sigma1[0]**2) + ((x[1] - c1[1])**2 / sigma1[1]**2)
    gauss1 = np.exp(-argument1)
    argument2 = ((x[0] - c2[0])**2 / sigma2[0]**2) + ((x[1] - c2[1])**2 / sigma2[1]**2)
    gauss2 = np.exp(-argument2)
    z = -gauss1 - 0.5 * gauss2
    return z

def normalize_BC_3():
    return -1.5, 0.0
    
def BC_4(x):
    x = np.asarray(x)
    c1 = np.array([0, 0])
    c2 = np.array([0.2, 0])
    sigma1 = np.array([0.06, 0.6]) 
    sigma2 = np.array([0.03, 0.03]) 
    argument1 = ((x[0] - c1[0])**2 / sigma1[0]**2) + ((x[1] - c1[1])**2 / sigma1[1]**2)
    gauss1 = np.exp(-argument1)
    argument2 = ((x[0] - c2[0])**2 / sigma2[0]**2) + ((x[1] - c2[1])**2 / sigma2[1]**2)
    gauss2 = np.exp(-argument2)
    z = -gauss1 - 0.5 * gauss2
    return z

def normalize_BC_4():
    return -1.5, 0.0

def BC_5(x):
    x = np.asarray(x)
    c1 = np.array([0, 0])
    c2 = np.array([0.4, 0]) 
    sigma1 = np.array([0.06, 0.6])
    sigma2 = np.array([0.03, 0.03])
    argument1 = ((x[0] - c1[0])**2 / sigma1[0]**2) + ((x[1] - c1[1])**2 / sigma1[1]**2)
    gauss1 = np.exp(-argument1)
    argument2 = ((x[0] - c2[0])**2 / sigma2[0]**2) + ((x[1] - c2[1])**2 / sigma2[1]**2)
    gauss2 = np.exp(-argument2)
    z = -gauss1 - 0.5 * gauss2
    return z

def normalize_BC_5():
    return -1.5, 0.0

def Rastrigin(x):
    x = np.asarray(x)
    x_scaled = 5.12 * x
    A = 10.0
    n = x.shape[-1] if x.ndim > 0 else 0
    if x.ndim > 1 and x.shape[0]==2: 
        n = 2
        z = A * n + np.sum(x_scaled**2 - A * np.cos(2 * np.pi * x_scaled), axis=0)
    elif x.ndim == 1: 
        n = len(x)
        z = A * n + np.sum(x_scaled**2 - A * np.cos(2 * np.pi * x_scaled))
    else: 
        return np.nan 
    return z

def normalize_Rastrigin():
    x_range = np.arange(-1, 1.001, 0.02)
    y_range = np.arange(-1, 1.001, 0.02)
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    Z = Rastrigin([X, Y])
    return 0.0, np.max(Z)

def Schwefel(x):
    x = np.asarray(x)
    x_scaled = 500.0 * x
    n = x.shape[-1] if x.ndim > 0 else 0
    if x.ndim > 1 and x.shape[0]==2: 
        n = 2
        z = 418.9829 * n - np.sum(x_scaled * np.sin(np.sqrt(np.abs(x_scaled))), axis=0)
    elif x.ndim == 1: 
        n = len(x)
        z = 418.9829 * n - np.sum(x_scaled * np.sin(np.sqrt(np.abs(x_scaled))))
    else:
        return np.nan
    return z

def normalize_Schwefel():
    x_range = np.arange(-1, 1.001, 0.02)
    y_range = np.arange(-1, 1.001, 0.02)
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    Z = Schwefel([X, Y])
    return np.min(Z), np.max(Z) 

def Griewank(x, zoom=0):
    x = np.asarray(x)
    x_scaled = 600.0 * x
    n = x.shape[-1] if x.ndim > 0 else 0
    if x.ndim > 1 and x.shape[0]==2: 
        n=2
        part1 = np.sum(x_scaled**2, axis=0) / 4000.0
        ii = np.arange(1, n + 1).reshape(-1, 1, 1) 
        part2 = np.prod(np.cos(x_scaled / np.sqrt(ii)), axis=0)
    elif x.ndim == 1: 
        n = len(x)
        part1 = np.sum(x_scaled**2) / 4000.0
        ii = np.arange(1, n + 1)
        part2 = np.prod(np.cos(x_scaled / np.sqrt(ii)))
    else:
        return np.nan
    return 1.0 + part1 - part2

def PitsAndHoles(x):
    x_np = np.asarray(x)
    if not ((x_np.ndim == 1 and len(x_np) == 2) or (x_np.ndim > 1 and x_np.shape[0] == 2)):
         raise ValueError("PitsAndHoles function as defined here requires 2 dimensions")

    mu = np.array([ [0,0], [0.4,0], [0,0.4], [-0.4,0], [0,-0.4], [0.2,0.2], [-0.2,-0.2], [-0.2,0.2], [0.2,-0.2] ])
    c =  np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.02, 0.02, 0.02, 0.02])
    v = np.array([2.0, 2.5, 2.7, 2.5, 2.3, 1.0, 1.5, 1.2, 1.3]) 

    result = 0.0 
    for i in range(len(mu)):
        if x_np.ndim > 1 and x_np.shape[0]==2: 
            point_for_pdf = np.stack([x_np[0], x_np[1]], axis=-1) 
        else: 
             point_for_pdf = x_np 

        cov_matrix = np.array([[c[i], 0], [0, c[i]]])
        try:
             pdf_val = multivariate_normal.pdf(point_for_pdf, mean=mu[i], cov=cov_matrix)
             result -= pdf_val * v[i] 
        except np.linalg.LinAlgError:
             continue 
        except ValueError as ve:
             raise ve

    return result

def normalize_PitsAndHoles():
    x_range = np.arange(-1, 1.001, 0.01) 
    y_range = np.arange(-1, 1.001, 0.01)
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    Z = PitsAndHoles([X, Y])
    return np.min(Z), np.max(Z)

functions = {
    "BC_1": {"func": BC_1, "normalize": normalize_BC_1, "scale_factor": (1, 1), "original_bounds" : ((-1,1), (-1,1)), "n_dimensions": 2},
    "BC_2": {"func": BC_2, "normalize": normalize_BC_2, "scale_factor": (1, 1), "original_bounds" : ((-1,1), (-1,1)), "n_dimensions": 2},
    "BC_3": {"func": BC_3, "normalize": normalize_BC_3, "scale_factor": (1, 1), "original_bounds" : ((-1,1), (-1,1)), "n_dimensions": 2},
    "BC_4": {"func": BC_4, "normalize": normalize_BC_4, "scale_factor": (1, 1), "original_bounds" : ((-1,1), (-1,1)), "n_dimensions": 2},
    "BC_5": {"func": BC_5, "normalize": normalize_BC_5, "scale_factor": (1, 1), "original_bounds" : ((-1,1), (-1,1)), "n_dimensions": 2},
    "Rastrigin": {"func": Rastrigin, "normalize": normalize_Rastrigin, "scale_factor": (5.12, 5.12), "original_bounds" : ((-5.12,5.12), (-5.12, 5.12)), "n_dimensions": 2},
    "Schwefel": {"func": Schwefel, "normalize": normalize_Schwefel, "scale_factor": (500, 500), "original_bounds" : ((-500,500), (-500,500)), "n_dimensions": 2}, 
    "P&H": {"func": PitsAndHoles, "normalize": normalize_PitsAndHoles, "scale_factor": (1, 1), "original_bounds" : ((-1,1), (-1,1)), "n_dimensions": 2},
}


def get_function(func_name):

    if func_name in functions:
        func_data = functions[func_name]
        n_dim = func_data.get("n_dimensions", 2) 
        normalize_func = func_data.get("normalize", lambda: (0,1)) 
        return func_data["func"], normalize_func, func_data["scale_factor"], func_data["original_bounds"], n_dim
    else:
        return None, None, None, None, None
