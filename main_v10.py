
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
try:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    
    matplotlib.rcParams['font.sans-serif'] = ['Arial'] + matplotlib.rcParams['font.sans-serif']
except Exception as e:
    print(f"WARNING: Could not set default font to Arial. Using Matplotlib defaults. Error: {e}")
import bc_functions2 as bc_functions 
from bc_functions2 import test_local_minimum 
from ea_fs2 import run_evolution
from ba_sn import bees_algorithm_sn, PruningNone, PruningAbsScoreCutoff, PruningPercScoreCutoff
from ba_lorre import LORRE, PruningProximity, PruningPercScoreCutoff as LORREPruningPercScoreCutoff
from ba_fs import bees_algorithm_fs
from ba import bees_algorithm_standard
from ea_sn import run_evolution_sn, plot_results_ea_sn
from matplotlib.ticker import MaxNLocator, LinearLocator
from scipy.spatial.distance import cdist 
import logging 
import traceback 
import csv  
import time 
import math 
import multiprocessing 
import os 
TRUE_MINIMA_DB = {
    "BC_1": [
        [-0.75000000, -0.25000000], [-0.75000000, 0.75000000], [-0.25000000, -0.75000000],
        [-0.24800000, 0.24800000], [0.16300000, 0.48700000], [0.24800000, -0.24800000],
        [0.32000000, 0.96400000], [0.32700000, 0.54300000], [0.45200000, 0.99400000],
        [0.45600000, 0.64000000], [0.48700000, 0.16300000], [0.54300000, 0.32700000],
        [0.59800000, 0.76800000], [0.64000000, 0.45600000], [0.75000000, -0.75000000],
        [0.75400000, 0.92200000], [0.76800000, 0.59800000], [0.92200000, 0.75400000],
        [0.96400000, 0.32000000], [0.99400000, 0.45200000]
    ],
    "BC_2": [
        [-0.75000000, -0.75000000], [-0.75000000, -0.25000000], [-0.75000000, 0.25000000],
        [-0.74600000, 0.75000000], [-0.25000000, -0.75000000], [-0.25000000, -0.25000000],
        [-0.24600000, 0.25400000], [-0.24200000, 0.74800000], [0.25000000, -0.75000000],
        [0.25400000, -0.24600000], [0.26000000, 0.26000000], [0.26600000, 0.74400000],
        [0.73400000, 0.73400000], [0.74400000, 0.26600000], [0.74800000, -0.24200000],
        [0.75000000, -0.74600000]
    ],
    "BC_3": [[0.00000000, 0.00000000], [0.19800000, 0.00000000]],
    "BC_4": [[0.00000000, 0.00000000], [0.20000000, 0.00000000]],
    "BC_5": [[0.00000000, 0.00000000], [0.40000000, 0.00000000]],
    "Rastrigin": [
        [-0.39000000, -0.39000000], [-0.39000000, -0.19000000], [-0.39000000, 0.00000000],
        [-0.39000000, 0.19000000], [-0.39000000, 0.39000000], [-0.19000000, -0.39000000],
        [-0.19000000, -0.19000000], [-0.19000000, 0.00000000], [-0.19000000, 0.19000000],
        [-0.19000000, 0.39000000], [0.00000000, -0.39000000], [0.00000000, -0.19000000],
        [0.00000000, 0.00000000], [0.00000000, 0.19000000], [0.00000000, 0.39000000],
        [0.19000000, -0.39000000], [0.19000000, -0.19000000], [0.19000000, 0.00000000],
        [0.19000000, 0.19000000], [0.19000000, 0.39000000], [0.39000000, -0.39000000],
        [0.39000000, -0.19000000], [0.39000000, 0.00000000], [0.39000000, 0.19000000],
        [0.39000000, 0.39000000]
    ],
    "Schwefel": [
        [-0.61000000, -0.61000000], [-0.61000000, 0.41000000], [-0.61000000, 0.84000000],
        [-0.25000000, 0.84000000], [-0.05000000, 0.84000000], [0.13000000, 0.84000000],
        [0.41000000, -0.61000000], [0.41000000, 0.84000000], [0.84000000, -0.61000000],
        [0.84000000, -0.25000000], [0.84000000, -0.05000000], [0.84000000, 0.13000000],
        [0.84000000, 0.41000000], [0.84000000, 0.84000000]
    ],
    "P&H": [
        [-0.50000000, -0.50000000], [-0.50000000, 0.50000000], [0.50000000, -0.50000000]
    ],
}
DEFAULT_PRUNING_PERCENTAGE = 50
FUNCTION_PRUNING_PERCENTAGES = { 
        "BC_1": 10, "BC_2": 10, "BC_3": 50, "BC_4": 90, "BC_5": 90,
        "Rastrigin": 10, "Schwefel": 24, "P&H": 10,
}

def plot_results(final_optima, bounds, function_name, get_score, title_algorithm,
                 save_svg=False, algo_name_clean=None, run_number=None, show_plots=True): 
    run_suffix = f" (Run {run_number})" if run_number is not None else ""
    full_title = f"{title_algorithm}{run_suffix}"

    title_fontsize = 24
    label_fontsize = 18
    tick_fontsize = 16

    x_range = np.linspace(bounds[0][0], bounds[1][0], 100)
    y_range = np.linspace(bounds[0][1], bounds[1][1], 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
             val = get_score(np.array([X[i, j], Y[i, j]]))
             Z[i, j] = val if np.isfinite(val) else 0
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    function_surface = ax_3d.plot_surface(X, Y, 1 - Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none',
                                          alpha=0.3)
    ax_3d.set_xlabel('x', fontsize=label_fontsize)
    ax_3d.set_ylabel('y', fontsize=label_fontsize)
    ax_3d.set_zlabel('z', fontsize=label_fontsize)
    ax_3d.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax_3d.view_init(60, 35)
    ax_3d.set_zlim(0, 1)
    z_tick_positions = np.linspace(0, 1, 3)
    z_tick_labels = [f"{tick:.2f}" for tick in z_tick_positions]
    ax_3d.set_zticks(z_tick_positions)
    ax_3d.set_zticklabels(z_tick_labels)


    try:
        if isinstance(final_optima, dict):
            if "centre" in final_optima and final_optima["centre"] is not None and len(final_optima["centre"]) == 2:
                optima_x = final_optima["centre"][0]; optima_y = final_optima["centre"][1]
                optima_z = get_score(np.array([optima_x, optima_y]))
                if np.isfinite(optima_z): ax_3d.scatter(optima_x, optima_y, 1 - optima_z, c='red', s=20, depthshade=False)
        elif final_optima is not None and len(final_optima) > 0:
            optima_x_list, optima_y_list = [], []
            n_dims_plot = None
            for item in final_optima:
                 pt_plot = item.get("centre") if isinstance(item, dict) else item
                 if pt_plot is not None and hasattr(pt_plot, '__len__'): n_dims_plot = len(pt_plot); break

            if n_dims_plot is not None:
                 if isinstance(final_optima[0], dict):
                     for opt in final_optima:
                         if opt and "centre" in opt and opt["centre"] is not None and len(opt["centre"]) == n_dims_plot:
                              optima_x_list.append(opt["centre"][0])
                              if n_dims_plot > 1: optima_y_list.append(opt["centre"][1])
                 elif isinstance(final_optima[0], (list, tuple, np.ndarray)):
                      for opt in final_optima:
                           if opt is not None and len(opt) == n_dims_plot:
                               optima_x_list.append(opt[0])
                               if n_dims_plot > 1: optima_y_list.append(opt[1])

                 if optima_x_list:
                     optima_x = np.array(optima_x_list); optima_y = np.array(optima_y_list) if n_dims_plot > 1 else np.zeros_like(optima_x)
                     optima_points = np.vstack((optima_x, optima_y)).T if n_dims_plot > 1 else optima_x.reshape(-1, 1)
                     optima_z = np.array([get_score(p) for p in optima_points])
                     valid_z_idx = np.isfinite(optima_z)
                     if np.any(valid_z_idx):
                         ax_3d.scatter(optima_x[valid_z_idx], optima_y[valid_z_idx], 1 - optima_z[valid_z_idx], c='red', s=20, depthshade=False)
    except Exception as e: print(f"Plotting Warning (3D Scatter): {e}")
    ax_3d.set_title(f'3D Plot for {function_name} using {title_algorithm}', fontsize=title_fontsize)
    fig_3d.tight_layout()
    fig_contour, ax_contour = plt.subplots(figsize=(7, 6))
    contour = ax_contour.contourf(X, Y, 1 - Z, 20, cmap=cm.viridis)

    try:
        if isinstance(final_optima, dict):
            if "centre" in final_optima and final_optima["centre"] is not None and len(final_optima["centre"]) == 2:
                 ax_contour.scatter(final_optima["centre"][0], final_optima["centre"][1], c='red', marker='o', s=50)
        elif final_optima is not None and len(final_optima) > 0:
             plot_x, plot_y = [], []
             n_dims_plot = None
             for item in final_optima:
                  pt_plot = item.get("centre") if isinstance(item, dict) else item
                  if pt_plot is not None and hasattr(pt_plot, '__len__'): n_dims_plot = len(pt_plot); break

             if n_dims_plot == 2:
                 if isinstance(final_optima[0], dict):
                      for opt in final_optima:
                           if opt and "centre" in opt and opt["centre"] is not None and len(opt["centre"]) == 2: plot_x.append(opt["centre"][0]); plot_y.append(opt["centre"][1])
                 elif isinstance(final_optima[0], (list, tuple, np.ndarray)):
                      for opt in final_optima:
                           if opt is not None and len(opt) == 2: plot_x.append(opt[0]); plot_y.append(opt[1])
                 if plot_x: ax_contour.scatter(plot_x, plot_y, c='red', marker='o', s=50)
    except Exception as e: print(f"Plotting Warning (Contour Scatter): {e}")
    ax_contour.set_xlabel('x', fontsize=label_fontsize)
    ax_contour.set_ylabel('y', fontsize=label_fontsize)
    ax_contour.set_title(f'Contour Plot for {function_name}\nusing {title_algorithm}', fontsize=title_fontsize)
    ax_contour.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax_contour.yaxis.set_major_locator(LinearLocator(numticks=5))
    ax_contour.set_aspect('equal')
    cb = fig_contour.colorbar(contour, ax=ax_contour)
    cb.set_label('z', fontsize=label_fontsize)
    cb.ax.tick_params(labelsize=tick_fontsize)

    if save_svg and algo_name_clean and function_name and run_number is not None:
        safe_func_name = function_name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
        safe_algo_name = algo_name_clean.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
        svg_filename_3d = f"{safe_func_name}_{safe_algo_name}_run{run_number}_final_3d.svg"
        svg_filename_contour = f"{safe_func_name}_{safe_algo_name}_run{run_number}_final_contour.svg"
        try: fig_3d.savefig(svg_filename_3d, format='svg')#, bbox_inches='tight')
        except Exception as e: print(f"Plotting Warning: Failed to save 3D SVG '{svg_filename_3d}': {e}")
        try: fig_contour.savefig(svg_filename_contour, format='svg', bbox_inches='tight')
        except Exception as e: print(f"Plotting Warning: Failed to save Contour SVG '{svg_filename_contour}': {e}")

    if show_plots:
        plt.show()

    plt.close(fig_3d)
    plt.close(fig_contour)

def validate_optima(optima_list, func_eval, n_dims, bounds_norm, func_name, algo_name,
                    distinctness_threshold=0.1, verbose=False):

    if verbose: print(f"\n--- Validating Optima for {func_name} using {algo_name} ---")
    if verbose: print(f"--- Stage 1: Individual Point Validation (on pre-filtered points) ---")

    if optima_list is None or not optima_list:
        return []

    use_strict_validation = False; validation_threshold = 1e-8
    non_strict_functions = []
    if func_name in non_strict_functions:
         if verbose: print(f"Using non-strict validation (strict=False) for function {func_name}.")
         use_strict_validation = False; validation_threshold = 1e-14

    test_params = {"radius_factor": 1e-9, "score_threshold": validation_threshold,
                   "n_tests": 100000, "strict": use_strict_validation}

    initial_points, initial_scores = [], []
    for item in optima_list:
        pt = item.get("centre") if isinstance(item, dict) else item
        if pt is not None and hasattr(pt, '__len__') and len(pt) == n_dims:
            try:
                pt_array = np.asarray(pt, dtype=float); score = func_eval(pt_array)
                if np.isfinite(score): initial_points.append(list(pt)); initial_scores.append(score)
            except Exception as score_err: pass

    if not initial_points:
         return []

    validated_points_stage1, validated_scores_stage1 = [], []
    for i, opt_point in enumerate(initial_points):
        try:
            potential_min_val = [float(coord) for coord in opt_point]
            is_valid, failing_point, message = test_local_minimum(func_eval, potential_min_val, n_dims, bounds_norm, **test_params)
            if is_valid: validated_points_stage1.append(opt_point); validated_scores_stage1.append(initial_scores[i])
        except Exception as e: pass

    if not validated_points_stage1: return []

    sorted_indices = np.argsort(validated_scores_stage1)
    sorted_points = [validated_points_stage1[i] for i in sorted_indices]
    sorted_scores = [validated_scores_stage1[i] for i in sorted_indices]
    final_distinct_optima = []

    if sorted_points:
        final_distinct_optima.append(sorted_points[0])
        for i in range(1, len(sorted_points)):
            current_point = np.asarray(sorted_points[i]); is_distinct = True
            for accepted_index, accepted_point in enumerate(final_distinct_optima):
                dist = np.linalg.norm(current_point - np.asarray(accepted_point))
                if dist < distinctness_threshold:
                    is_distinct = False
                    break
            if is_distinct:
                final_distinct_optima.append(sorted_points[i])



    return final_distinct_optima
def calculate_error_metric(found_optima_points, true_optima_points):
    """ Calculates the error metric based on Equation (17). """
    if not true_optima_points: return np.inf
    if not found_optima_points: return np.inf

    no = len(true_optima_points)
    try:
        valid_found_points = []
        expected_dim = len(true_optima_points[0])
        for p in found_optima_points:
             if p is not None and hasattr(p, '__len__') and len(p) == expected_dim: valid_found_points.append(p)
        if not valid_found_points: return np.inf

        found_pts_arr = np.array(valid_found_points, dtype=float)
        true_pts_arr = np.array(true_optima_points, dtype=float)
        if found_pts_arr.ndim != 2 or true_pts_arr.ndim != 2 or found_pts_arr.shape[1] != true_pts_arr.shape[1]: return np.inf
    except (ValueError, IndexError) as e: return np.inf

    distances = cdist(true_pts_arr, found_pts_arr, metric='euclidean')
    min_distances_to_found = np.min(distances, axis=1)
    total_error = np.sum(min_distances_to_found)
    average_error = total_error / no
    return average_error
def run_single_test(function_name, algo_num, run_i, total_runs, bounds_normalized, bounds_normalized_list, true_minima_norm, algo_params):

    print(f"Starting: Func={function_name}, Algo={algo_num}, Run={run_i+1}/{total_runs}")
    n_dims = None 
    bc_func = None 
    try:
        bc_func, normalize_func, scale_factor, original_bounds, n_dims = bc_functions.get_function(function_name)
        if bc_func is None: raise ValueError("Function loading returned None.")
    except Exception as func_load_err:
        print(f"ERROR (Worker): Failed to load function '{function_name}': {func_load_err}. Skipping run.")
        algo_name = algo_params.get(algo_num, {}).get('name', f"UnknownAlgoNum{algo_num}")
        return {"func": function_name, "algo_num": algo_num, "run": run_i, "error": np.inf, "count": 0, "time": np.nan, "algo_name": algo_name}
    get_score = None 
    try:
        minF, maxF = normalize_func()
        global_minF, global_maxF = minF, maxF
        global_rangeF = global_maxF - global_minF if abs(global_maxF - global_minF) >= 1e-14 else 1e-14
        global_minF_stable = global_minF
        def get_score_local(x):
            x_np = np.asarray(x)
            try:
                if x_np.shape[-1] != n_dims: raise ValueError(f"Dim mismatch {x_np.shape[-1]} vs {n_dims}")
                if x_np.ndim > 1: raise ValueError(f"get_score expects 1D array")
                raw_val = bc_func(x_np) 
                if np.iscomplexobj(raw_val): raw_val = np.real(raw_val)
                if abs(global_rangeF) < 1e-14: raw_score = 0.5 if abs(raw_val - global_minF_stable) < 1e-9 else (1.0 if raw_val > global_minF_stable else 0.0)
                else: raw_score = (raw_val - global_minF_stable) / global_rangeF
                return np.clip(raw_score, 0.0, 1.0)
            except Exception as e:
                return np.nan
        get_score = get_score_local 

    except Exception as e:
        print(f"ERROR (Worker): Normalization setup failed for {function_name}: {e}. Skipping run.")
        algo_name = algo_params.get(algo_num, {}).get('name', f"UnknownAlgoNum{algo_num}")
        return {"func": function_name, "algo_num": algo_num, "run": run_i, "error": np.inf, "count": 0, "time": np.nan, "algo_name": algo_name}
    final_optima_run = None
    get_score_for_run = get_score 
    run_successful = False
    final_validated_optima = []
    num_distinct_minima_this_run = 0
    duration_this_run = np.nan
    algo_name = "Unknown"

    try:
        start_time = time.time()

        if algo_num not in algo_params:
            raise ValueError(f"Algorithm number '{algo_num}' not found in algo_params.")
        params = algo_params[algo_num]

        if algo_num == "1": algo_name = "BA (Standard)"
        elif algo_num == "2": algo_name = "LORRE-BA"
        elif algo_num == "3": algo_name = "BA-SN"
        elif algo_num == "4": algo_name = "BA-FS"
        elif algo_num == "5": algo_name = "EA-FS"
        elif algo_num == "6": algo_name = "EA-SN"
        else: algo_name = f"InvalidAlgoNum{algo_num}"

        if algo_num == "3": 
            final_optima_run = bees_algorithm_sn(bc_func, params['bounds'], params['max_iter'], params['radius'], params['n_sites'], params['n_elite'], params['sel_bees'], params['elite_bees'], params['stlim'], params['pruning'])
        elif algo_num == "2": 
            if get_score is None: raise RuntimeError("get_score function was not defined due to earlier error.")
            def score_for_lorre(x): val = get_score(x); return -val if np.isfinite(val) else -np.inf
            alg = LORRE(score_function=score_for_lorre, range_min=params['bounds'][0], range_max=params['bounds'][1])
            alg.performFullOptimisation(max_iteration=params['max_iter'])
            raw_optima_lorre = alg.getFoundOptima(pruning_functions=params['pruning'])
            if raw_optima_lorre:
                temp_list = []
                for opt in raw_optima_lorre:
                    point = None
                    if isinstance(opt, np.ndarray): point = opt.tolist()
                    elif hasattr(opt, 'values') and isinstance(getattr(opt, 'values', None), (np.ndarray, list, tuple)): point = list(opt.values)
                    elif isinstance(opt, (list, tuple)): point = list(opt)
                    if point and n_dims is not None and len(point) == n_dims: temp_list.append(point)
                final_optima_run = temp_list
            else: final_optima_run = []
        elif algo_num == "4": 
            if get_score is None: raise RuntimeError("get_score function was not defined due to earlier error.")
            final_optima_run = bees_algorithm_fs(get_score, params['bounds'], params['max_iter'], params['radius'], params['alpha'], params['n_sites'], params['n_elite'], params['sel_bees'], params['elite_bees'], params['stlim'])
        elif algo_num == "1": 
            opt_dict = bees_algorithm_standard(bc_func, params['bounds'], params['max_iter'], params['n_sites'], params['n_elite'], params['sel_bees'], params['elite_bees'], params['stlim'])
            final_optima_run = [opt_dict] if opt_dict else []
        elif algo_num == "5": 
            optima_list_ea_fs, _, _, _, _ = run_evolution(function_name, params['pop_size'], params['n_offspr'], params['gens'], params['bounds'], params['radius'], params['alpha'])
            if optima_list_ea_fs is None: raise RuntimeError(f"EA:FS run_evolution failed for {function_name}")
            final_optima_run = optima_list_ea_fs
        elif algo_num == "6": 
            raw_optima_ea_sn, _, bc_func_ea_sn, min_f_ea_sn, max_f_ea_sn = run_evolution_sn(function_name, params['pop_size'], params['n_offspr'], params['gens'], params['bounds'], params['radius'])
            if raw_optima_ea_sn is None: raise RuntimeError(f"EA:SN run_evolution_sn failed for {function_name}")
            final_optima_run = [opt['centre'] for opt in raw_optima_ea_sn if opt and 'centre' in opt]

            range_ea_sn = max_f_ea_sn - min_f_ea_sn if abs(max_f_ea_sn - min_f_ea_sn) >= 1e-14 else 1e-14
            min_f_ea_sn_stable = min_f_ea_sn
            def get_score_ea_sn_local(x):
                 try:
                     x_arr = np.asarray(x);
                     if n_dims is None: raise RuntimeError("n_dims not available for EA:SN score calc")
                     if x_arr.shape[-1] != n_dims: raise ValueError("Dim mismatch")
                     raw = bc_func_ea_sn(x_arr);
                     if np.iscomplexobj(raw): raw = np.real(raw)
                     if abs(range_ea_sn) < 1e-14: score = 0.5 if abs(raw - min_f_ea_sn_stable) < 1e-9 else (1.0 if raw > min_f_ea_sn_stable else 0.0)
                     else: score = (raw - min_f_ea_sn_stable) / range_ea_sn
                     return np.clip(score, 0.0, 1.0)
                 except Exception: return np.nan
            get_score_for_run = get_score_ea_sn_local 
        else:
             raise ValueError(f"Unhandled algorithm number '{algo_num}' during execution.")

        end_time = time.time()
        duration_this_run = end_time - start_time
        run_successful = True

    except Exception as e:
        print(f"!!!!!!!! ERROR during run {run_i+1} for Algo={algo_name} on {function_name} !!!!!!!!!!")
        print(f"Error details: {e}")
        run_successful = False
    run_error = np.inf
    if run_successful:
        if n_dims is None or get_score is None:
             print(f"Skipping post-processing for {function_name}/{algo_name}/Run{run_i+1} due to earlier setup error.")
             run_successful = False 
             num_distinct_minima_this_run = 0 
        else:
            current_pruning_percentage = FUNCTION_PRUNING_PERCENTAGES.get(function_name, DEFAULT_PRUNING_PERCENTAGE)
            cutoff_score = current_pruning_percentage / 100.0
            points_to_validate = []
            raw_points_extracted = 0
            score_pruning_applied = False
            kept_score_pruning = 0
            rejected_score_pruning = 0
            score_func_for_pruning = get_score_for_run

            extracted_points_for_processing = []
            temp_optima_list_proc = final_optima_run if isinstance(final_optima_run, (list, np.ndarray)) else []
            if not temp_optima_list_proc and isinstance(final_optima_run, dict):
                 temp_optima_list_proc = [final_optima_run]

            for item_proc in temp_optima_list_proc:
                 pt_proc = None
                 pt_proc = item_proc.get("centre") if isinstance(item_proc, dict) else item_proc
                 if pt_proc is not None and hasattr(pt_proc, '__len__') and len(pt_proc) == n_dims:
                     extracted_points_for_processing.append(list(pt_proc))
            raw_points_extracted = len(extracted_points_for_processing)

            if algo_num in ["2", "3", "4", "5", "6"]:
                score_pruning_applied = True
                if raw_points_extracted > 0:
                    for point in extracted_points_for_processing:
                        try:
                            point_score = score_func_for_pruning(np.asarray(point))
                            if np.isfinite(point_score) and point_score <= cutoff_score:
                                points_to_validate.append(point); kept_score_pruning += 1
                            else: rejected_score_pruning += 1
                        except Exception as score_err_initial_prune: rejected_score_pruning += 1
                points_to_validate = points_to_validate
            else:
                points_to_validate = extracted_points_for_processing
            final_validated_optima = []
            if points_to_validate:
                dist_thresh = 0.1
                score_func_for_validation = score_func_for_pruning if score_pruning_applied else get_score
                final_validated_optima = validate_optima(points_to_validate, score_func_for_validation,
                                                         n_dims, bounds_normalized, function_name, algo_name,
                                                         distinctness_threshold=dist_thresh, verbose=False)

            num_distinct_minima_this_run = len(final_validated_optima)
            run_error = calculate_error_metric(final_validated_optima, true_minima_norm)
            if final_validated_optima:
                plot_results(final_validated_optima, bounds_normalized, function_name, get_score,
                             f"{algo_name}", save_svg=True, algo_name_clean=algo_name,
                             run_number=run_i + 1, show_plots=False)

            print(f"Finished: Func={function_name}, Algo={algo_name}, Run={run_i+1}/{total_runs} -> Err={run_error:.4f}, Count={num_distinct_minima_this_run}, Time={duration_this_run:.2f}s")
    if not run_successful:
        num_distinct_minima_this_run = 0
        run_error = np.inf 
    return {
        "func": function_name,
        "algo_num": algo_num,
        "run": run_i,
        "error": run_error,
        "count": num_distinct_minima_this_run,
        "time": duration_this_run,
        "algo_name": algo_name
    }
    try:
        minF, maxF = normalize_func()
        global_minF, global_maxF = minF, maxF
        global_rangeF = global_maxF - global_minF if abs(global_maxF - global_minF) >= 1e-14 else 1e-14
        global_minF_stable = global_minF

        def get_score(x):
            x_np = np.asarray(x)
            try:
                if x_np.shape[-1] != n_dims: raise ValueError(f"Dim mismatch {x_np.shape[-1]} vs {n_dims}")
                if x_np.ndim > 1: raise ValueError(f"get_score expects 1D array")
                raw_val = bc_func(x_np)
                if np.iscomplexobj(raw_val): raw_val = np.real(raw_val)
                if abs(global_rangeF) < 1e-14: raw_score = 0.5 if abs(raw_val - global_minF_stable) < 1e-9 else (1.0 if raw_val > global_minF_stable else 0.0)
                else: raw_score = (raw_val - global_minF_stable) / global_rangeF
                return np.clip(raw_score, 0.0, 1.0)
            except Exception as e:
                return np.nan
    except Exception as e:
        print(f"ERROR (Worker): Normalization setup failed for {function_name}: {e}. Skipping run.")
        algo_name = algo_params.get(algo_num, {}).get('name', f"UnknownAlgoNum{algo_num}")
        return {"func": function_name, "algo_num": algo_num, "run": run_i, "error": np.inf, "count": 0, "time": np.nan, "algo_name": algo_name}
    final_optima_run = None
    get_score_for_run = get_score 
    run_successful = False
    final_validated_optima = []
    num_distinct_minima_this_run = 0
    duration_this_run = np.nan
    algo_name = "Unknown" 

    try:
        start_time = time.time()
        
        if algo_num not in algo_params:
            raise ValueError(f"Algorithm number '{algo_num}' not found in algo_params.")
        params = algo_params[algo_num]
        if algo_num == "1": algo_name = "BA (Standard)"
        elif algo_num == "2": algo_name = "LORRE-BA"
        elif algo_num == "3": algo_name = "BA-SN"
        elif algo_num == "4": algo_name = "BA-FS"
        elif algo_num == "5": algo_name = "EA-FS"
        elif algo_num == "6": algo_name = "EA-SN"
        else: algo_name = f"InvalidAlgoNum{algo_num}" 
        if algo_num == "3": 
            final_optima_run = bees_algorithm_sn(bc_func, params['bounds'], params['max_iter'], params['radius'], params['n_sites'], params['n_elite'], params['sel_bees'], params['elite_bees'], params['stlim'], params['pruning'])
        elif algo_num == "2": 
            def score_for_lorre(x): val = get_score(x); return -val if np.isfinite(val) else -np.inf
            alg = LORRE(score_function=score_for_lorre, range_min=params['bounds'][0], range_max=params['bounds'][1])
            alg.performFullOptimisation(max_iteration=params['max_iter'])
            raw_optima_lorre = alg.getFoundOptima(pruning_functions=params['pruning'])
            if raw_optima_lorre:
                temp_list = []
                for opt in raw_optima_lorre:
                    point = None
                    if isinstance(opt, np.ndarray): point = opt.tolist()
                    elif hasattr(opt, 'values') and isinstance(getattr(opt, 'values', None), (np.ndarray, list, tuple)): point = list(opt.values)
                    elif isinstance(opt, (list, tuple)): point = list(opt)
                    if point and len(point) == n_dims: temp_list.append(point)
                final_optima_run = temp_list
            else: final_optima_run = []
        elif algo_num == "4": 
            final_optima_run = bees_algorithm_fs(get_score, params['bounds'], params['max_iter'], params['radius'], params['alpha'], params['n_sites'], params['n_elite'], params['sel_bees'], params['elite_bees'], params['stlim'])
        elif algo_num == "1": 
            opt_dict = bees_algorithm_standard(bc_func, params['bounds'], params['max_iter'], params['n_sites'], params['n_elite'], params['sel_bees'], params['elite_bees'], params['stlim'])
            final_optima_run = [opt_dict] if opt_dict else []
        elif algo_num == "5": 
            optima_list_ea_fs, _, _, _, _ = run_evolution(function_name, params['pop_size'], params['n_offspr'], params['gens'], params['bounds'], params['radius'], params['alpha'])
            if optima_list_ea_fs is None: raise RuntimeError(f"EA:FS run_evolution failed for {function_name}")
            final_optima_run = optima_list_ea_fs
        elif algo_num == "6": 
            raw_optima_ea_sn, _, bc_func_ea_sn, min_f_ea_sn, max_f_ea_sn = run_evolution_sn(function_name, params['pop_size'], params['n_offspr'], params['gens'], params['bounds'], params['radius'])
            if raw_optima_ea_sn is None: raise RuntimeError(f"EA:SN run_evolution_sn failed for {function_name}")
            final_optima_run = [opt['centre'] for opt in raw_optima_ea_sn if opt and 'centre' in opt]

            range_ea_sn = max_f_ea_sn - min_f_ea_sn if abs(max_f_ea_sn - min_f_ea_sn) >= 1e-14 else 1e-14
            min_f_ea_sn_stable = min_f_ea_sn
            def get_score_ea_sn(x):
                 try:
                     x_arr = np.asarray(x);
                     if x_arr.shape[-1] != n_dims: raise ValueError("Dim mismatch")
                     raw = bc_func_ea_sn(x_arr);
                     if np.iscomplexobj(raw): raw = np.real(raw)
                     if abs(range_ea_sn) < 1e-14: score = 0.5 if abs(raw - min_f_ea_sn_stable) < 1e-9 else (1.0 if raw > min_f_ea_sn_stable else 0.0)
                     else: score = (raw - min_f_ea_sn_stable) / range_ea_sn
                     return np.clip(score, 0.0, 1.0)
                 except Exception: return np.nan
            get_score_for_run = get_score_ea_sn 
        else:
             raise ValueError(f"Unhandled algorithm number '{algo_num}' during execution.")


        end_time = time.time()
        duration_this_run = end_time - start_time
        run_successful = True

    except Exception as e:
        print(f"!!!!!!!! ERROR during run {run_i+1} for Algo={algo_name} on {function_name} !!!!!!!!!!")
        print(f"Error details: {e}")
        run_successful = False 
    run_error = np.inf 
    if run_successful:
        current_pruning_percentage = FUNCTION_PRUNING_PERCENTAGES.get(function_name, DEFAULT_PRUNING_PERCENTAGE)
        cutoff_score = current_pruning_percentage / 100.0
        points_to_validate = []
        raw_points_extracted = 0
        score_pruning_applied = False
        kept_score_pruning = 0
        rejected_score_pruning = 0
        score_func_for_pruning = get_score_for_run

        extracted_points_for_processing = []
        temp_optima_list_proc = final_optima_run if isinstance(final_optima_run, (list, np.ndarray)) else []
        if not temp_optima_list_proc and isinstance(final_optima_run, dict):
             temp_optima_list_proc = [final_optima_run]
        for item_proc in temp_optima_list_proc:
             pt_proc = item_proc.get("centre") if isinstance(item_proc, dict) else item
             if pt_proc is not None and hasattr(pt_proc, '__len__') and len(pt_proc) == n_dims:
                 extracted_points_for_processing.append(list(pt_proc))
        raw_points_extracted = len(extracted_points_for_processing)

        if algo_num in ["2", "3", "4", "5", "6"]:
            score_pruning_applied = True
            if raw_points_extracted > 0:
                for point in extracted_points_for_processing:
                    try:
                        point_score = score_func_for_pruning(np.asarray(point))
                        if np.isfinite(point_score) and point_score <= cutoff_score:
                            points_to_validate.append(point); kept_score_pruning += 1
                        else: rejected_score_pruning += 1
                    except Exception as score_err_initial_prune: rejected_score_pruning += 1
            points_to_validate = points_to_validate
        else:
            points_to_validate = extracted_points_for_processing
        final_validated_optima = []
        if points_to_validate:
            dist_thresh = 0.1
            score_func_for_validation = score_func_for_pruning if score_pruning_applied else get_score
            final_validated_optima = validate_optima(points_to_validate, score_func_for_validation,
                                                     n_dims, 
                                                     bounds_normalized, function_name, algo_name,
                                                     distinctness_threshold=dist_thresh, verbose=False)

        num_distinct_minima_this_run = len(final_validated_optima)
        
        run_error = calculate_error_metric(final_validated_optima, true_minima_norm)
        if final_validated_optima:
            plot_results(final_validated_optima, bounds_normalized, function_name, get_score,
                         f"{algo_name}", save_svg=True, algo_name_clean=algo_name,
                         run_number=run_i + 1, show_plots=False)

        print(f"Finished: Func={function_name}, Algo={algo_name}, Run={run_i+1}/{total_runs} -> Err={run_error:.4f}, Count={num_distinct_minima_this_run}, Time={duration_this_run:.2f}s")
    
    return {
        "func": function_name,
        "algo_num": algo_num, 
        "run": run_i,
        "error": run_error,
        "count": num_distinct_minima_this_run,
        "time": duration_this_run,
        "algo_name": algo_name 
    }
def main():
    
    population_size_ea = 500; n_offsprings_ea = 50; generations_ea = 100
    initial_niche_radius_ea = 0.1; alpha_ea = 2
    max_iterations_ba_sn = 1000; niching_radius_ba_sn = 0.35
    num_sites_ba_sn = 16; num_elite_ba_sn = 8; selected_bees_ba_sn = 4
    elite_bees_ba_sn = 8; stlim_ba_sn = 20
    pruning_functions_ba_sn = [PruningPercScoreCutoff(cutoff=1)]
    max_iterations_lorre = 1000
    pruning_functions_lorre = [PruningProximity(), LORREPruningPercScoreCutoff(cutoff=1)]
    max_iterations_ba_fs = 1000; niche_radius_ba_fs = 0.1; alpha_ba_fs = 2
    num_sites_ba_fs = 16; num_elite_ba_fs = 8; selected_bees_ba_fs = 4
    elite_bees_ba_fs = 8; stlim_ba_fs = 20
    max_iterations_ba = 1000; num_sites_ba = 16; num_elite_ba = 8
    selected_bees_ba = 4; elite_bees_ba = 8; stlim_ba = 20
    population_size_ea_sn = 500; n_offsprings_ea_sn = 50; generations_ea_sn = 100
    niche_radius_ea_sn = 1
    bounds_normalized = ([-1.0, -1.0], [1.0, 1.0]) 
    bounds_normalized_list = [(-1.0, 1.0), (-1.0, 1.0)] 
    
    all_algo_params = {
        "1": {"bounds": bounds_normalized, "max_iter": max_iterations_ba, "n_sites": num_sites_ba, "n_elite": num_elite_ba, "sel_bees": selected_bees_ba, "elite_bees": elite_bees_ba, "stlim": stlim_ba},
        "2": {"bounds": bounds_normalized, "max_iter": max_iterations_lorre, "pruning": pruning_functions_lorre},
        "3": {"bounds": bounds_normalized, "max_iter": max_iterations_ba_sn, "radius": niching_radius_ba_sn, "n_sites": num_sites_ba_sn, "n_elite": num_elite_ba_sn, "sel_bees": selected_bees_ba_sn, "elite_bees": elite_bees_ba_sn, "stlim": stlim_ba_sn, "pruning": pruning_functions_ba_sn},
        "4": {"bounds": bounds_normalized, "max_iter": max_iterations_ba_fs, "radius": niche_radius_ba_fs, "alpha": alpha_ba_fs, "n_sites": num_sites_ba_fs, "n_elite": num_elite_ba_fs, "sel_bees": selected_bees_ba_fs, "elite_bees": elite_bees_ba_fs, "stlim": stlim_ba_fs},
        "5": {"bounds": bounds_normalized_list, "pop_size": population_size_ea, "n_offspr": n_offsprings_ea, "gens": generations_ea, "radius": initial_niche_radius_ea, "alpha": alpha_ea},
        "6": {"bounds": bounds_normalized_list, "pop_size": population_size_ea_sn, "n_offspr": n_offsprings_ea_sn, "gens": generations_ea_sn, "radius": niche_radius_ea_sn}
    }
    NUM_RUNS = 10
    
    bc_functions_to_run = ["BC_1", "BC_2", "BC_3", "P&H", "Rastrigin", "Schwefel"]

    print("\nAvailable algorithms:")
    print("1. BA (Standard)")
    print("2. LORRE")
    print("3. BA:SN")
    print("4. BA:FS")
    print("5. EA:FS")
    print("6. EA:SN")
    print("7. Run All")

    selected_algorithm = input("Enter the number(s) of the algorithm(s) you want to run (comma-separated, or 7 for all): ")

    algo_choices = selected_algorithm.split(',')
    run_all = "7" in [choice.strip() for choice in algo_choices]
    valid_choices = ["1", "2", "3", "4", "5", "6"]
    if run_all:
        algorithms_to_run_nums = valid_choices
        print("Running ALL selected algorithms.")
    else:
        algorithms_to_run_nums = [choice.strip() for choice in algo_choices if choice.strip() in valid_choices]
        print(f"Selected algorithms numbers: {', '.join(algorithms_to_run_nums)}")

    if not algorithms_to_run_nums:
        print("No valid algorithm selected. Exiting.")
        return
    results_summary = {} 
    minima_counts_summary = {} 
    run_times_summary = {} 
    tasks = []
    for func_name in bc_functions_to_run:
        true_minima_norm = TRUE_MINIMA_DB.get(func_name)
        if true_minima_norm is None or len(true_minima_norm) == 0:
             print(f"WARNING: True minima not defined for {func_name}. Skipping this function for all algorithms.")
             continue 
        for algo_num in algorithms_to_run_nums:
            for run_i in range(NUM_RUNS):
                task_args = (
                    func_name,
                    algo_num,
                    run_i,
                    NUM_RUNS,
                    bounds_normalized, 
                    bounds_normalized_list, 
                    true_minima_norm, 
                    all_algo_params 
                )
                tasks.append(task_args)

    if not tasks:
        print("No valid tasks generated (check functions and algorithms). Exiting.")
        return
    num_workers = multiprocessing.cpu_count()
    print(f"\nDetected {num_workers} CPU cores. Creating worker pool...")
    
    num_workers = 10 

    print(f"Starting {len(tasks)} tasks using {num_workers} worker processes...")
    start_parallel_time = time.time()
    all_results = []
    try:
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            all_results = pool.starmap(run_single_test, tasks)
    except Exception as pool_err:
        print(f"\n!!!!!!!! FATAL ERROR during parallel processing !!!!!!!!")
        print(f"Error: {pool_err}")
        traceback.print_exc()
        print("Attempting to aggregate any partial results...")

    end_parallel_time = time.time()
    print(f"\n--- Parallel execution finished in {end_parallel_time - start_parallel_time:.2f} seconds ---")
    print("Aggregating results...")
    algo_name_map = {"1": "BA (Standard)", "2": "LORRE-BA", "3": "BA-SN", "4": "BA-FS", "5": "EA-FS", "6": "EA-SN", "Error": "Error"} 

    for result in all_results:
        if result is None: continue 

        func_name = result["func"]
        algo_num = result["algo_num"]
        algo_name = result.get("algo_name", algo_name_map.get(algo_num, "Unknown"))

        if algo_name == "Error" or algo_name == "Unknown": continue 
        results_summary.setdefault(func_name, {}).setdefault(algo_name, [])
        minima_counts_summary.setdefault(func_name, {}).setdefault(algo_name, [])
        run_times_summary.setdefault(func_name, {}).setdefault(algo_name, [])
        results_summary[func_name][algo_name].append(result["error"])
        minima_counts_summary[func_name][algo_name].append(result["count"])
        run_times_summary[func_name][algo_name].append(result["time"])
    print("\n\n=================================================================================================")
    print("                               FINAL RESULTS SUMMARY (Error Metric, Minima Count, Time)")
    print("=================================================================================================")
    print(f"{'Function':<15} {'Algorithm':<15} {'Runs':<6} {'Minima':<8} {'Avg Time (s)':<14} {'Min Err':<10} {'1st Qu.':<10} {'Median Err':<10} {'3rd Qu.':<10} {'Max Err':<10}")
    print("-" * 120)

    summary_table = []
    all_processed_funcs = set(results_summary.keys()) | set(minima_counts_summary.keys()) | set(run_times_summary.keys())
    sorted_func_names = sorted(list(all_processed_funcs))

    all_algo_names_run = set()
    for func_name in sorted_func_names:
        if func_name in results_summary: all_algo_names_run.update(results_summary[func_name].keys())
        if func_name in minima_counts_summary: all_algo_names_run.update(minima_counts_summary[func_name].keys())
        if func_name in run_times_summary: all_algo_names_run.update(run_times_summary[func_name].keys())
    algo_name_map_for_sort = {"BA (Standard)": "1", "LORRE-BA": "2", "BA-SN": "3", "BA-FS": "4", "EA-FS": "5", "EA-SN": "6"}
    sorted_algo_names = sorted(list(all_algo_names_run), key=lambda name: algo_name_map_for_sort.get(name, '99'))


    for func_name in sorted_func_names:
        for algo_name in sorted_algo_names:
            median_count = np.nan; avg_time = np.nan
            min_err, q1_err, med_err, q3_err, max_err = (np.nan,) * 5
            runs_str = f"0/{NUM_RUNS}" 

            if func_name in results_summary and algo_name in results_summary[func_name]:
                errors = results_summary[func_name][algo_name]
                num_runs_recorded = len(errors) 
                if errors:
                    errors_arr = np.array(errors); finite_errors = errors_arr[np.isfinite(errors_arr)]
                    num_finite_runs = len(finite_errors)
                    runs_str = f"{num_finite_runs}/{num_runs_recorded}"
                    if num_finite_runs > 0:
                        min_err, q1_err, med_err, q3_err, max_err = (np.min(finite_errors), np.percentile(finite_errors, 25), np.median(finite_errors), np.percentile(finite_errors, 75), np.max(finite_errors))
                    elif num_runs_recorded > 0: min_err, q1_err, med_err, q3_err, max_err = (np.inf,) * 5

            if func_name in minima_counts_summary and algo_name in minima_counts_summary[func_name]:
                 counts = minima_counts_summary[func_name][algo_name]
                 if counts: median_count = np.max(np.array(counts))

            if func_name in run_times_summary and algo_name in run_times_summary[func_name]:
                 times = run_times_summary[func_name][algo_name]
                 if times:
                     times_arr = np.array(times); valid_times = times_arr[np.isfinite(times_arr)]
                     if valid_times.size > 0: avg_time = np.mean(valid_times)

            summary_table.append([func_name, algo_name, runs_str, median_count, avg_time, min_err, q1_err, med_err, q3_err, max_err])

    for row in summary_table:
        runs_str = row[2]; med_count_val = row[3]; avg_time_val = row[4]
        med_count_str = f"{int(med_count_val)}" if np.isfinite(med_count_val) else "nan"
        avg_time_str = f"{avg_time_val:.3f}" if np.isfinite(avg_time_val) else "nan"
        min_str = f"{row[5]:.4f}" if np.isfinite(row[5]) else ("inf" if row[5] == np.inf else "nan")
        q1_str  = f"{row[6]:.4f}" if np.isfinite(row[6]) else ("inf" if row[6] == np.inf else "nan")
        med_str = f"{row[7]:.4f}" if np.isfinite(row[7]) else ("inf" if row[7] == np.inf else "nan")
        q3_str  = f"{row[8]:.4f}" if np.isfinite(row[8]) else ("inf" if row[8] == np.inf else "nan")
        max_str = f"{row[9]:.4f}" if np.isfinite(row[9]) else ("inf" if row[9] == np.inf else "nan")
        print(f"{row[0]:<15} {row[1]:<15} {runs_str:<6} {med_count_str:<8} {avg_time_str:<14} {min_str:<10} {q1_str:<10} {med_str:<10} {q3_str:<10} {max_str:<10}")
    print("-" * 120)
    
    csv_filename = "results_summary.csv"
    print(f"\nExporting summary table to {csv_filename}...")
    try:
        header = ['Function', 'Algorithm', 'Runs', 'Median Minima', 'Avg Time (s)', 'Min Error', '1st Qu. Error', 'Median Error', '3rd Qu. Error', 'Max Error']
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            formatted_summary_table = []
            for row in summary_table:
                formatted_row = []
                for item in row:
                    if isinstance(item, float):
                        if np.isnan(item): formatted_row.append('nan')
                        elif item == np.inf: formatted_row.append('inf')
                        elif item == -np.inf: formatted_row.append('-inf')
                        else: formatted_row.append(item)
                    else: formatted_row.append(item)
                formatted_summary_table.append(formatted_row)
            writer.writerows(formatted_summary_table)
        print(f"Successfully exported results to {csv_filename}")
    except Exception as e: print(f"ERROR: Failed to export results to CSV: {e}")
    
    print("\nGenerating Box Plot(s) for Results (sequentially)...")
    plotted_functions = set()
    sorted_func_names_plot = [fn for fn in sorted_func_names if results_summary.get(fn)]
    boxplot_title_fontsize = 26
    boxplot_label_fontsize = 24
    boxplot_tick_fontsize = 18
    boxplot_xtick_fontsize = 24

    for func_name in sorted_func_names_plot:
        target_function = func_name  
        print(f"  Generating Box Plot for {target_function}...")

        data_to_plot = [];
        labels_for_plot = []
        if target_function in results_summary:
            algorithms_in_summary = [name for name in sorted_algo_names if name in results_summary[target_function]]
            for algo_name in algorithms_in_summary:
                errors = results_summary[target_function].get(algo_name, [])
                finite_errors = [e for e in errors if np.isfinite(e)]
                if finite_errors:
                    data_to_plot.append(finite_errors)
                    labels_for_plot.append(algo_name)
                else:
                    print(f"    Skipping {algo_name} for {target_function} plot (no finite error data).")
        else:
            print(f"    Skipping plot for {target_function} (no error results found).");
            continue

        if data_to_plot:
            fig, ax = plt.subplots(figsize=(10, 7))  
            bp = ax.boxplot(data_to_plot,
                            patch_artist=False,
                            showfliers=False,  
                            labels=labels_for_plot,
                            whis=[0, 100])  
            colors = ['black', 'blue'];
            for i, box in enumerate(bp['boxes']): box.set_color(colors[i % 2])
            for i, whisker in enumerate(bp['whiskers']): whisker.set_color(colors[(i // 2) % 2]); whisker.set_linestyle(
                '-')
            for i, cap in enumerate(bp['caps']): cap.set_color(colors[(i // 2) % 2])
            for median in bp['medians']: median.set_color('red'); median.set_linewidth(1.5)
            

            plt.xticks(rotation=30, ha='right', fontsize=boxplot_xtick_fontsize)  
            ax.tick_params(axis='y', which='major', labelsize=boxplot_tick_fontsize)  
            ax.set_title(f'Algorithm Performance Comparison on {target_function}', fontsize=boxplot_title_fontsize)
            ax.set_ylabel('Error', fontsize=boxplot_label_fontsize)
            
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            try:
                safe_func_name = target_function.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
                svg_filename_boxplot = f"{safe_func_name}_boxplot.svg"

                print(f"    Saving Box Plot to {svg_filename_boxplot}")
                fig.savefig(svg_filename_boxplot, format='svg')#, bbox_inches='tight')

            except Exception as e:
                print(f"    WARNING: Failed to save box plot SVG '{svg_filename_boxplot}': {e}")

            plt.close(fig) 
            plotted_functions.add(target_function)
            print(f"    Box plot for {target_function} displayed and saved.")
        else: print(f"    No valid data found to generate box plot for {target_function}.")

    if not plotted_functions: print("No results found for any function to generate box plots.")
if __name__ == "__main__":
    main()
