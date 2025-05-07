import numpy as np
import matplotlib.pyplot as plt
from ba_lorre import LORRE, PruningProximity 
import time
import math
from tabulate import tabulate 
import csv                     
import os
import numpy as np
import matplotlib.pyplot as plt
from ba_lorre import LORRE, PruningProximity 
import time
import math
from tabulate import tabulate 
import csv                     
import os                      
import matplotlib              
try:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    
    matplotlib.rcParams['font.sans-serif'] = ['Arial'] + matplotlib.rcParams['font.sans-serif']
    print("INFO: Attempted to set default plot font to Arial.")
except Exception as e:
    print(f"WARNING: Could not set default font to Arial. Using Matplotlib defaults. Error: {e}")
E_MOD = 68947.6  
DENSITY = 2.76799e-6 
STRESS_LIM = 137.895 
DISP_LIM = 50.8    
AMAX = 1451.61       
epsilon = 32.258    
PRESENCE_THRESHOLD = epsilon 
CENTER_X = 6096.0 
OUTPUT_DIR = "truss_results_metric"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")
nodes_coord = np.array([
    [0.0,    0.0], [3048.0,    0.0], [6096.0,    0.0], [9144.0,    0.0], [12192.0, 0.0],
    [0.0, 3048.0], [3048.0, 3048.0],                 [9144.0, 3048.0], [12192.0, 3048.0],
                [3048.0, 6096.0], [6096.0, 6096.0], [9144.0, 6096.0]
]) * 1.0 
N_NODES = len(nodes_coord)
print(f"Using node layout: 5 Bottom, 4 Middle (Aligned), 3 Top ({N_NODES} total nodes) - METRIC")
potential_members_systematic = []
for i in range(N_NODES):
    for j in range(i + 1, N_NODES):
        potential_members_systematic.append(tuple(sorted((i, j))))
potential_members = potential_members_systematic
N_MEMBERS_POTENTIAL = len(potential_members)
print(f"Defined COMPREHENSIVE {N_NODES}-node, {N_MEMBERS_POTENTIAL}-potential-member ground structure.")
node_mirror_map = {}
mirrored_nodes = set()
centerline_nodes = []
node_pairs = []
for i in range(N_NODES):
    if i in mirrored_nodes:
        continue
    x_i, y_i = nodes_coord[i]
    if math.isclose(x_i, CENTER_X):
        node_mirror_map[i] = i
        mirrored_nodes.add(i)
        centerline_nodes.append(i)
    else:
        mirror_x = 2 * CENTER_X - x_i
        found_mirror = False
        for j in range(i + 1, N_NODES):
             if j in mirrored_nodes:
                 continue
             x_j, y_j = nodes_coord[j]
             if math.isclose(x_j, mirror_x) and math.isclose(y_j, y_i):
                node_mirror_map[i] = j
                node_mirror_map[j] = i
                mirrored_nodes.add(i)
                mirrored_nodes.add(j)
                node_pairs.append(tuple(sorted((i,j))))
                found_mirror = True
                break
        if not found_mirror and i not in mirrored_nodes:
             print(f"Critical Warning: No mirror found for node {i} at ({x_i}, {y_i})")
             node_mirror_map[i] = i
             mirrored_nodes.add(i)
             centerline_nodes.append(i)
left_indices = []
center_indices = []
member_map = {}
potential_members_tuples = [tuple(sorted(m)) for m in potential_members]
potential_members_map = {mem_tuple: idx for idx, mem_tuple in enumerate(potential_members_tuples)}
processed_indices = set()
warnings = 0
for idx, (n1, n2) in enumerate(potential_members):
    if idx in processed_indices:
        continue
    try:
        n1_mir = node_mirror_map[n1]
        n2_mir = node_mirror_map[n2]
    except KeyError as e:
        print(f"Error: Node {e} not in node_mirror_map during member classification.")
        continue

    mirror_member_tuple = tuple(sorted((n1_mir, n2_mir)))

    if mirror_member_tuple == tuple(sorted((n1, n2))): 
        center_indices.append(idx)
        member_map[idx] = idx
        processed_indices.add(idx)
    elif mirror_member_tuple in potential_members_map: 
        mirror_idx = potential_members_map[mirror_member_tuple]
        if mirror_idx != idx:
            member_center_x = (nodes_coord[n1, 0] + nodes_coord[n2, 0]) / 2.0
            if member_center_x < CENTER_X - 1e-6 or (math.isclose(member_center_x, CENTER_X) and idx < mirror_idx):
                 left_indices.append(idx)
                 member_map[idx] = mirror_idx
                 member_map[mirror_idx] = idx
            processed_indices.add(idx)
            processed_indices.add(mirror_idx)
        else: 
             center_indices.append(idx)
             member_map[idx] = idx
             processed_indices.add(idx)
    else: 
        print(f"Warning: Mirror member {mirror_member_tuple} for member {idx} {(n1, n2)} not found in potential list.")
        center_indices.append(idx)
        member_map[idx] = idx 
        processed_indices.add(idx)
        warnings += 1

if warnings > 0:
    print(f"Finished symmetry analysis with {warnings} mirror member warnings.")

optimizable_indices = sorted(left_indices + center_indices)
N_SYMM_VARS = len(optimizable_indices)

print(f"\nSymmetry analysis: {len(left_indices)} left members, {len(center_indices)} center members.")
print(f"Number of optimization variables (Symmetry enforced): {N_SYMM_VARS}")
loads = np.zeros((N_NODES, 2))
load_nodes = [9, 10, 11]
LOAD_MAG_N = -44482.2 
for node_idx in load_nodes:
    loads[node_idx, 1] = LOAD_MAG_N
fixed_dofs = [0, 1, 9] 
def analyze_truss(elements, areas):
    n_elements = len(elements)
    if n_elements == 0:
        return False, np.zeros(N_NODES * 2), np.array([]), 0.0, np.array([])

    K_global = np.zeros((N_NODES * 2, N_NODES * 2))
    elem_lengths = np.zeros(n_elements)
    elem_stresses = np.zeros(n_elements)

    for i, (n1, n2) in enumerate(elements):
        A = areas[i]
        node1_coord = nodes_coord[n1]
        node2_coord = nodes_coord[n2]
        L = np.linalg.norm(node2_coord - node1_coord)
        if L < 1e-6:
            elem_lengths[i] = 0
            continue
        elem_lengths[i] = L
        angle = np.arctan2(node2_coord[1]-node1_coord[1], node2_coord[0]-node1_coord[0])
        c = np.cos(angle)
        s = np.sin(angle)
        c2 = c*c
        s2 = s*s
        cs = c*s
        k_global_elem=(E_MOD*A/L)*np.array([[c2,cs,-c2,-cs],[cs,s2,-cs,-s2],[-c2,-cs,c2,cs],[-cs,-s2,cs,s2]])
        dofs_map = [n1*2, n1*2+1, n2*2, n2*2+1]
        K_global[np.ix_(dofs_map, dofs_map)] += k_global_elem

    all_dofs = np.arange(N_NODES * 2)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    if len(free_dofs) == 0:
        return True, np.zeros(N_NODES * 2), np.zeros(n_elements), 0.0, elem_lengths

    try:
        K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
        if K_reduced.shape[0] == 0 or K_reduced.size == 0:
            return False, np.zeros(N_NODES * 2), np.zeros(n_elements), 0.0, elem_lengths
        condK = np.linalg.cond(K_reduced)
        if condK > 1e10:
            return False, np.zeros(N_NODES * 2), np.zeros(n_elements), 0.0, elem_lengths

        F_reduced = loads.flatten()[free_dofs]
        disp_reduced = np.linalg.solve(K_reduced, F_reduced)
        displacements_full = np.zeros(N_NODES * 2)
        displacements_full[free_dofs] = disp_reduced
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"FEM Error: {e}")
        return False, np.zeros(N_NODES * 2), np.zeros(n_elements), 0.0, elem_lengths

    for i, (n1, n2) in enumerate(elements):
        if elem_lengths[i] < 1e-6:
            elem_stresses[i] = 0
            continue
        L=elem_lengths[i]
        angle = np.arctan2(nodes_coord[n2,1]-nodes_coord[n1,1], nodes_coord[n2,0]-nodes_coord[n1,0])
        c = np.cos(angle)
        s = np.sin(angle)
        u1 = displacements_full[n1*2]
        v1 = displacements_full[n1*2+1]
        u2 = displacements_full[n2*2]
        v2 = displacements_full[n2*2+1]
        d_local = np.array([-c, -s, c, s]) @ np.array([u1, v1, u2, v2])
        elem_stresses[i] = E_MOD * (d_local / L)

    return True, displacements_full, elem_stresses, 0.0, elem_lengths
def truss_combined_score_function(candidate_symm_vars):
    full_potential_areas = np.zeros(N_MEMBERS_POTENTIAL)
    potential_areas = np.array(candidate_symm_vars)
    current_area_idx = 0
    for left_idx in left_indices:
        area = potential_areas[current_area_idx]
        full_potential_areas[left_idx] = area
        right_idx = member_map.get(left_idx)
        if right_idx is not None and right_idx != left_idx:
            full_potential_areas[right_idx] = area
        current_area_idx += 1
    for center_idx in center_indices:
         area = potential_areas[current_area_idx]
         full_potential_areas[center_idx] = area
         current_area_idx += 1

    existing_member_indices = np.where(full_potential_areas >= epsilon)[0]
    if len(existing_member_indices) == 0:
        return -1e12

    current_elements = [potential_members[i] for i in existing_member_indices]
    current_actual_areas = full_potential_areas[existing_member_indices]

    is_stable, displacements, stresses, _, elem_lengths = analyze_truss(current_elements, current_actual_areas)

    weight = 0.0
    if is_stable and len(elem_lengths) == len(current_elements):
        valid_indices = np.where(elem_lengths > 1e-6)[0]
        if len(valid_indices) > 0:
            valid_areas = current_actual_areas[valid_indices]
            weight = np.sum(DENSITY * elem_lengths[valid_indices] * valid_areas) 

    penalty = 0.0
    if not is_stable:
        penalty += 1e8
        return - (weight + penalty)

    if len(stresses) > 0:
        penalty += 1e5 * np.sum(np.maximum(0, np.abs(stresses)/STRESS_LIM - 1.0))

    disp_mags = np.linalg.norm(displacements.reshape((N_NODES, 2)), axis=1)
    penalty += 1e5 * np.sum(np.maximum(0, disp_mags / DISP_LIM - 1.0))

    penalty += 1e5 * np.sum(np.maximum(0, current_actual_areas - AMAX))

    score = - (weight + penalty)

    if np.isnan(score):
        return -1e12
    else:
        return score
print(f"\nOptimizing COMBINED SIZING & SYMMETRIC {N_MEMBERS_POTENTIAL}-potential-member topology ({N_SYMM_VARS} variables)...")
print(f"Area variable range: [0.0, {AMAX:.2f} mm^2], Epsilon threshold: {epsilon:.2f} mm^2")

params = {
    'nb': 15,
    'nrb': 20,
    'stlim': 40,
    
    'derating_type': 'linear'
}

lorre_alg = LORRE(
    score_function=truss_combined_score_function,
    range_min=[0.0] * N_SYMM_VARS,
    range_max=[AMAX] * N_SYMM_VARS,
    **params
)

start_time = time.time()
n_iterations, best_score = lorre_alg.performFullOptimisation(max_iteration=1000)
end_time = time.time()

print(f"\nOptimization finished after {n_iterations} iterations in {end_time - start_time:.2f} seconds.")
print(f"Best score found: {best_score}")
found_optima_raw_symm = []
try:
    found_optima_raw_symm = lorre_alg.getFoundOptima()
except Exception as e:
    print(f"\nWarning: Error retrieving optima: {e}")

found_optima_pruned_symm = []
if found_optima_raw_symm:
    try:
        found_optima_pruned_symm = lorre_alg.getFoundOptima(pruning_functions=[PruningProximity()])
    except Exception as e:
        print(f"\nWarning: Error pruning optima: {e}")
        found_optima_pruned_symm = found_optima_raw_symm 
else:
    print("Skipping pruning.")

print(f"\nProceeding analysis with {len(found_optima_pruned_symm)} candidate symmetric solutions.")
final_solutions = []
evaluated_topologies_sizing = set()
print("\n--- Evaluating Candidate Solutions (Combined Sizing & Topology) ---")
evaluated_count = 0

for i, solution_symm_vars in enumerate(found_optima_pruned_symm):
    full_potential_areas = np.zeros(N_MEMBERS_POTENTIAL)
    potential_areas = np.array(solution_symm_vars)
    current_area_idx = 0
    for left_idx in left_indices:
        area = potential_areas[current_area_idx]
        full_potential_areas[left_idx] = area
        right_idx = member_map.get(left_idx)
        if right_idx is not None and right_idx != left_idx:
            full_potential_areas[right_idx] = area
        current_area_idx += 1
    for center_idx in center_indices:
         area = potential_areas[current_area_idx]
         full_potential_areas[center_idx] = area
         current_area_idx += 1
    presence_vector_int = (full_potential_areas >= epsilon).astype(int)
    existing_member_indices = np.where(presence_vector_int == 1)[0]
    topology_tuple = tuple(presence_vector_int)
    if topology_tuple in evaluated_topologies_sizing:
        continue
    evaluated_topologies_sizing.add(topology_tuple)
    evaluated_count += 1

    if len(existing_member_indices) == 0:
        continue

    final_elements = [potential_members[idx] for idx in existing_member_indices]
    final_areas = full_potential_areas[existing_member_indices]
    is_stable, displacements, stresses, _, elem_lengths = analyze_truss(final_elements, final_areas)

    weight = 0.0
    if is_stable and len(elem_lengths) == len(final_elements):
         valid_indices = np.where(elem_lengths > 1e-6)[0]
         if len(valid_indices) > 0:
             valid_areas = final_areas[valid_indices]
             weight = np.sum(DENSITY * elem_lengths[valid_indices] * valid_areas) 

    if not is_stable:
        continue
    max_stress = 0.0
    if len(stresses) > 0:
        max_stress = np.max(np.abs(stresses))
    max_disp = np.max(np.linalg.norm(displacements.reshape((N_NODES, 2)), axis=1))
    stress_ok = max_stress <= STRESS_LIM * 1.001
    disp_ok = max_disp <= DISP_LIM * 1.001
    if is_stable and stress_ok and disp_ok:
        member_areas_dict = {tuple(sorted(potential_members[idx])): area for idx, area in zip(existing_member_indices, final_areas)}
        final_solutions.append({
            'elements': final_elements,
            'areas_dict': member_areas_dict,
            'weight': weight
        })

print(f"\nEvaluated {evaluated_count} unique topologies.")
print(f"Found {len(final_solutions)} valid final SYMMETRIC topologies with sizing.")
def plot_topology_sized(nodes_mm, elements, areas_dict, title="Truss Topology & Sizing", save_path=None):
    """
    Plots the truss with node coordinates converted to meters for display.
    Internal calculations assume nodes_mm are in millimeters.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    title_fontsize = 20
    label_fontsize = 18
    tick_fontsize = 16
    nodes_m = nodes_mm / 1000.0

    min_plot_area = epsilon 
    max_plot_area = AMAX    
    present_areas = np.array(list(areas_dict.values())) if areas_dict else np.array([]) 

    if len(present_areas) > 0:
        actual_max = np.max(present_areas)
        actual_min = np.min(present_areas)
        max_plot_area = max(AMAX, actual_max)
        min_plot_area = max(epsilon, actual_min)
    else:
        max_plot_area = AMAX
        min_plot_area = epsilon

    min_lw = 0.5
    max_lw = 5.0
    ax.plot(nodes_m[:, 0], nodes_m[:, 1], 'bo', markersize=4)

    for element in elements:
        n1, n2 = element
        area = areas_dict.get(tuple(sorted(element))) 
        if area is None or area < epsilon:
            continue
        if max_plot_area > min_plot_area :
             norm_area = (area - min_plot_area) / (max_plot_area - min_plot_area)
        else:
             norm_area = 0.5

        line_width = min_lw + norm_area * (max_lw - min_lw)
        ax.plot([nodes_m[n1, 0], nodes_m[n2, 0]],
                [nodes_m[n1, 1], nodes_m[n2, 1]],
                'k-', linewidth=line_width)

    support_nodes_indices = [0, 4]
    ax.plot(nodes_m[support_nodes_indices[0], 0], nodes_m[support_nodes_indices[0], 1], 'g^', markersize=8)
    ax.plot(nodes_m[support_nodes_indices[1], 0], nodes_m[support_nodes_indices[1], 1], 'go', markersize=8, markerfacecolor='none')

    ax.set_aspect('equal', adjustable='box')
    
    ax.set_xlabel("X (m)", fontsize=label_fontsize)
    ax.set_ylabel("Y (m)", fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize) 
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize) 

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")
    plt.close(fig) 
final_solutions.sort(key=lambda x: x['weight'])
print("\n--- Top Valid Symmetric Solutions (Combined Sizing & Topology) ---")

num_to_report_console = min(len(final_solutions), 5) 
num_to_save = len(final_solutions)                   

print(f"Found {len(final_solutions)} valid solutions.")
print(f"Saving plots and CSV files for all {num_to_save} valid solutions to '{OUTPUT_DIR}'...")
print(f"Printing details for top {num_to_report_console} solutions to console...")

if num_to_save == 0:
    print("No valid solutions found to save.")
else:
    for i, sol in enumerate(final_solutions): 
        rank = i + 1
        if i < num_to_report_console:
            print(f"\nRank {rank}: Weight={sol['weight']:.3f} kg, Members={len(sol['elements'])}")
        plot_title = f'Preliminary Design {rank} ({sol["weight"]:.1f} kg, {len(sol["elements"])} members)'
        plot_filename = os.path.join(OUTPUT_DIR, f"truss_rank_{rank}.svg")
        plot_topology_sized(nodes_coord, sol['elements'], sol['areas_dict'], plot_title, save_path=plot_filename)
        if (i + 1) % 50 == 0 or i == num_to_save - 1: 
             print(f"  Saved files for Rank {rank}...")
        csv_filename = os.path.join(OUTPUT_DIR, f"truss_rank_{rank}_areas.csv")
        
        table_data = []
        headers = ["Member (N1-N2)", "Area (mm^2)"] 
        sorted_elements = sorted(sol['elements'])
        for element in sorted_elements:
            area = sol['areas_dict'].get(tuple(sorted(element)), 0.0)
            table_data.append([f"{element[0]}-{element[1]}", f"{area:.6f}"]) 

        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                writer.writerows(table_data)
        except IOError as e:
            print(f"  Error writing CSV file {csv_filename}: {e}")
        if i < num_to_report_console:
             print(tabulate(table_data, headers=headers, tablefmt="grid"))
    

print(f"\nFinished saving {num_to_save} plot(s) and CSV file(s).")
print("Done.")