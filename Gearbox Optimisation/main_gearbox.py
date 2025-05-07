import numpy as np
from ba_lorre import LORRE
import matplotlib.pyplot as plt
import time
import csv import matplotlib
try:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial'] + matplotlib.rcParams['font.sans-serif']
    print("INFO: Attempted to set default plot font to Arial.")
except Exception as e:
    print(f"WARNING: Could not set default font to Arial. Using Matplotlib defaults. Error: {e}")

start_time = time.time()
MOTOR_SPEEDS = [700, 1000, 1500, 3000]  OUTPUT_SPEEDS = [30, 50, 75, 100]  MIN_GEAR_TEETH = 18
MAX_GEAR_TEETH = 274
MAX_GEAR_RATIO = 5.0
MAX_SHAFTS = 8
MAX_RATIO_TOLERANCE = 0.15
W1 = 0.286
W2 = 1.0
A1 = 0.7925
A2 = 1.0
MAX_ALLOWED_FITNESS = 5000

def calculate_gear_ratio(driven_gear_teeth, driver_gear_teeth):
    """Calculates the gear ratio for a single pair."""
    if driver_gear_teeth == 0:
        return float('inf')     return driven_gear_teeth / driver_gear_teeth


def calculate_speed(input_speed, gear_ratio):
    """Calculates the output speed given input speed and total gear ratio."""
    if gear_ratio == 0:
        return float('inf')     return input_speed / gear_ratio


def evaluate_gearbox(gearbox_config, motor_speed, target_output_speed, num_shafts):
    """Evaluates the fitness of a given gearbox configuration."""
    if not (2 <= num_shafts <= MAX_SHAFTS):
        return 0.0     if len(gearbox_config) != 2 * num_shafts:
         return 0.0

    total_gear_ratio = 1.0
    wrong_pairs = 0
    sum_of_excess = 0

    for i in range(0, len(gearbox_config), 2):
        driver = gearbox_config[i]
        driven = gearbox_config[i + 1]
        if not (MIN_GEAR_TEETH <= driver <= MAX_GEAR_TEETH and MIN_GEAR_TEETH <= driven <= MAX_GEAR_TEETH):
            return 0.0 
        if driven <= driver:
            wrong_pairs += 1
        if driver == 0:
            return 0.0

        gear_ratio = calculate_gear_ratio(driven, driver)
        max_allowed_individual_ratio = MAX_GEAR_RATIO * (1 + MAX_RATIO_TOLERANCE)
        if gear_ratio > max_allowed_individual_ratio:
            sum_of_excess += (gear_ratio - max_allowed_individual_ratio)

        total_gear_ratio *= gear_ratio
    if wrong_pairs > 0:
        return 0.0     penalty_factor_excess = max(0.0, 1.0 - 0.1 * sum_of_excess)
    if penalty_factor_excess == 0:         return 0.0
    if total_gear_ratio == 0:
         return 0.0
    output_speed = calculate_speed(motor_speed, total_gear_ratio)
    if target_output_speed == 0:
        return 0.0
    target_total_ratio = motor_speed / target_output_speed
    ratio_difference = abs(target_total_ratio - total_gear_ratio)
    if ratio_difference > 10:         return 0.0
    fitness_term_shafts = W1 * (3 ** (A1 * num_shafts))
    fitness_term_ratio_diff = W2 * (3 ** (A2 * ratio_difference))
    if fitness_term_shafts + fitness_term_ratio_diff >= MAX_ALLOWED_FITNESS:
        return 0.0 
    fitness = MAX_ALLOWED_FITNESS - fitness_term_shafts - fitness_term_ratio_diff
    fitness *= penalty_factor_excess
    return max(0.0, fitness)


def objective_function(solution):
    """Calculates the maximum fitness across all target output speeds for a solution."""
    try:
        motor_speed_index = int(round(solution[0]))
        if not (0 <= motor_speed_index < len(MOTOR_SPEEDS)):
            return 0.0 
        num_shafts = int(round(solution[1]))
        if not (2 <= num_shafts <= MAX_SHAFTS):
             return 0.0 
        motor_speed = MOTOR_SPEEDS[motor_speed_index]
        num_gear_teeth_values = 2 * num_shafts
        if len(solution) < 2 + num_gear_teeth_values:
             return 0.0
        gear_teeth_raw = solution[2 : 2 + num_gear_teeth_values]
        gear_teeth = []
        for x in gear_teeth_raw:
            rounded_teeth = int(round(x))
            if not (MIN_GEAR_TEETH <= rounded_teeth <= MAX_GEAR_TEETH):
                 return 0.0             gear_teeth.append(rounded_teeth)

    except (IndexError, ValueError) as e:         return 0.0     except Exception as e:
        return 0.0 

    max_fitness = 0
    for target_output_speed in OUTPUT_SPEEDS:
        fitness = evaluate_gearbox(gear_teeth, motor_speed, target_output_speed, num_shafts)
        max_fitness = max(max_fitness, fitness)

    return max_fitness
range_min = [0, 2] + [MIN_GEAR_TEETH] * (2 * MAX_SHAFTS)
range_max = [len(MOTOR_SPEEDS) - 1, MAX_SHAFTS] + [MAX_GEAR_TEETH] * (2 * MAX_SHAFTS)
alg = LORRE(score_function=objective_function,
            range_min=range_min,
            range_max=range_max,
            nb=20,
            nrb=40,
            stlim=5,
            derating_type='linear'
            )

print("Starting LORRE Optimization...")
iterations, best_score = alg.performFullOptimisation(max_iteration=1000)
end_time = time.time()
print(f"Optimization finished.")
print(f"Execution time: {end_time - start_time:.2f} seconds")
print(f"Number of iterations: {iterations}")
print(f"Best score found by LORRE (max fitness for any target speed): {best_score}")

optima = alg.getFoundOptima(pruning_functions=[])
print(f"Number of unique optima found: {len(optima)}")
speed_tolerance = 0.10

results = {
    0:    {speed: 0 for speed in OUTPUT_SPEEDS},
    1000: {speed: 0 for speed in OUTPUT_SPEEDS},
    2000: {speed: 0 for speed in OUTPUT_SPEEDS},
    3000: {speed: 0 for speed in OUTPUT_SPEEDS},
    4000: {speed: 0 for speed in OUTPUT_SPEEDS},
    4990: {speed: 0 for speed in OUTPUT_SPEEDS},
    4991: {speed: 0 for speed in OUTPUT_SPEEDS},
    4992: {speed: 0 for speed in OUTPUT_SPEEDS},
    4993: {speed: 0 for speed in OUTPUT_SPEEDS},
    4994: {speed: 0 for speed in OUTPUT_SPEEDS},
    4995: {speed: 0 for speed in OUTPUT_SPEEDS},
}

all_configurations = {speed: [] for speed in OUTPUT_SPEEDS}

print("\nProcessing found optima...")
processed_count = 0
skipped_count = 0
for i, optimum in enumerate(optima):
    try:
        motor_speed_index = int(round(optimum[0]))
        if not (0 <= motor_speed_index < len(MOTOR_SPEEDS)):
            skipped_count += 1
            continue

        num_shafts = int(round(optimum[1]))
        if not (2 <= num_shafts <= MAX_SHAFTS):
            skipped_count += 1
            continue

        motor_speed = MOTOR_SPEEDS[motor_speed_index]

        num_gear_teeth_values = 2 * num_shafts
        if len(optimum) < 2 + num_gear_teeth_values:
            skipped_count += 1
            continue

        gear_teeth = []
        valid_teeth = True
        for x in optimum[2:2 + num_gear_teeth_values]:
            rounded_teeth = int(round(x))
            if not (MIN_GEAR_TEETH <= rounded_teeth <= MAX_GEAR_TEETH):
                valid_teeth = False
                break
            gear_teeth.append(rounded_teeth)
        if not valid_teeth:
            skipped_count += 1
            continue
        total_gear_ratio = 1.0
        valid_pairs = True
        for k in range(0, len(gear_teeth), 2):
            driver = gear_teeth[k]
            driven = gear_teeth[k + 1] 
            if driver == 0:
                valid_pairs = False
                break
            ratio = calculate_gear_ratio(driven, driver)
            total_gear_ratio *= ratio

        if not valid_pairs or total_gear_ratio == 0 or total_gear_ratio == float('inf'):
            skipped_count += 1
            continue 
        achieved_speed = calculate_speed(motor_speed, total_gear_ratio)
        if achieved_speed == float('inf'):             skipped_count += 1
            continue

        processed_count += 1         for target_speed in OUTPUT_SPEEDS:
            if target_speed == 0: continue             if abs(achieved_speed - target_speed) <= target_speed * speed_tolerance:
                fitness = evaluate_gearbox(gear_teeth, motor_speed, target_speed, num_shafts)
                config_tuple = (motor_speed, num_shafts, gear_teeth)
                all_configurations[target_speed].append((config_tuple, fitness))

                for threshold in results:
                    if fitness >= threshold:
                        results[threshold][target_speed] += 1

    except (IndexError, ValueError) as e:
        skipped_count += 1
        continue
    except Exception as e:
        skipped_count += 1
        continue

print(f"Finished processing optima. Processed: {processed_count}, Skipped: {skipped_count}")
print("\n--- Detailed Configurations Found (within tolerance) ---")
for target_speed in OUTPUT_SPEEDS:
    sorted_configurations = sorted(all_configurations[target_speed], key=lambda x: x[1], reverse=True)
    total_count = len(sorted_configurations)
    print(f"\nConfigurations for Target Speed {target_speed} RPM (Total found: {total_count}):")
    if not sorted_configurations:
        print("  None found within tolerance.")
        continue
    for config, fitness in sorted_configurations:
        motor_speed, num_shafts, gear_teeth = config
        current_total_ratio = 1.0
        valid = True
        for i in range(0, len(gear_teeth), 2):
             if gear_teeth[i] == 0:
                 valid = False; break
             current_total_ratio *= calculate_gear_ratio(gear_teeth[i+1], gear_teeth[i])
        achieved_spd_str = f"{calculate_speed(motor_speed, current_total_ratio):.2f} RPM" if valid and current_total_ratio != 0 else "Invalid Ratio"
        print(f"  Motor={motor_speed}, Shafts={num_shafts}, Fitness={fitness:.2f}, AchievedSpeed={achieved_spd_str}")
        print(f"    Gear Pairs: ", end="")
        pairs_str = [f"({gear_teeth[i]}/{gear_teeth[i+1]})" for i in range(0, len(gear_teeth), 2)]
        print(" * ".join(pairs_str))
print("\n--- Summary Table: Count of Configurations Meeting Fitness Threshold ---")
print("Fitness   ", end="")
for speed in OUTPUT_SPEEDS: print(f"{speed: >9} RPM", end="")
print("\nThreshold ", end="")
for speed in OUTPUT_SPEEDS: print("--------- ", end="")
print()
sorted_thresholds = sorted(results.keys())
for threshold in sorted_thresholds:
    print(f"{threshold: >9} ", end="")
    for target_speed in OUTPUT_SPEEDS: print(f"{results[threshold][target_speed]: >9} ", end="")
    print()
csv_filename = "results_table.csv"
print(f"\nExporting summary table to {csv_filename}...")
try:
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["Fitness Threshold"] + [f"{speed} RPM" for speed in OUTPUT_SPEEDS]
        writer.writerow(header)
        for threshold in sorted_thresholds:
            row = [threshold] + [results[threshold][target_speed] for target_speed in OUTPUT_SPEEDS]
            writer.writerow(row)
    print(f"Successfully exported summary table to {csv_filename}")
except IOError as e:
    print(f"Error writing CSV file {csv_filename}: {e}")
except Exception as e:
    print(f"An unexpected error occurred during CSV export: {e}")
print("\nGenerating plot...")
plt.figure(figsize=(6, 4), facecolor='white')

fitness_threshold_plot = 4995
plot_data = {ts: {ms: 0 for ms in MOTOR_SPEEDS} for ts in OUTPUT_SPEEDS}

for target_speed in OUTPUT_SPEEDS:
    for config, fitness in all_configurations[target_speed]:
        if fitness >= fitness_threshold_plot:
            motor_speed, _, _ = config
            if motor_speed in plot_data[target_speed]:
                 plot_data[target_speed][motor_speed] += 1
            else:
                 print(f"Warning: Motor speed {motor_speed} not in initial MOTOR_SPEEDS list used for plotting.")

colors = {700: '#1f77b4', 1000: '#ff7f0e', 1500: '#2ca02c', 3000: '#d62728'}
bar_width = 0.7
x_positions = np.arange(len(OUTPUT_SPEEDS))
bottom = np.zeros(len(OUTPUT_SPEEDS))

for motor_speed in MOTOR_SPEEDS:
    counts = [plot_data.get(target_speed, {}).get(motor_speed, 0) for target_speed in OUTPUT_SPEEDS]
    plt.bar(x_positions, counts, bottom=bottom, label=f"{motor_speed} RPM", color=colors.get(motor_speed, '#808080'),
            edgecolor='white', linewidth=0.7, width=bar_width)
    bottom += np.array(counts)

plt.xlabel("Target Output Speed (RPM)")
plt.ylabel("Number of Configurations")
plt.title(f"Gearbox Configurations with Fitness Score ≥ {fitness_threshold_plot}")
plt.xticks(x_positions, OUTPUT_SPEEDS)
plt.yticks()
max_height = np.max(bottom) if len(bottom) > 0 and np.max(bottom) > 0 else 10
plt.ylim(0, max_height * 1.15)
plt.legend(title="Input Speed", loc="upper left", frameon=True, framealpha=1, edgecolor='lightgray')
ax = plt.gca()
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('darkgray'); ax.spines['left'].set_color('darkgray')
ax.tick_params(axis='x', colors='black'); ax.tick_params(axis='y', colors='black')
plt.tight_layout()

try:
    plot_filename = "stacked_histogram.svg"
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.show()
except Exception as e:
    print(f"Error saving or showing plot: {e}")

print("\nScript finished.")
