from itertools import product
from configs import default_experiment
from helpers import set_nested_attr
from dataclasses import replace
from copy import deepcopy

sweep_index = 0

def sweep(marked_parameters, global_parameters):
    global sweep_index

    # Extract sweep keys and value lists
    sweep_keys = list(marked_parameters.keys())
    sweep_values = list(marked_parameters.values())

# Generate all combinations (Cartesian product)
    combinations = product(*sweep_values)
    experiments = []

    for values in combinations:
        # Start with default experiment
        # Deep copy so nested dataclasses (like global_parameters) are fresh
        experiment = deepcopy(default_experiment)

        # Set each marked parameter to the corresponding value in this combination
        for key, value in zip(sweep_keys, values):
            # Remove 'hls4ml_config.' prefix if present for direct attributes
            experiment = set_nested_attr(experiment, key, value)
        # Update experiment project name and output dir by index 
        gp = deepcopy(global_parameters)
        experiment_name = f"{gp.project_name}_{sweep_index+1}"
        output_dir_name = f"{gp.ml2hls_project_dir}/{gp.output_dir}_{sweep_index+1}"
        gp = replace(gp, project_name=experiment_name, output_dir=output_dir_name)
        # Set project global parameters
        experiment = replace(experiment, global_parameters=gp)

        # Append the configured experiment to the list
        experiments.append(experiment)
        sweep_index += 1

    print(f"Generated {len(experiments)} experiments.")
    return experiments