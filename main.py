import os
import importlib.util
# import tensorflow as tf
import numpy as np
import hls4ml
import pprint
from dataclasses import replace

# - - - Custom Modules - - -
from configs import default_experiment, parametrize_hls4ml_config, parametrize_hls4ml_converter, global_config, \
                    remove_experiments, run_experiments
from sweep import sweep

# - - - Global Variables - - -
# experiments = []

## - - - Main - - - ##
def main():
    marked_parameters = {}
    global_parameters = replace(default_experiment.global_parameters)
    experiments = []

#   - - - User Parametrization - - -
#   Note that ml2hls does not check for correct values for the arguments, the respective libraries are responsible for that.
#   For example ml2hls does not check whether the granularity values are the strings 'name', 'type', 'model'
#   hls4ml will warn if the user entered wrong values.
    while True:
        # Append an experiment with default settings to the list of experiments
        print("Select option:\n\
            hls4ml_config: create a set of values to be sweeped from the hls4ml config arguments.\n\
            hls4ml_converter: create a set of values to be sweeped from the hls4ml converter arguments.\n\
            global_config: create a set of global configuration arguments.\n\
            sweep: generate all combinations of the marked parameters.\n\
            print_marked: print the currently marked parameters.\n\
            print_experiments: print the currently generated experiments.\n\
            run: execute the generated experiments.\n\
            quit: exit the program.")
        user_input = input(">>> ").strip()
        if not user_input:
            continue
        elif user_input in ["hls4ml_config", "hc"]:
            parametrize_hls4ml_config(marked_parameters)
            print(marked_parameters)
        elif user_input in ["hls4ml_converter", "hcv"]:
            parametrize_hls4ml_converter(marked_parameters)
        elif user_input in ["global_config", "gc"]:
            global_parameters = global_config()
            print(global_parameters)
        elif user_input in ["sweep", "s"]:
            experiments = sweep(marked_parameters, global_parameters)
        elif user_input in ["print_marked", "pm"]:
            pprint.pprint(marked_parameters, sort_dicts=False)
        elif user_input in ["print_experiments", "pe"]:
            pprint.pprint(experiments, sort_dicts=False)
        elif user_input in ["run", "execute", "e"]:
            run_experiments(experiments)
        elif user_input in ["quit", "exit", "q", "e"]:
            break
        else:
            print(f"Option {user_input} does not exist.")
            continue

if __name__ == "__main__":
    main()

