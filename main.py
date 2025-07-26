import os
import importlib.util
import tensorflow as tf
import numpy as np
import hls4ml
from configs import ExperimentConfig, HLS4MLConfig
from dataclasses import asdict, replace
import pprint

# - - - Global Variables - - -
# Default experiment values:
default_experiment = ExperimentConfig(
    hls4ml_config=HLS4MLConfig(
        granularity='model',
        backend=None,
        default_precision='fixed<16,6>',
        default_reuse_factor=1,
        max_precision=None
    )
)
# Experiments
# One entry for every value entered by user. 
# eg. if user enters: hls4ml_config granularity model name and then hls4ml_config default_reuse_factor 1 2 4 then 2+3=5 copies of experiments is created.
experiments = []


def parametrize_hls4ml_config():
    global experiments, default_experiment        

    #   - - - User Parametrization - - -
    while True:
        current_experiment = default_experiment
        print("Default hls4ml config values:")
        pprint.pprint(default_experiment)

    #   Receive user input and remove the ">>> " string from the start, split user input into strings.
    #   First string should be the argument selected and the rest of the strings should be the set of the values that the argument will take.
        user_input = input(">>> ").strip()
        user_input = user_input.split()
        argument = user_input[0]
        values = user_input[1:]
    #   Eliminate duplicates
        values = list(dict.fromkeys(values))

    #   Make sure user_input, argument and values are not empty (values can be empty for the back/ return option)
        if not user_input or not argument or (not values and argument not in ["back", "return", "b", "r"]):
            print(f"Syntax: [hls4ml_config argument]: value1 value2 value3...\neg. default_reuse_factor: 1, 2, 4, 8")
            continue

    #   Make sure that the arguments will receive proper values
        elif argument == "granularity":
            for value in values:
                # Since dataclasses are frozen, we have to use the replace function to change a value:
                replace(current_experiment, hls4ml_config=replace(current_experiment.hls4ml_config, granularity=value))
                experiments.append(current_experiment)
        elif argument == "default_precision":
            for value in values:
                replace(current_experiment, hls4ml_config=replace(current_experiment.hls4ml_config, default_precision=value))
                experiments.append(current_experiment)
                
        # elif args[0] == "backend":
        # elif args[0] == "default_precision":
        # elif args[0] == "default_reuse_factor":
        # elif args[0] == "max_precision":

        elif argument in ["back", "return", "b", "r"]:
            break
        else:
            print(f"Syntax: [hls4ml_config argument]: value1, value2, value3...\neg. default_reuse_factor: 1, 2, 4, 8")
            continue
        print(f"{argument} updated.")


    return hls4ml
    

##      Main        ##
def main():

#   - - - User Parametrization - - -
#   Note that ml2hls does not check for correct values for the arguments, the respective libraries are responsible for that.
#   For example ml2hls does not check whether the granularity values are the strings 'name', 'type', 'model'
#   hls4ml will warn if the user entered wrong values.
    while True:
        # Append an experiment with default settings to the list of experiments
        print("Select option:\n\
                hls4ml_config: create a set of values to be sweeped from the hls4ml config arguments.")
        user_input = input(">>> ").strip()
        if not user_input:
            continue
        elif user_input in ["hls4ml_config", "hc"]:
            parametrize_hls4ml_config()
        elif user_input in ["quit", "exit", "q", "e"]:
            break
        elif user_input in ["print", "p"]:
            pprint.pprint(experiments)
        else:
            print(f"Option {user_input} does not exist.")
            continue
#       - - - Sweep range - - -

    # config = hls4ml.utils.config_from_keras_model(model, **hls4ml_config)

if __name__ == "__main__":
    main()