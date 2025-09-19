from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import pprint
from tensorflow.keras.models import load_model
import hls4ml
from dataclasses import replace

from helpers import parse_input, set_nested_attr

# -Dataset -> DatasetConfig -> TODO
# -Model Architecture -> TODO
# -Model Training -> TODO
# -Pruning -> PruningConfig -> TODO
# -HLS4ML Config -> HLS4MLConfig 
# -HLS4ML Converter Config -> HLS4MLConverterConfig -> TODO

# @dataclass(frozen=True)
# class TrainingConfig:
#     epochs: int
#     batch_size: int
#     optimizer: str

@dataclass(frozen=True)
class GlobalParameters:
    output_dir: Optional[str] = "project"
    project_name : Optional[str] = "project"
    ml2hls_project_dir: Optional[str] = "ml2hls_project"
    input_data_tb: Optional[str] = None
    output_data_tb: Optional[str] = None

@dataclass(frozen=True)
class ModelConfig:
    path: str = 'model.h5' # Path to the model                  

@dataclass(frozen=True)
class DatasetConfig:
    path: str               # Path to the dataset                  
    shape: Tuple[int, ...]  # Input dimensions

# @dataclass(frozen=True)
# class PruningConfig:
#     enabled: bool               # False
#     amount: float               # 0

@dataclass(frozen=True)
class HLS4MLConfig:           
    granularity: str = 'model'              
    backend: Optional[str] = None          
    default_precision: str = 'fixed<16,6>'       
    default_reuse_factor: int = 1
    max_precision: Optional[str] = None

@dataclass(frozen=True)
class HLS4MLConverter:   
    backend: Optional[str] = 'Vitis'
    # board: Optional[str] = 'xc7z020clg400-1'
    part: Optional[str] = 'xc7z020clg400-1'
    clock_period: Optional[int] = 5 # Not recommended to change this, can do directly from Vitis after build
    clock_uncertainty: Optional[str] = None
    io_type: Optional[str] = 'io_parallel'

@dataclass(frozen=True)
class ExperimentConfig:
    # dataset: DatasetConfig
    hls4ml_config: HLS4MLConfig
    hls4ml_converter: HLS4MLConverter
    # pruning: Optional[PruningConfig]
    # training: TrainingConfig
    model_config: ModelConfig
    global_parameters: GlobalParameters

# Default experiment values:
default_experiment = ExperimentConfig(
    hls4ml_config=HLS4MLConfig(
        granularity='model',
        backend=None,
        default_precision='fixed<16,6>',
        default_reuse_factor=1,
        max_precision=None
    ),
    hls4ml_converter=HLS4MLConverter(
        backend='Vitis',
        part='xc7z020clg400-1',
        clock_period=5,
        clock_uncertainty=None,
        io_type='io_parallel'
    ),
    model_config=ModelConfig(
        path='model.h5'
    ),
    global_parameters=GlobalParameters(
        output_dir='project',
        project_name='project',
        ml2hls_project_dir='ml2hls_project',
        input_data_tb=None,
        output_data_tb=None
    )
)

# TODO: Add this to the experiment class
def remove_experiments():
    return


def global_config():
    global_parameters_arguments = {"output_dir", "ml2hls_project_dir", "project_name ", "input_data_tb", "output_data_tb"}
    global_parameters = replace(default_experiment.global_parameters)

    print("Default global parameters:")
    pprint.pprint(default_experiment.global_parameters, sort_dicts=False)

    #   - - - User Parametrization - - -
    while True:
        argument, value = parse_input()
    #   Proper arguments are set as global_parameters and their values are returned. No duplicates.
        if argument in global_parameters_arguments:
            global_parameters = replace(global_parameters, **{argument: value[0]})
        elif argument in ["back", "return", "b", "r"]:
            break
        else:
            print(f"Syntax: [global_parameters argument]: value\neg. project_name myprj")
            continue
    return global_parameters

# Set the parameters of the hls4ml converter to be sweeped
def parametrize_hls4ml_converter(marked_parameters):
    hls4ml_converter_arguments = ['backend', 'part', 'clock_period', 'clock_uncertainty', 'io_type']

    print("Default hls4ml config values:")
    pprint.pprint(default_experiment.hls4ml_converter, sort_dicts=False)
    
    #   - - - User Parametrization - - -
    while True:
        argument, values = parse_input()
    #   Proper arguments get marked as marked_parameters and their values are stored. No duplicates.
        if argument in hls4ml_converter_arguments:
            key = f"hls4ml_converter.{argument}"
            if key not in marked_parameters:
                marked_parameters[key] = []
            for value in values:
                if value not in marked_parameters[key]:
                    marked_parameters[key].append(value)
        elif argument in ["back", "return", "b", "r"]:
            break
        else:
            print(f"Syntax: [hls4ml_converter argument]: value1 value2 value3 ...\neg. io_type io_parallel io_stream")
            continue
    return 


# TODO: Add this to the HLS4MLConfig class if needed:
# Set the parameters of the hls4ml config to be sweeped
def parametrize_hls4ml_config(marked_parameters):
    hls4ml_config_arguments = ["granularity", "backend", "default_precision", "default_reuse_factor", "max_precision"]

    print("Default hls4ml config values:")
    pprint.pprint(default_experiment.hls4ml_config, sort_dicts=False)
    
    #   - - - User Parametrization - - -
    while True:
        argument, values = parse_input()
    #   Proper arguments get marked as marked_parameters and their values are stored. No duplicates.
        if argument in hls4ml_config_arguments:
            key = f"hls4ml_config.{argument}"
            if key not in marked_parameters:
                marked_parameters[key] = []
            for value in values:
                if value not in marked_parameters[key]:
                    marked_parameters[key].append(value)
        elif argument in ["back", "return", "b", "r"]:
            break
        else:
            print(f"Syntax: [hls4ml_config argument]: value1 value2 value3 ...\neg. default_reuse_factor 1 2 4 8")
            continue
    return 
    
def run_experiments(experiments, build=False, compile=False):
    experiment_index = 0
    # TODO: change to support model configs
    for experiment in experiments:
        print("\n\n- - - Running Experiment {}/{}: {} - - -\n".format(experiment_index+1, len(experiments), experiment.global_parameters.project_name))
    #   - - - Model - - -
        try:
            model = load_model(experiment.model_config.path)
        except Exception as e:
            print(f"Error in project '{global_parameters.project_name}':")
            print(f"Error during model loading: {e}")
            continue

    #   - - - HLS4ML - - - 
        try:
            config_kwargs = asdict(experiment.hls4ml_config)
            converter_kwargs = asdict(experiment.hls4ml_converter)
            converter_kwargs['project_name'] = experiment.global_parameters.project_name
            converter_kwargs['output_dir'] = experiment.global_parameters.output_dir

            hls_config = hls4ml.utils.config_from_keras_model(model=model, **config_kwargs)
            hls_model = hls4ml.converters.convert_from_keras_model(model=model, hls_config=hls_config, **converter_kwargs)
            
            if (compile):
                print("Compiling...")
                hls_model.compile()   
            if (build):
                print("Building...")
                hls_model.build()
    
        except Exception as e:
            print(f"Error in project '{experiment.global_parameters.project_name}':")
            print(f"Error during HLS4ML conversion or build: {e}")
            continue
        
        experiment_index += 1