import streamlit as st
from dataclasses import asdict
from itertools import product
import time
import hls4ml

# - - - Dataclasses and Custom Methods - - -
from configs import run_experiments, ExperimentConfig, HLS4MLConfig, HLS4MLConverter, GlobalParameters
from sweep import sweep, set_nested_attr, default_experiment

# - - - Global variables - - -
# Initialize once
if "experiments" not in st.session_state:
    st.session_state.experiments = []
if "marked_parameters" not in st.session_state:
    st.session_state.marked_parameters = {}
if "sweep_done" not in st.session_state:
    st.session_state.sweep_done = False
if "read_done" not in st.session_state:
    st.session_state.run_done = False

# - - - Streamlit GUI - - -
# TODO: add: icon with page_icon = icon
st.set_page_config(page_title='ml2hls', layout = 'wide', initial_sidebar_state = 'auto') 
# Remove deploy & settings buttons
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

# - - - Global Parameter - - -
st.sidebar.header("🌍Global Parameters")

global_project_name = st.sidebar.text_input("Project Name", "ml2hls_project")
global_output_dir = st.sidebar.text_input("Output Directory", "ml2hls_project")
global_ml2hls_project_dir = st.sidebar.text_input("Project Output Directory", "ml2hls_project")
global_input_data_tb = st.sidebar.text_input("Output for Testbench", None)
global_output_data_tb= st.sidebar.text_input("Input for Testbench", None)

st.markdown(
    f"""
    <h1 style="
        font-size: 60px;
        background: linear-gradient(to right, 
            #425df5 0%, 
            #6f42f5 10%, 
            #8d42f5 20%, 
            #a742f5 50%, 
            #c542f5 100%);
        -webkit-background-clip: text;
        color: transparent;
    ">
        {global_project_name}
    </h1>
    """,
    unsafe_allow_html=True
)

# - - - HLS4ML Config Parameters - - -
st.sidebar.header("⚙️HLS4ML Config Parameters")

hls4ml_config_granularity = st.sidebar.multiselect(
    "Granularity",
    options=["model", "name", "type"],
    default=["model"]
)
hls4ml_config_backend = st.sidebar.multiselect(
    "Config Backend",
    options=["Vitis", "Vivado"],
    default=None
)
hls4ml_config_default_precision = st.sidebar.multiselect(
    "Precision",
    options=["fixed<8,4>", "fixed<8,6>", "fixed<16,4>", "fixed<16,4>", "fixed<16,4>", "fixed<16,6>", "fixed<18,4", "fixed<18,8>"],
    default=["fixed<16,6>"],    
    accept_new_options=True
)
hls4ml_config_default_reuse_factor = st.sidebar.multiselect(
    "Reuse Factor",
    options=[1, 2, 4, 8, 16, 32, 64, 128],
    default=[1]
)
hls4ml_config_max_precision = st.sidebar.multiselect(
    "Max Precision",
    options=["fixed<8,4>", "fixed<8,6>", "fixed<16,4>", "fixed<16,4>", "fixed<16,4>", "fixed<16,6>", "fixed<18,4", "fixed<18,8>"],
    default=None,    
    accept_new_options=True
)

# - - - HLS4ML Converter Parameters - - -
st.sidebar.header("🛠️HLS4ML Converter Parameters")
hls4ml_converter_backend = st.sidebar.multiselect(
    "Converter Backend",
    options=["Vitis", "Vivado"],
    default=["Vitis"],
    accept_new_options=True
)
hls4ml_converter_part = st.sidebar.multiselect(
    "Part",
    options=["xc7z020clg400-1"], # TODO add more parts
    default=["xc7z020clg400-1"],
    accept_new_options=True
)
hls4ml_converter_clock_period = st.sidebar.multiselect(
    "Clock Period (ns)",
    options=[5],
    default=5,
    accept_new_options=True
)
hls4ml_converter_clock_uncertainty = st.sidebar.multiselect(
    "Clock Uncertainty (%)",
    options=[12.5, 27.5],
    default=None,
    accept_new_options=True
)
hls4ml_converter_io_type = st.sidebar.multiselect(
    "I/O Type",
    options=["io_parallel", "io_stream"],
    default=["io_parallel"]
)

# Sweep button
sweep_btn = st.sidebar.button("Sweep")
if sweep_btn:
    st.session_state.marked_parameters = {}
    # HLS4ML Config parameters
    st.session_state.marked_parameters["hls4ml_config.granularity"] = hls4ml_config_granularity
    st.session_state.marked_parameters["hls4ml_config.backend"] = hls4ml_config_backend
    st.session_state.marked_parameters["hls4ml_config.default_precision"] = hls4ml_config_default_precision
    st.session_state.marked_parameters["hls4ml_config.default_reuse_factor"] = hls4ml_config_default_reuse_factor
    st.session_state.marked_parameters["hls4ml_config.max_precision"] = hls4ml_config_max_precision

    # HLS4ML Converter parameters
    st.session_state.marked_parameters["hls4ml_converter.backend"] = hls4ml_converter_backend
    st.session_state.marked_parameters["hls4ml_converter.part"] = hls4ml_converter_part
    st.session_state.marked_parameters["hls4ml_converter.clock_period"] = hls4ml_converter_clock_period
    st.session_state.marked_parameters["hls4ml_converter.clock_uncertainty"] = hls4ml_converter_clock_uncertainty
    st.session_state.marked_parameters["hls4ml_converter.io_type"] = hls4ml_converter_io_type

    # keep only keys with >= 2 items
    st.session_state.marked_parameters = {
        k: v for k, v in st.session_state.marked_parameters.items() if len(v) > 0
    }

    # Global parameters
    global_params = GlobalParameters(
        project_name=global_project_name,
        output_dir=global_output_dir,
        ml2hls_project_dir=global_ml2hls_project_dir,
        input_data_tb=global_input_data_tb,
        output_data_tb=global_output_data_tb
    )

    st.write("### Running Sweep...")
    st.session_state.experiments = sweep(st.session_state.marked_parameters, global_params)

    st.success(f"Generated {len(st.session_state.experiments)} experiments.")

    # Show experiment configs
    for i, exp in enumerate(st.session_state.experiments, 0):
        st.write(f"#### Experiment {i+1}")
        with st.expander(f"Experiment {i+1}", expanded=False):
            st.json(asdict(exp))

    clear_btn = st.button("Clear")
    if clear_btn:
        # Clear everything in session_state
        for key in st.session_state.keys():
            del st.session_state[key]
        st.session_state.experiments = {}
        st.session_state.marked_parameters = {}
        st.rerun()
        
    print(st.session_state.experiments)
    st.session_state.sweep_done = True


# Compile button
compile_btn = st.sidebar.button("Compile", disabled=not st.session_state.sweep_done)
if (compile_btn):
    st.session_state.run_done = True
    run_experiments(st.session_state.experiments, build=False, compile=True)
    # hls4ml.report.read_vivado_report(hls4ml_dir)
    
# Build button
build_btn = st.sidebar.button("Build", disabled=not st.session_state.sweep_done)
if (build_btn):
    st.session_state.run_done = True
    run_experiments(st.session_state.experiments, build=True, compile=False)

report_btn = st.sidebar.button("Read Report", disabled=not st.session_state.run_done)
if (report_btn):
    for exp in st.session_state.experiments:
        #hls4ml_dir = f"{exp.global_parameters.ml2hls_project_dir}/{exp.global_parameters.output_dir}/hls4ml_prj"
        hls4ml_dir = "/home/vag/ml2hls/ml2hls_project/ml2hls_project_1"
        hls4ml.report.read_vivado_report(hls4ml_dir)

# TODO global parameters don't work