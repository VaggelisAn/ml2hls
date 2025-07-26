from pathlib import Path
from dataclasses import asdict
import json, time
import tensorflow as tf
from tensorflow.keras.models import load_model
import hls4ml

class ExperimentRunner:
    def __init__(self, output_dir="experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run(self, config):
        run_dir = self.output_dir
        run_dir.mkdir(exist_ok=True)

    #   - - - Model - - -
        model = self._load_keras2_model(config)

    #   - - - HLS4ML - - - 
        hls_config = hls4ml.utils.config_from_keras_model(model, **asdict(config.hls4ml_config))
        hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=hls_config, output_dir=run_dir)
        # hls_model.build()

    def _load_keras2_model(self, config):
        model = load_model('model.h5', compile=True)
        return model