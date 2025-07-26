from itertools import product
from runners.base_runner import ExperimentRunner
from configs import ExperimentConfig, HLS4MLConfig

granularities = ["name", "model"]
reuse_factors = [1, 2]
default_precision = ["fixed<16,6>", "fixed<8,2>"]

def sweep():
    runner = ExperimentRunner()

    for r, p in product(reuse_factors, default_precision):
        cfg = ExperimentConfig(
            hls4ml_config=HLS4MLConfig(
                    granularity='model',
                    backend='Vitis',
                    default_precision = p,
                    default_reuse_factor = r,
                    max_precision='None'
            ),
        )
        runner.run(cfg)
