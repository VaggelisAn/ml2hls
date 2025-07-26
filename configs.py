from dataclasses import dataclass
from typing import Optional

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
class DatasetConfig:
    path: str                   
    dimensions: int             # 1

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
class ExperimentConfig:
    # dataset: DatasetConfig
    hls4ml_config: HLS4MLConfig
    # pruning: Optional[PruningConfig]
    # training: TrainingConfig

