from dataclasses import dataclass


@dataclass
class OneTransConfig:
    """
    Configuration class to build a OneTrans PyTorch module.

    :param n_layers: Number of TransformerEncoderLayer
    :param n_heads: Number of heads in the TransformerEncoderLayer
    """

    n_layers: int = 2
    n_heads: int = 2


@dataclass
class DatasetConfig:
    dataset_type: str = 'flat'
    dataset_size: str = '50m'
    interaction_name: str = 'multi_event'
    default_like_window_seconds: int = 24 * 60 * 60
    lag_seconds: int = 15 * 60

dataset_config = DatasetConfig()
one_trans_config = OneTransConfig()