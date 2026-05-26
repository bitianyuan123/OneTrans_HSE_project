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
