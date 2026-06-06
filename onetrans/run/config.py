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


DENSE_COLUMNS: tuple[str, ...] = (
    "user_lag_listen_cnt",
    "user_lag_like_cnt",
    "user_lag_full_play_cnt",
    "user_lag_skip_cnt",
    "item_lag_listen_cnt",
    "item_lag_like_cnt",
    "item_lag_full_play_cnt",
    "item_lag_skip_cnt",
    "ui_lag_listen_cnt",
    "ui_lag_like_cnt",
    "ui_lag_full_play_cnt",
    "ui_lag_skip_cnt",
    "user_lag_avg_played_ratio",
    "item_lag_avg_played_ratio",
    "ui_lag_avg_played_ratio",
)
MULTIVALENT_COLUMNS: tuple[str, ...] = ("artist_ids", "album_ids")
SPARSE_COLUMNS = ("uid", "item_id")
LABEL_COLUMNS = ("is_like", "is_full_play")
CORE_MIN_INTERACTIONS_PER_ITEM = 5
TEST_INTERVAL_SECONDS = 7 * 24 * 60 * 60

dataset_config = DatasetConfig()
one_trans_config = OneTransConfig()
