"""权重版本化加载（P0-2）：checkpoint 优先，seed 重建兜底。

Serving 启动时按 ``model_version`` 装载 backbone 权重：

1. **checkpoint 优先**：从 ``checkpoint_dir/<model_version>.pt`` 加载已发布版本。
2. **seed 兜底**：checkpoint 缺失 / 损坏 / 结构不匹配时，回退到确定性
   ``torch.manual_seed(seed)`` 初始化。这是「最差路径」，保证服务总能拉起可复现的
   基线权重（seed 应对齐离线训练时的初始化种子）。

checkpoint 仅存 ``state_dict`` + 轻量 ``meta``，不 pickle 任意对象；用
``weights_only=False`` 以保证跨 torch 版本的 state_dict 兼容读取。

对应设计文档 §5 的「版本化 + 重建可靠」与 ``docs/implementation_status.md`` 的 P0-2。
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from onetrans.models.one_trans import OneTrans

logger = logging.getLogger(__name__)


def save_checkpoint(
    backbone: OneTrans,
    path: str | os.PathLike[str],
    model_version: str | None = None,
    seed: int | None = None,
) -> str:
    """保存 backbone ``state_dict`` 到 ``path``（返回实际路径）。"""
    checkpoint: dict[str, Any] = {
        "state_dict": backbone.state_dict(),
        "meta": {
            "model_version": model_version,
            "seed": seed,
        },
    }
    torch.save(checkpoint, path)
    return str(path)


def _checkpoint_path(checkpoint_dir: str | os.PathLike[str], model_version: str) -> str:
    return os.path.join(checkpoint_dir, f"{model_version}.pt")


def load_backbone(
    model_version: str,
    *,
    checkpoint_dir: str | os.PathLike[str] | None = None,
    seed: int = 0,
    **build_kwargs: Any,
) -> tuple[OneTrans, str]:
    """构建并装载 backbone 权重。

    :param model_version: 目标模型版本（决定 checkpoint 文件名）。
    :param checkpoint_dir: checkpoint 目录；``None`` 表示跳过（纯 seed）。
    :param seed: seed 兜底初始化种子（checkpoint 缺失/损坏时使用）。
    :param build_kwargs: 透传给 :class:`OneTrans` 的结构参数（d_model/num_blocks/...）。
    :return: ``(backbone, source)``，``source`` 取 ``"checkpoint"`` 或 ``"seed"``。
    """
    # 1) 先按 seed 构建基线（同时是兜底路径）
    torch.manual_seed(seed)
    backbone = OneTrans(**build_kwargs)
    source = "seed"

    # 2) 尝试 checkpoint 加载
    if checkpoint_dir is not None:
        path = _checkpoint_path(checkpoint_dir, model_version)
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                state = ckpt["state_dict"] if isinstance(ckpt, dict) else ckpt
                backbone.load_state_dict(state)
                source = "checkpoint"
            except Exception as exc:  # noqa: BLE001 - 损坏/结构不匹配一律 seed 兜底
                # 损坏 / 结构不匹配 → 保留 seed 重建（最差路径）
                logger.warning(
                    "checkpoint 加载失败，回退 seed=%d 重建: %s", seed, exc
                )

    return backbone, source