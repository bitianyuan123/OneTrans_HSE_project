import numpy as np
import torch


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class ToNumpy(Transform):
    """ Convert all lists to numpy """

    def __init__(self, dtype=np.int64):
        super().__init__()
        self._dtype = dtype

    def __call__(self, sample):
        res = {}
        for key, value in sample.items():
            if isinstance(value, dict):
                res[key] = self.__call__(value)
            elif isinstance(value, list):
                res[key] = np.array(value, dtype=self._dtype)
            else:
                res[key] = value
        return res


class ToTorch(Transform):
    """Convert all lists or numpy arrays in torch tensors."""

    def __call__(self, obj):
        if isinstance(obj, dict):
            return {key: self.__call__(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return torch.tensor(obj)
        elif isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        else:
            return obj


class ToDevice(Transform):
    """Move obj to device."""

    def __init__(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ):
        self._device = device
        self._non_blocking = non_blocking

    def __call__(self, obj):
        def _to_device_recursive(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(self._device, non_blocking=self._non_blocking)
            if isinstance(obj, dict):
                return {k: _to_device_recursive(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_to_device_recursive(x) for x in obj)
            return obj
        return _to_device_recursive(obj)