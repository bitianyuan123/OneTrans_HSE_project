import torch
from torch.utils.data import DataLoader as TorchDataLoader

from onetrans.data.transforms import ToDevice


class DataLoader(TorchDataLoader):
    def __init__(self, *args, transforms=None, **kwargs):
        self._transforms = transforms if transforms is not None else []
        if not isinstance(self._transforms, (list, tuple)):
            self._transforms = [self._transforms]
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for batch in super().__iter__():
            for t in self._transforms:
                batch = t(batch)
            yield batch


class PrefetchDataLoader(TorchDataLoader):
    def __init__(self, *args, device, transforms=None, buffer_size=1, **kwargs):
        assert buffer_size > 0, "buffer_size must be greater than 0"
        self._transforms = transforms if transforms is not None else []
        if not isinstance(self._transforms, (list, tuple)):
            self._transforms = [self._transforms]
        if any(isinstance(t, ToDevice) for t in self._transforms):
            raise ValueError("transforms must not contain ToDevice; PrefetchDataloader adds it at the end")
        self._transforms.append(ToDevice(device, non_blocking=True))
        self._device = device
        self._buffer_size = buffer_size
        super().__init__(*args, **kwargs)
        self._stream = torch.cuda.Stream(self._device)
        self._buffer = [None] * self._buffer_size
        self._events = [torch.cuda.Event() for _ in range(self._buffer_size)]

    def __iter__(self):
        base_iter = super().__iter__()
        current_stream = torch.cuda.current_stream(self._device)

        filled = 0
        for i in range(self._buffer_size):
            try:
                batch = next(base_iter)
            except StopIteration:
                break
            with torch.cuda.stream(self._stream):
                for t in self._transforms:
                    batch = t(batch)
                self._buffer[i] = batch
                self._events[i].record(self._stream)
            filled += 1

        idx = 0
        while filled > 0:
            current_stream.wait_event(self._events[idx])
            batch = self._buffer[idx]

            try:
                cpu_batch = next(base_iter)
                with torch.cuda.stream(self._stream):
                    for t in self._transforms:
                        cpu_batch = t(cpu_batch)
                    self._buffer[idx] = cpu_batch
                    self._events[idx].record(self._stream)
            except StopIteration:
                self._buffer[idx] = None
                filled -= 1

            idx = (idx + 1) % self._buffer_size
            yield batch