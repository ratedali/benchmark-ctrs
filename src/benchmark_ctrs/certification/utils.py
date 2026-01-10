from collections.abc import Iterable
from heapq import heapify, heappop, heappush
from typing import NamedTuple, NewType, Union

import torch
from torch import Tensor

from benchmark_ctrs.types import Batch

InputId = NewType("InputId", int)
BatchIndex = NewType("BatchIndex", int)


class _QueueItem(NamedTuple):
    cnt: int
    id: InputId


class SamplingQueue:
    def __init__(self, batch_size: int, data: Union[Tensor, Batch]) -> None:
        self.batch_size: int = batch_size
        self.data = data

        if isinstance(data, tuple):
            inputs, targets = data
        else:
            inputs = data
            targets = None

        self.total: int = inputs.size(0)
        self.input_ids: list[InputId] = [InputId(i) for i in range(self.total)]

        self.batch_indices: dict[InputId, list[BatchIndex]] = {
            i: [] for i in self.input_ids
        }
        self.batch_ids: dict[BatchIndex, InputId] = {}

        self.heap: list[_QueueItem] = [_QueueItem(0, i) for i in self.input_ids]
        heapify(self.heap)

        X = []
        y = []

        for i in range(self.batch_size):
            batch_idx = BatchIndex(i)
            item = heappop(self.heap)

            self.batch_indices[item.id].append(batch_idx)
            self.batch_ids[batch_idx] = item.id

            X.append(inputs[batch_idx, ...])
            if targets is not None:
                y.append(targets[batch_idx].item())

            heappush(self.heap, _QueueItem(item.cnt + 1, item.id))
        self.X = torch.stack(X)
        self.y = None
        if targets is not None:
            self.y = torch.tensor(y, dtype=targets.dtype, device=targets.device)

        self.new: dict[BatchIndex, bool] = {
            BatchIndex(i): False for i in range(self.batch_size)
        }

    @property
    def batch(self) -> Union[Tensor, Batch]:
        if self.y is not None:
            return self.X, self.y
        return self.X

    def replace_input(self, input_id: InputId, exclude: Iterable[InputId]) -> None:
        for j in self.batch_indices[input_id]:
            while True:
                cnt, idx = heappop(self.heap)
                if idx not in exclude:
                    break

            self.batch_indices[idx].append(j)
            self.batch_ids[j] = idx

            inputs, targets = self.data[:2]
            X, y = self.batch
            if cnt == 0:
                X[j], y[j] = inputs[idx, ...], targets[idx]
            else:
                k = self.batch_indices[idx][0]
                X[j], y[j] = X[k], y[k]
            heappush(self.heap, _QueueItem(cnt + 1, idx))
