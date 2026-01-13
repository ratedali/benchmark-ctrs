import dataclasses
from collections.abc import Iterable
from heapq import heapify, heappop, heappush
from typing import Any, NamedTuple, NewType, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Self

from benchmark_ctrs.types import Batch

InputId = NewType("InputId", int)
BatchIndex = NewType("BatchIndex", int)


class QueueItem(NamedTuple):
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

        self.heap: list[QueueItem] = [QueueItem(0, i) for i in self.input_ids]
        heapify(self.heap)

        X = []
        y = []

        for i in range(self.batch_size):
            batch_idx = BatchIndex(i)
            item = heappop(self.heap)

            self.batch_indices[item.id].append(batch_idx)
            self.batch_ids[batch_idx] = item.id

            X.append(inputs[item.id, ...])
            if targets is not None:
                y.append(targets[item.id].item())

            heappush(self.heap, QueueItem(item.cnt + 1, item.id))
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
                next_item = heappop(self.heap)
                if next_item.id not in exclude:
                    break

            self.batch_indices[next_item.id].append(j)
            self.batch_ids[j] = next_item.id

            if next_item.cnt == 0:
                self.X[j] = self.data[0][next_item.id, ...]
                if self.y is not None:
                    self.y[j] = self.data[1][next_item.id]
            else:
                k = self.batch_indices[next_item.id][0]
                self.X[j] = self.X[k]
                if self.y is not None:
                    self.y[j] = self.y[k]
            heappush(self.heap, QueueItem(next_item.cnt + 1, next_item.id))


@dataclasses.dataclass(frozen=True)
class RunningTrial:
    num_samples: int
    countA: int
    pA: float = 0.0
    done: bool = False
    stopping_val: float = 0.0
    stopping_n: int = 0

    @classmethod
    def create_initial(cls, *args: Any, **kwargs: Any) -> "Self":
        return cls(0, 0, *args, **kwargs)

    def add_sample(self, pred: int, y: int) -> "Self":
        return dataclasses.replace(
            self,
            num_samples=self.num_samples + 1,
            countA=self.countA + (pred == y),
        )

    def update_pA(self, pA: float) -> "Self":
        return dataclasses.replace(self, pA=max(pA, self.pA))

    def mark_done(self) -> "Self":
        return dataclasses.replace(self, done=True)

    def check_stopping(self, patience: int, delta: float) -> "Self":
        if self.pA - self.stopping_val > delta:
            return dataclasses.replace(
                self,
                stopping_val=self.pA,
                stopping_n=self.num_samples,
            )

        if self.num_samples - self.stopping_n > patience:
            return self.mark_done()

        return self
