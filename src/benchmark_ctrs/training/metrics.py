from __future__ import annotations

from typing import TYPE_CHECKING, Final

from typing_extensions import Protocol, final

if TYPE_CHECKING:
    from collections.abc import Mapping


class SupportsScalars(Protocol):
    def get_scalars(self) -> Mapping[str, float]: ...


@final
class ScalarTags:
    @final
    class Time:
        Prefix: Final = "time/"
        Epoch: Final = Prefix + "epoch"
        Data: Final = Prefix + "data"
        Batch: Final = Prefix + "batch"

    @final
    class Loss:
        Prefix: Final = "loss/"
        Total: Final = Prefix + "total"
        Classification: Final = Prefix + "classification"
        Robust: Final = Prefix + "robust"

    @final
    class Accuracy:
        Prefix: Final = "accuracy/"
        Top1: Final = Prefix + "top1"
        Top5: Final = Prefix + "top5"

    @final
    class Gradients:
        Prefix: Final = "gradients"

        @classmethod
        def for_layer(cls, name: str):
            return f"{cls.Prefix}/{name}"


class Metrics:
    def __init__(
        self,
        *,
        data_time: float | None = None,
        batch_time: float | None = None,
        loss: float | None = None,
        top1_acc: float | None = None,
        top5_acc: float | None = None,
        layer_gradients: Mapping[str, float] | None = None,
        extra: Mapping[str, float] | None = None,
    ):
        self.data_time = data_time
        self.batch_time = batch_time
        self.loss = loss
        self.top1_acc = top1_acc
        self.top5_acc = top5_acc
        self.layer_gradients = layer_gradients
        self.extra = extra

    @property
    def scalars(self) -> Mapping[str, float]:
        """called to get the scalars which get logged for each epoch

        Returns:
            Mapping[str, float]: the (tag, value) pairs for the scalars
        """
        scalars = {}
        if self.data_time is not None:
            scalars[ScalarTags.Time.Data] = self.data_time
        if self.batch_time is not None:
            scalars[ScalarTags.Time.Batch] = self.batch_time
        if self.loss is not None:
            scalars[ScalarTags.Loss.Total] = self.loss
            scalars[ScalarTags.Loss.Classification] = self.loss
        if self.top1_acc is not None:
            scalars[ScalarTags.Accuracy.Top1] = self.top1_acc
        if self.top5_acc is not None:
            scalars[ScalarTags.Accuracy.Top5] = self.top5_acc
        if self.layer_gradients is not None:
            for layer, grad in self.layer_gradients.items():
                scalars[ScalarTags.Gradients.for_layer(layer)] = grad
        if self.extra is not None:
            scalars.update(self.extra)
        return scalars
