from typing_extensions import override

from benchmark_ctrs.training.methods.abc import (
    BatchResults,
    TrainingMethod,
    TrainingParameters,
)


class NoopTraining(TrainingMethod):
    @staticmethod
    def create_instance(params: TrainingParameters):
        return NoopTraining()

    @property
    @override
    def instance_tag(self):
        return ("noop",)

    def train(self, ctx, batch):
        predictions = ctx.model_wrapper.model(batch.inputs)

        ctx.optimizer.zero_grad()
        ctx.optimizer.step()

        return BatchResults(
            predictions=predictions,
            loss=ctx.criterion(predictions, batch.targets),
        )
