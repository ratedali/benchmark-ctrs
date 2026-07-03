# benchmark-ctrs

`benchmark-ctrs` is a PyTorch Lightning framework for training and evaluating
classifiers for certified robustness with randomized smoothing. It provides:

- reusable Lightning data modules for MNIST, CIFAR-10, and ImageNet;
- baseline training modules such as standard training and Gaussian augmentation;
- randomized smoothing certification methods, including fixed-sample,
  union-bound, and betting-style sequential certification;
- a Pluggy extension interface so experiment packages can add new methods,
  losses, data modules, callbacks, and certification procedures without
  modifying this package.

The companion repository `benchmark-ctrs-experiments` contains the method
implementations and experiment configurations used for the thesis experiments.

## Installation

Use Python 3.10 or newer. A virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the package from a local checkout:

```bash
git clone https://github.com/ratedali/benchmark-ctrs.git
cd benchmark-ctrs
python -m pip install -e .
```

Install directly from GitHub:

```bash
python -m pip install "benchmark-ctrs @ git+https://github.com/ratedali/benchmark-ctrs.git"
```

For development tools:

```bash
python -m pip install -e ".[dev,results]"
```

The package depends on PyTorch, torchvision, Lightning, torchmetrics, scipy,
statsmodels, numpy, and pluggy. Install the PyTorch build that matches your CUDA
runtime if you plan to use GPUs.

## Data And Outputs

By default, datasets are stored in `./datasets_cache` and training outputs in
`./logs`. Both paths are ignored by Git.

To use a shared dataset location:

```bash
export DATASETS_CACHE=/path/to/datasets
```

On PowerShell:

```powershell
$env:DATASETS_CACHE = "C:\path\to\datasets"
```

CIFAR-10 and MNIST are downloaded through torchvision. ImageNet must already be
available in torchvision's expected ImageNet folder layout under
`DATASETS_CACHE`.

## CLI Usage

Installing the package exposes the `benchmark-ctrs` command. It is a Lightning
CLI with the usual subcommands, including `fit`, `validate`, `test`, and
`predict`.

Print the full configurable schema:

```bash
benchmark-ctrs fit --print_config
```

Training is usually driven by YAML files. A minimal CIFAR-10 Gaussian
augmentation configuration, saved as `config/cifar10-gaussian.yaml`, looks like
this:

```yaml
seed_everything: 42
trainer:
  accelerator: auto
  devices: auto
  max_epochs: 150
  deterministic: warn
  default_root_dir: ./logs
data:
  class_path: benchmark_ctrs.datasets.cifar10.CIFAR10
  init_args:
    batch_size: 256
    validation: 1000
model:
  class_path: benchmark_ctrs.modules.gaussian_aug.GaussianAug
  init_args:
    sigma: 0.25
    certification_params:
      sigma: 0.25
```

Run training:

```bash
benchmark-ctrs fit --config config/cifar10-gaussian.yaml
```

Run prediction or certification from a checkpoint:

```bash
benchmark-ctrs predict \
  --config config/cifar10-gaussian.yaml \
  --ckpt_path logs/lightning_logs/version_0/checkpoints/best.ckpt
```

To write certified radii to CSV, add
`benchmark_ctrs.callbacks.certified_radius_writer.CertifiedRadiusWriter` as a
callback and configure `model.init_args.certification`.

## Built-In Components

Data modules:

- `benchmark_ctrs.datasets.mnist.MNIST`
- `benchmark_ctrs.datasets.cifar10.CIFAR10`
- `benchmark_ctrs.datasets.imagenet.ImageNet`

Training modules:

- `benchmark_ctrs.modules.standard.MNISTStandard`
- `benchmark_ctrs.modules.standard.CIFARStandard`
- `benchmark_ctrs.modules.standard.ImageNetStandard`
- `benchmark_ctrs.modules.gaussian_aug.GaussianAug`

Certification methods:

- `benchmark_ctrs.certification.rs_certification.RSCertification`
- `benchmark_ctrs.certification.sequence.union_bound.UBCertification`
- `benchmark_ctrs.certification.sequence.betting.BettingCertification`

Callbacks:

- `benchmark_ctrs.callbacks.certified_radius_writer.CertifiedRadiusWriter`

## Reproducing Thesis Experiments

This repository provides the core benchmark framework. The thesis methods and
the full experiment grid live in `benchmark-ctrs-experiments`.

Use both repositories together:

```bash
git clone https://github.com/ratedali/benchmark-ctrs.git
git clone https://github.com/ratedali/benchmark-ctrs-experiments.git
cd benchmark-ctrs
python -m pip install -e .
cd ../benchmark-ctrs-experiments
python -m pip install -e .
```

Then run the commands from the experiment repository's README. Installing the
experiment package registers its models and losses through the `benchmark_ctrs`
entry point, so the same `benchmark-ctrs` CLI can instantiate the thesis
methods from YAML configs.

## Extending The Framework

Extensions are regular Python packages that expose Pluggy hooks through the
`benchmark_ctrs` entry-point group.

In your extension package's `pyproject.toml`:

```toml
[project.entry-points."benchmark_ctrs"]
my_methods = "my_package.register_modules"
```

In `my_package/register_modules.py`:

```python
from benchmark_ctrs import plugins

from my_package.models import MyMethod
from my_package.losses import MyLoss


@plugins.hookimpl
def register_models():
    return (MyMethod,)


@plugins.hookimpl
def register_criterions():
    return (MyLoss,)
```

Available hooks are:

- `register_callbacks()`
- `register_data_modules()`
- `register_models()`
- `register_certification_methods()`
- `register_criterions()`
- `register_lr_schedulers()`

### Implementing A New Training Method

Subclass `benchmark_ctrs.modules.BaseModule` or one of the built-in modules.
The training and validation steps must return a dictionary containing at least:

- `loss`: the tensor to optimize;
- `predictions`: class logits or probabilities used for automatic accuracy
  metrics.

Example skeleton:

```python
from typing import Any

from torch.nn import CrossEntropyLoss

from benchmark_ctrs.modules import BaseModule
from benchmark_ctrs.types import Batch, StepOutput


class MyMethod(BaseModule):
    def __init__(
        self,
        *args: Any,
        criterion: CrossEntropyLoss | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=self.ignore_hyperparameters)
        self.criterion = criterion or CrossEntropyLoss()

    def training_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput:
        inputs, targets = batch[:2]
        predictions = self.forward(inputs)
        loss = self.criterion(predictions, targets)
        return {"loss": loss, "predictions": predictions}

    def validation_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput:
        inputs, targets = batch[:2]
        predictions = self.forward(inputs)
        loss = self.criterion(predictions, targets)
        return {"loss": loss, "predictions": predictions}
```

Use it in YAML after installing the extension:

```yaml
model:
  class_path: my_package.models.MyMethod
  init_args:
    sigma: 0.25
```

### Implementing A New Dataset

Subclass `benchmark_ctrs.datasets.BaseDataModule` and implement:

- `prepare_data()`;
- `setup(stage)`;
- `classes`;
- `mean`;
- `std`.

Register the class with `register_data_modules()`, then reference it with a
Lightning CLI `class_path`.

### Implementing A New Certification Method

Subclass `benchmark_ctrs.certification.CertificationMethod` and implement
`certify()` and `predict()`. Return `benchmark_ctrs.certification.Certificate`
objects. Register the class with `register_certification_methods()`.

## Development

Format and lint with Ruff:

```bash
ruff check .
```

Useful smoke checks after editing package metadata or plugin registration:

```bash
python -m pip install -e .
python -c "import benchmark_ctrs; print(benchmark_ctrs.__version__)"
benchmark-ctrs fit --print_config
```

## License

This project is licensed under the MIT License. See `LICENSE`.
