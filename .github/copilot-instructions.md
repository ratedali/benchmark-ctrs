# Copilot Instructions for `benchmark-ctrs`

## Project Purpose

This project provides a framework for training, testing, and benchmarking certified training methods for randomized smoothing robustness certification 
on standard datasets (e.g., MNIST, CIFAR-10, ImageNet). It is designed for research and comparison of certified robustness techniques in deep learning, 
with a focus on reproducibility and extensibility.

## Frameworks, Libraries, and Tools Used

- **Python 3.9+**
- **PyTorch** (`torch`, `torchvision`): Core deep learning framework
- **Lightning** (`lightning.pytorch`): High-level training loop and CLI
- **TorchMetrics**: Metrics for model evaluation
- **NumPy**, **SciPy**, **statsmodels**: Scientific computing and statistics
- **pluggy**: Plugin system for extensibility
- **ruff**: Linting (dev only)
- **pandas**, **tbparse**: Optional, for results analysis

## Codebase Structure

- `src/benchmark_ctrs/`
  - `cli/`: Command-line interface, entry points, plugin registration, and default configs
    - `__main__.py`: Main CLI entry point (`benchmark-ctrs` script)
    - `plugins.py`, `_default_plugin.py`, `plugins/`: Plugin system for registering models, datasets, and callbacks
    - `default_config_fit.yml`, `default_config_predict.yml`: Default Lightning config files
  - `datasets/`: Dataset modules for MNIST, CIFAR-10, ImageNet, and base data module
  - `models/`: Model architectures (LeNet, ResNet, smoothing wrappers, layers)
  - `modules/`: Training modules implementing certified training algorithms (e.g., GaussianAug)
  - `metrics/`: Custom metrics, e.g., certified radius
  - `callbacks/`: Lightning callbacks, e.g., certified radius writer
  - `plugins/`: Plugin specification and registration
  - `__init__.py`: Project version and docstring
- `datasets_cache/`: Downloaded datasets (ignored by VCS)
- `logs/`: Training and evaluation logs
- `tests/`: (Currently empty) — add tests here

## Adding New Features

- **New Model**: Implement in `src/benchmark_ctrs/models/` and register via plugin in `cli/_default_plugin.py`.
- **New Dataset**: Implement a new DataModule in `src/benchmark_ctrs/datasets/` and register it in the plugin.
- **New Training Algorithm**: Add a new module in `src/benchmark_ctrs/modules/` and register it.
- **New Metric/Callback**: Add to `metrics/` or `callbacks/` and register via plugin.
- **Plugins**: Use the pluggy system to extend models, datasets, or callbacks.

## Useful Commands

- **Install dependencies**:
  ```sh
  flit install --pth-file
  ```
- **Run training**:
  ```sh
  python -m benchmark_ctrs.cli fit --data [DataModule] --model [TrainingModule]
  ```
- **Run prediction/certification**:
  ```sh
  python -m benchmark_ctrs.cli predict --data [DataModule] --model [TrainingModule]
  ```
- **Lint code**:
  ```sh
  ruff check src/
  ```

## Coding Practices

- Follow [PEP8](https://peps.python.org/pep-0008/) and use `ruff` for linting.
- Use type annotations throughout (enforced by codebase).
- Prefer subclassing and plugin registration for extensibility.
- Keep dataset and model code modular and reusable.
- Document new modules, classes, and functions clearly.
- Use Lightning's configuration and callback system for experiment management.

## Additional Notes

- The project is licensed under the MIT License.
- All code should be compatible with Python 3.9+ and PyTorch 2.6+.
- For new datasets, ensure data is downloaded to `datasets_cache/`.
- For new metrics or callbacks, prefer using Lightning's callback and metric APIs.
- For questions, see the [GitHub repository](https://github.com/ratedali/benchmark_ctrs).

---

*This file is intended for coding assistants and contributors to understand and extend the project efficiently.*
