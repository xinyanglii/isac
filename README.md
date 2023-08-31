# Integrated Sensing and Communications

## Introduction

This package aims to provide a modularized, portable framework and useful tools for simulation, verification and deployment of integrated sensing and communications (ISAC) systems. It is implemented mainly based on [PyTorch](https://pytorch.org/), utilizing its modularization mechanism and GPU acceleration and making it possible to design more advanced TRx algorithms.

**Note: This package is still under development and is currently only used for personal research.**

## Installation

Pull this repository to your local directory and go into it:

```bash
git clone https://github.com/xinyanglii/isac
cd isac
```

Create a `conda` virtual environment with `python3.8` and `pip` pre-installed:

```bash
conda create -n isac python=3.8 pip
conda activate isac
```

Install this package and its dependencies using `pip`:

```bash
pip install -r requirements.txt
pip install -e .
```

Now you are able to import this package in your code such as:

```python
import isac
from isac.module.ofdm import OFDMModulator
...
```

## Contributing

To contribute to this project, please install the development dependencies and install the `pre-commit` hooks

```bash
pip install -r requirements_dev.txt
pre-commit install
```

Tests are written using `pytest`. To run the tests, simply run

```bash
pytest tests
pytest tests --all # for a full parameter sweep
```
