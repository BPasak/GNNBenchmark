# When Events Meet Graphs: Evaluating Graph Neural Networks for Event-Based Vision with Object Classification

This repository is a benchmark implementation that replicates and compares Graph Neural Network approaches for event-based vision. It includes implementations based on three key papers:

1. [**AEGNN: Asynchronous Event-based Graph Neural Networks**](http://rpg.ifi.uzh.ch/docs/CVPR22_Schaefer.pdf) (CVPR 2022) - Schaefer, Gehrig, and Scaramuzza
2. [**EGSST: Event-based Graph Spatiotemporal Sensitive Transformer for Object Detection**](https://neurips.cc/virtual/2024/poster/94394) (NeurIPS 2024) - Sheng Wu, Hang Sheng, Hui Feng and Bo Hu
3. [**EvGNN: An Event-driven Graph Neural Network Accelerator for Edge Vision**](https://arxiv.org/abs/2404.19489) - (IEEE Transactions on Circuits and Systems for Artificial Intelligence) - Yufeng Yang, Adrian Kneip and Charlotte Frenkel 

The codebase supports flexible Graph Neural Network models that can be evaluated across multiple event-based vision benchmarks.

## File Structure

The repository is organized into the following directories:

- **`confs/`** - Configuration files for model training (e.g., `rtdtr_head_gen1.yml`, `rtdtr_head.yml`)
- **`src/`** - Main source code containing:
  - `Benchmarks/` - Model testing and benchmarking utilities (`ModelTester.py`, `OnModule.py`)
  - `Datasets/` - Dataset handling and preprocessing (`base.py`, `batching.py`, etc.)
  - `External/` - External dependencies and utilities
  - `Models/` - Model implementations
  - `utils/` - Utility functions
- **`training_scripts/`** - Jupyter notebooks and scripts for training models:
  - `Train_Recognition_Model.ipynb` - Training script for recognition tasks
  - `Train_Detection_Model.ipynb` - Training script for detection tasks
- **`testing_scripts/`** - Jupyter notebooks and scripts for testing and evaluation:
  - `Accuracy_Computation.ipynb` - Compute model accuracy
  - `Compute_mAP.ipynb` - Compute mean Average Precision
  - `EvGNN_AEGNN_async_jpn.ipynb` - Asynchronous evaluation notebook

## Installation

This project requires PyTorch and the [PyG (PyTorch Geometric)](https://github.com/pyg-team/pytorch_geometric) framework. The environment is configured for CUDA 12.4 with GPU support.

Install the project requirements using the provided environment file:

```bash
conda env create --file=environment_windows.yaml
conda activate GNNBenchmark
```

As for measuring the power consumption it requires https://github.com/GreenAI-Uppa/AIPowerMete. Here are the commands required to install it as an operating system Linux is required due to Linux specific drivers :

```bash
git clone https://github.com/GreenAI-Uppa/AIPowerMeter
pip install AIPowerMeter/
```

#### Known Issues

- When manually installing packages you might encounter a problem with `torch-scatter` and `torch-sparse`: "torch not found". In such a case once you install torch, you can try to install the package using the following command:

```bash
pip install torch-scatter --no-build-isolation
```

## Datasets

We evaluated our approach on three datasets:
- [NCars](http://www.prophesee.ai/dataset-n-cars/)
- [NCaltech101](https://www.garrickorchard.com/datasets/n-caltech101)
- [Prophesee Gen1 Automotive](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)

### Data Preprocessing

The preprocessing of the datasets is contained in (src/Datasets) for detailed instructions on data structure and preprocessing procedures.

## Training

Train recognition or detection models using the provided notebooks:

```bash
# Recognition model training
jupyter notebook training_scripts/Train_Recognition_Model.ipynb

# Detection model training
jupyter notebook training_scripts/Train_Detection_Model.ipynb
```

## Evaluation

Evaluate your trained models using:

```bash
# For accuracy computation
jupyter notebook testing_scripts/Accuracy_Computation.ipynb

# For mean Average Precision (mAP)
jupyter notebook testing_scripts/Compute_mAP.ipynb

# For asynchronous model testing
python testing_scripts/EVGNN_AEGNN_async_test_comp.py
```

## Model Architecture

The codebase supports flexible Graph Neural Network models that can be made asynchronous and sparse. Tested model configurations for EGSST are stored in YAML files in the `confs/` directory.
