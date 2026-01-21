# When Events Meet Graphs: Evaluating Graph Neural Networks for Event-Based Vision with Object Classification

This repository contains code from our 2022 CVPR paper [**AEGNN: Asynchronous Event-based Graph Neural Networks**](http://rpg.ifi.uzh.ch/docs/CVPR22_Schaefer.pdf) by Simon Schaefer*, [Daniel Gehrig*](https://danielgehrig18.github.io/), and [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html). 

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
  - `EVGNN_AEGNN_training.ipynb` - AEGNN-specific training notebook
  - `Accuracy_Computation.ipynb` - Compute model accuracy
  - `Compute_mAP.ipynb` - Compute mean Average Precision
- **`testing_scripts/`** - Jupyter notebooks and scripts for testing and evaluation:
  - `EVGNN_AEGNN_async_test_comp.py` - Asynchronous model testing
  - `EvGNN_AEGNN_async_jpn.ipynb` - Asynchronous evaluation notebook
  - `EVGNN_AEGNN_sync_test.ipynb` - Synchronous model testing

## Installation

This project requires PyTorch and the [PyG (PyTorch Geometric)](https://github.com/pyg-team/pytorch_geometric) framework. The environment is configured for CUDA 12.4 with GPU support.

Install the project requirements using the provided environment file:

```bash
conda env create --file=environment_windows.yaml
conda activate GNNBenchmark
```

As for measuring the power consumption it requires https://github.com/GreenAI-Uppa/AIPowerMete. Here are the commands required to install it (linux may be needed as the operating system???) :

```bash
git clone https://github.com/GreenAI-Uppa/AIPowerMeter
pip install AIPowerMeter/
```

## Datasets

We evaluated our approach on three datasets:
- [NCars](http://www.prophesee.ai/dataset-n-cars/)
- [NCaltech101](https://www.garrickorchard.com/datasets/n-caltech101)
- [Prophesee Gen1 Automotive](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)

Download and extract the datasets. By default, they are assumed to be in `/data/storage/`. You can change this by setting the `AEGNN_DATA_DIR` environment variable.

### Data Preprocessing

The preprocessing of the datasets is contained in (src/Datasets) for detailed instructions on data structure and preprocessing procedures.

## Training

Train recognition or detection models using the provided notebooks:

```bash
# Recognition model training
jupyter notebook training_scripts/Train_Recognition_Model.ipynb

# Detection model training
jupyter notebook training_scripts/Train_Detection_Model.ipynb

# AEGNN-specific training
jupyter notebook training_scripts/EVGNN_AEGNN_training.ipynb
```

## Evaluation

Evaluate your trained models using:

```bash
# For accuracy computation
jupyter notebook training_scripts/Accuracy_Computation.ipynb

# For mean Average Precision (mAP)
jupyter notebook training_scripts/Compute_mAP.ipynb

# For asynchronous model testing
python testing_scripts/EVGNN_AEGNN_async_test_comp.py
```

## Model Architecture

The codebase supports flexible Graph Neural Network models that can be made asynchronous and sparse. Model configurations are stored in YAML files in the `confs/` directory.

## Contributing

If you spot any bugs or plan to contribute bug-fixes, please open an issue and discuss it with us.