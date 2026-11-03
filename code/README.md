# Flow Equivariant State Space Model

Code repository for **Flow Equivariant State Space Model** (NeurIPS 2026).

## Overview

Many spatiotemporal sequences contain structures that move over time. Conventional spatial state space models update recurrent memory at fixed coordinates, so evidence from a moving object can be accumulated at inconsistent locations across frames.

**Flow-SSM** addresses this problem by maintaining recurrent memory over candidate motion branches. Before each selective state update, a branch transports its hidden state along the corresponding candidate flow. This lets the model accumulate temporal information in a motion-aligned reference frame while retaining input-dependent state space dynamics.

**Hierarchical Flow-SSM** extends this design with a coarse-to-fine organization of flow-aligned memory, allowing broad motion and local deformation to be represented at multiple scales.

## Experiments

| Dataset | Task | Models |
| --- | --- | --- |
| Moving MNIST | Video prediction with controlled translation | Flow-SSM, FERNN |
| Rotating MNIST | Video prediction with controlled rotation | Flow-SSM, FERNN |
| KTH | Action recognition under induced translation | Flow-Mamba, plain Mamba, FERNN |
| SEVIR | VIL weather forecasting | Mamba, FERNN, Flow-SSM, Hierarchical Flow-SSM |
| UCF101 | out-of-grid Video prediction | Mamba, Flow-Mamba, FERNN |

## Repository Structure

| Directory | Contents |
| --- | --- |
| `MovingMNIST/` | Moving MNIST models, dataset utilities, training, evaluation, and tests |
| `RotatingMNIST/` | Rotating MNIST models, dataset utilities, training, evaluation, and tests |
| `KTH/` | KTH action-recognition implementation  |
| `SEVIR/SEVIR_FlowSSM/` | SEVIR datasets, models, metrics, training, evaluation, and hierarchy probes |
| `UCF101/` | UCF101 data preparation, models, video prediction, evaluation, and probing |


## Installation

Python 3.10 or later is recommended. Install dependencies from the directory of the experiment you intend to run:

```bash
# Moving MNIST
cd MovingMNIST && pip install -r requirements.txt

# Rotating MNIST
cd ../RotatingMNIST && pip install -r requirements.txt

# SEVIR
cd ../SEVIR/SEVIR_FlowSSM && pip install -r requirements.txt

# UCF101
cd ../../UCF101 && pip install -r requirements.txt
```

KTH does not currently include a `requirements.txt`. Its main dependencies can be installed with:

```bash
cd KTH
pip install torch numpy opencv-python h5py matplotlib tqdm
```


## Data Preparation

- **Moving MNIST / Rotating MNIST:** MNIST is downloaded automatically on the first run and cached locally. Use the training script's `--root` option to change the data location.
- **KTH:** The script downloads the source videos and sequence annotations when needed, then preprocesses the videos into an HDF5 cache. The data directory can be changed with `--data_root`.
- **SEVIR:** Place the VIL HDF5 files in subdirectories under a data root, set `SEVIR_DATA_ROOT`, and build the index before training:

  ```bash
  export SEVIR_DATA_ROOT=/path/to/sevir_vil
  cd SEVIR/SEVIR_FlowSSM
  python -m tools.build_vil_index
  ```

- **UCF101:** Download and extract the videos and official recognition splits with:

  ```bash
  cd UCF101
  python prepare_ucf101.py --root ./data/ucf101
  ```

  The video archive is large. RAR extraction requires a system utility such as `unrar` or `7z`. The resulting data directory should contain `UCF-101/` and `ucfTrainTestlist/`.

## Running Experiments


```bash
# Moving MNIST
cd MovingMNIST
python train.py --model flow_ssm
python evaluate.py --checkpoint path/to/best.pt

# Rotating MNIST
cd ../RotatingMNIST
python train.py --model flow_ssm
python evaluate.py --checkpoint path/to/best.pt

# KTH
cd ../KTH
python kth.py --model flow_mamba
python kth.py analyze --checkpoint path/to/best.pt

# SEVIR
cd ../SEVIR/SEVIR_FlowSSM
python -m scripts.train --model flow_ssm --run_name flow_ssm_run
python -m scripts.evaluate --checkpoint path/to/best.pt

# Hierarchical Flow-SSM (initialized from a Flow-SSM checkpoint)
python -m scripts.train --model hier_flow_ssm \
  --run_name hier_flow_ssm_run \
  --flow_checkpoint path/to/flow_ssm/best.pt

# UCF101
cd ../../UCF101
python run.py pretrain --model flow_mamba \
  --video-root ./data/ucf101/UCF-101 \
  --annotation-root ./data/ucf101/ucfTrainTestlist
python run.py evaluate \
  --video-root ./data/ucf101/UCF-101 \
  --annotation-root ./data/ucf101/ucfTrainTestlist \
  --checkpoint path/to/best.pt
python run.py probe \
  --video-root ./data/ucf101/UCF-101 \
  --annotation-root ./data/ucf101/ucfTrainTestlist \
  --checkpoint path/to/best.pt
```

The available model choices are listed in the experiment table and exposed by the corresponding scripts. For SEVIR, Hierarchical Flow-SSM training requires a Flow-SSM checkpoint and accepts it through `--flow_checkpoint`. For UCF101, the workflow includes video-prediction pretraining, checkpoint evaluation, and frozen-backbone linear probing.

