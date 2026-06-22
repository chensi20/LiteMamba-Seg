# LiteMamba-Seg

Official PyTorch implementation of **LiteMamba-Seg** for polyp segmentation.

## Overview
LiteMamba-Seg is a lightweight polyp segmentation framework designed to balance segmentation accuracy, computational efficiency, and cross-dataset robustness. The model is built on a U-Net-like encoder-decoder architecture with a ResNet34 backbone and selective Mamba-based contextual modeling.

## Dataset Preparation
LiteMamba-Seg is evaluated on the following public polyp segmentation benchmarks:

- **Kvasir-SEG**
- **CVC-ClinicDB**
- **CVC-ColonDB**
- **ETIS-LaribPolypDB**

### Option 1: Download from original public sources
Please obtain the datasets from their original public sources whenever available:

- **Kvasir-SEG**: `https://datasets.simula.no/kvasir-seg/`
- **CVC-ClinicDB / CVC-ColonDB**: `https://pages.cvc.uab.es/CVC-Colon/index.php/databases/`

For **ETIS-LaribPolypDB**, please refer to the original source cited in the manuscript or use the prepared split in Option 2 below.

### Option 2: Use a prepared polyp segmentation split
For easier reproduction, users may also follow the widely adopted data organization used in prior polyp segmentation repositories. One commonly used reference is the PraNet repository:

- **PraNet repository**: `https://github.com/DengPingFan/PraNet`

The PraNet repository provides a commonly used training/testing data organization for polyp segmentation research and can be used as a reference for preparing local dataset folders.

After downloading, organize the local dataset directory according to the paths used in `dataset.py`. A typical structure is as follows:

```text
datasets/
├── TrainDataset/
│   ├── image/
│   └── masks/
└── TestDataset/
    ├── CVC-ClinicDB/
    │   ├── images/
    │   └── masks/
    ├── CVC-ColonDB/
    │   ├── images/
    │   └── masks/
    ├── ETIS-LaribPolypDB/
    │   ├── images/
    │   └── masks/
    └── Kvasir/
        ├── images/
        └── masks/
```

Please adjust the folder names if your local implementation uses different paths.

## Pretrained Weights
The pretrained model weights are archived on Zenodo:

- **DOI:** `https://doi.org/10.5281/zenodo.20798373`

### Setup Steps
1. Create a folder named `./checkpoints/` in the project root directory.
2. Download the pretrained checkpoint from the Zenodo record.
3. Place the downloaded file into `./checkpoints/`.
4. Rename the checkpoint file to match `Config.EXP_NAME` defined in `config.py`.

For example, if `Config.EXP_NAME` is set to:

```python
best_model_Conv_Bottleneck_D4
```

then the checkpoint file should be renamed as:

```text
./checkpoints/best_model_Conv_Bottleneck_D4.pth
```

before running evaluation.

## Environment Setup
The code was tested with the following environment:

- Python 3.10.8
- PyTorch 2.1.2
- torchvision 0.16.2
- CUDA 11.8
- NVIDIA GeForce RTX 3080 Ti

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

If you prefer to use conda, you may create an environment first:

```bash
conda create -n litemamba python=3.10 -y
conda activate litemamba
pip install -r requirements.txt
```

Please note that `mamba-ssm` and `causal-conv1d` are environment-sensitive dependencies. The experiments in this repository were tested with CUDA 11.8, PyTorch 2.1, and Python 3.10. You may need to install the corresponding compatible wheels according to your local system configuration.

If you encounter OpenMP-related warnings on some systems, you may optionally run:

```bash
export OMP_NUM_THREADS=1
```

before training or evaluation.

## Evaluation
To reproduce the reported test results, run:

```bash
python test.py
```

Please make sure that:
- the dataset path is correctly configured
- the pretrained checkpoint is placed in `./checkpoints/`
- the checkpoint filename matches `Config.EXP_NAME` in `config.py`

## Training
To train LiteMamba-Seg from scratch, run:

```bash
python train.py
```

Please modify the dataset paths according to your local setup and training protocol.

## Outputs
Predicted masks and evaluation results will be saved to the output directory defined in the testing script. You may also save qualitative results for visualization and comparison.

## Notes
- Public datasets are **not redistributed** in this repository or in the Zenodo record.
- Please follow the corresponding dataset licenses and usage conditions.
- The dataset links above are provided for reproducibility. Please refer to the original publications for detailed dataset descriptions and citation information.
- The PraNet repository link is provided as a reference for dataset organization and reproducibility rather than as the official source of all datasets.
- This repository is intended for research and reproducibility purposes only.

## Citation
If you find this repository useful, please cite the corresponding paper:

```bibtex
@article{LiteMambaSeg,
  title={Efficient and Real-Time Polyp Segmentation: A Systematic Evaluation of Lightweight Mamba Integration Strategies},
  author={...},
  journal={...},
  year={2026}
}
```
