# ALoRa

Official code for **[Low Rank Transformer for Multivariate Time Series Anomaly Detection and Localization](https://openreview.net/pdf?id=ZtPIBpVojC)**, ICLR 2026.
## Method Overview

<p align="center">
  <img src="images/NewArchitecture_ICLR_3.png" width="700"/>
  <br>
</p>




## 🛠 Environment & System Info

This project was developed and tested under the following environment:

- **Python version**: 3.7.6  
- **PyTorch version**: 1.10.2+cu102  
- **CUDA version (PyTorch)**: 10.2  
- **NVIDIA Driver version**: 440.64.00  
- **GPU**: 2 × Tesla V100-PCIE-32GB  
- **OS**: Linux (bash 4.2 environment)

To check your environment setup, run:

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
nvidia-smi
```

---

## ⚙️ Setup & Installation

We recommend using a Python virtual environment for a clean installation.

### Step 1: Create and Activate a Virtual Environment

```bash
# Navigate to your project directory (if not already there)
cd ./ALoRa

# Create the virtual environment
python -m venv ./alora_env

# Activate the virtual environment
source ./alora_env/bin/activate      # For Linux/macOS
# OR
alora_env\Scripts\activate           # For Windows
```

### Step 2: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 3: Install PyTorch with CUDA 10.2 Support

```bash
pip install torch==1.10.2+cu102 torchvision==0.11.3+cu102 torchaudio==0.10.2+cu102 \
  -f https://download.pytorch.org/whl/cu102/torch_stable.html
```

### Step 4: Install Other Required Packages

```bash
pip install -r ./requirements.txt
```
## 📂 Dataset Format & Organization

Each dataset should be stored in the following directory:
```
./ALoRa/Datasets/<DATASET_NAME>/
```

### Required Files

For each dataset, the folder must include:

- `train.csv` — training data  
- `test.csv` — testing data  
- `test_label.csv` — anomaly labels corresponding to `test.csv` (0 = normal, 1 = anomaly)

These file names must match what is expected in the dataset loader logic inside:
```
data_factory/data_loader.py
```

Specifically, the loader class named `{DatasetName}SegLoader` handles loading for each dataset. If you use custom formats, you must modify this class accordingly.

### ⚠️ If Using `.npy` Files

If your dataset is saved as `.npy` files (e.g., `train.npy`, `test.npy`, `test_label.npy`), you need to update the corresponding loader:

Example change inside `{DatasetName}SegLoader`:

```python
# Instead of:
train = pd.read_csv(os.path.join(data_path, 'train.csv'))

# Use:
train = np.load(os.path.join(data_path, 'train.npy'))
```

### Public Datasets

We include the **SMD** and **HAI** dataset in this repository to allow easy reproducibility of the results.  
Other datasets require download permissions or request-based access.

The datasets used in this work are listed below:
- **SMD**: [https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset)
- **PSM**: [https://github.com/eBay/RANSynCoders/tree/main/data](https://github.com/eBay/RANSynCoders/tree/main/data)
- **MSL**: [https://github.com/khundman/telemanom](https://github.com/khundman/telemanom)
- **SWaT**: [https://itrust.sutd.edu.sg/itrust-labs-home/itrust-labs_swat/](https://itrust.sutd.edu.sg/itrust-labs-home/itrust-labs_swat/)
- **HAI**: [https://github.com/icsdataset/hai]
- **MSDS**: [https://zenodo.org/record/3842450](https://zenodo.org/record/3842450)


## How to Run (Anomaly Detection)

All training and testing commands for each dataset are included in the `How_to_run.sh` script.

Below is an example for the **SMD** dataset:

```bash
# Train on GPU
python main.py --num_epochs 4 --batch_size 128 --mode train --dataset SMD --data_path ./Datasets/SMD --input_c 38 --output_c 38 --win_size 20 --d_model 702

# Test after training
python main.py --num_epochs 4 --batch_size 128 --mode test --dataset SMD --data_path ./Datasets/SMD --input_c 38 --win_size 20 --d_model 702 --rank_threshold 0.01
```

## How to Run (Anomaly Localization)

All localization commands for each dataset are included in the `./Localization/Loc_How_to_run.sh` script.

Below is an example for the **SMD** dataset:

```bash
python ./Localization/ALoRa_Loc.py --dataset SMD
```
Please check the `instructions.txt` file for detailed usage instructions.

## Requirements
This project requires Python 3.7 or higher. Please see the `requirements.txt` file for a list of necessary libraries and packages.

## Citation Request
If you find our paper or any part of our code useful, please cite our work as follows:
@inproceedings{
shimillas2026low,
title={Low Rank Transformer for Multivariate Time Series Anomaly Detection and Localization},
author={Charalampos Shimillas and Kleanthis Malialis and Konstantinos Fokianos and Marios Polycarpou},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=ZtPIBpVojC}
}
