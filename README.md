
# 🧠 Morgoth: Toward Automated EEG Interpretation

This repository contains code and tools for running EEG analysis using **Morgoth**, including and event-level and EEG-level detection with probabilities for:

- Normal / Abnormal
- Slowing: No Slowing / Focal Slowing / Generalized Slowing
- Burst suppression: No / Burst suppression
- Spike detection: No / Spike
- Spike localization: No / Focal Spike / Generalized Spike  
- IIIC classification: Other / Seizure / LPD / GPD / LRDA / GRDA
- Sleep staging: Awake / N1 / N2 / N3 / REM
## 📁 Directory Structure

```
├── test_data/                  # EEG data (.mat/.pkl/.edf)
├── checkpoints/                # Pretrained models and checkpoints
├── xxx.py                      # Code files
├── xxx.sh/bat                  # Run model
└── README.md                   # Project overview
```
## 📥 Download Model and Test Data

Before running the code, please download the pretrained model (checkpoints) and test dataset (test_data) from Dropbox and place them in the appropriate folders:

- [Download Link – Model and Data](https://www.dropbox.com/) 

## ⚙️ Setup

### 1. Create environment (conda recommended) in morgoth folder
### python 3.12 + pytorch 2.4 + CUDA 12.4

```bash
conda create -n morgoth python=3.12 

conda activate morgoth

pip install -r requirements.txt

# If you have GPU with cuda driver
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```


### 2. Optional: python 3.11 + pytorch 2.1 + CUDA 12.2 / 12.1
```bash
conda create -n morgoth python=3.12 

conda activate morgoth

pip install -r requirements_cuda122.txt

# If you have GPU with cuda driver
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3. If Windows and no GPU
```bash
conda create -n morgoth python=3.12 

conda activate morgoth

pip install -r requirements_windows.txt
```


## 🚀 Usage

### 1. Run continuous prediction on event level

See the comments at the beginning of the Bash file to understand the meaning of each parameter in the command. You can modify parameters inside each script.

```bash
bash continuous_event_level.sh
```

### 2. Run discrete prediction
See the comments at the beginning of the Bash file to understand the meaning of each parameter in the command. You can modify parameters inside each script.

The data in test_data folder should be processed first.

```bash
bash discrete_event_level.sh
```

### 3. Run EEG level prediction
See the comments at the beginning of the Bash file to understand the meaning of each parameter in the command. You can modify parameters inside each script.

The EEG_level inputs are the event-level output results with 1-second slipping step.

```bash
bash EEG_level.sh
```

### 4.Optional: If linux and cpu

```bash
bash bash EEG_level_cpu.sh
```

### 5.Optional: If windows and cpu

```bash
EEG_level_windows_cpu.bat
```


## 📬 Contact

For questions, please contact:  
Chenxi Sun – csun8@bidmc.harvard.edu
