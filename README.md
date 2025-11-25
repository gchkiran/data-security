# Robust Federated Learning - Final Project

This repository contains the implementation for the CSC 8370 Final Project, covering:

- Level 1: Centralized Learning
- Level 2: Federated Learning
- Level 3: Robust Federated Learning with Malicious Client Detection

## 🚀 Environment Setup (Using Conda)

Follow the exact commands used during development:



```bash

# Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

conda create --name final_project python=3.6
conda activate final_project

pip install torch==2.1.0 torchvision==0.15.0 numpy==1.24.0
```

## ▶️ Running the Project

Once dependencies are installed, run the three levels:

```bash
python3 Level1_solution.py
python3 Level2_solution.py
python3 Level3_solution.py
```

## 📦 Notes

- The environment is based on Python 3.6
- You may install additional dependencies using `pip install -r requirements.txt`
- All experiments were tested on Linux (Ubuntu)
