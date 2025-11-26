# Robust Federated Learning – Final Project

This repository contains the full implementation for the **CSC 8370 – Data Security Final Project**, covering:

- **Level 1:** Centralized MNIST Training
- **Level 2:** Federated Learning with 10 Clients
- **Level 3:** Robust Federated Learning with Malicious Client Detection
- **Model Testing Script:** Load saved model and evaluate on MNIST test set

All code, experimentation, architecture design, hyperparameter tuning, evaluation, and documentation were completed individually as the sole team member.

---

## 📂 Repository Structure

```
├── Level1_solution.py
├── Level2_solution.py
├── Level3_solution.py
├── model_definition.py
├── test_saved_model.py
├── requirements.txt
└── README.md
```

---

## ✔ Environment Setup (Using Conda)

This project was developed using **Miniconda**. Use the exact steps below to reproduce the environment.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

conda create --name final_project python=3.6
conda activate final_project

pip install torch==2.1.0 torchvision==0.15.0 numpy==1.24.0
pip install -r requirements.txt   # optional
```

---

## ▶️ Running Each Level

### **Level 1 – Centralized Training**
```bash
python3 Level1_solution.py
```

### **Level 2 – Federated Learning**
```bash
python3 Level2_solution.py
```

### **Level 3 – Robust Federated Learning with Malicious Detection**
```bash
python3 Level3_solution.py
```

---

## 🧪 Testing Saved Models

Use the included testing script to load a saved model (e.g., `Level1.pth`) and evaluate on MNIST test data.

### **Run Model Test**
```bash
python3 test_saved_model.py
```

### **test_saved_model.py Overview**
- Loads the MNIST test set
- Loads the trained model from disk
- Runs inference over all test batches
- Prints final accuracy

---

## 🌐 Clone This Repository

```bash
git clone https://github.com/gchkiran/data-security
```

To push your changes:

```bash
git add .
git commit -m "Updated project files"
git push origin main
```

---

## 📦 Dependencies

All dependencies are listed in **requirements.txt**.

```
torch==2.1.0
torchvision==0.15.0
numpy==1.24.0
```

---

## 📘 Project Notes

- Entire architecture, training framework, and robust FL design were implemented manually.
- Hyperparameter tuning was performed to select the best configurations across levels.
- Code for all three levels and evaluation scripts is deployed in a single GitHub repository for reproducibility and clarity.

---

## Contact

**Author:** Guntupalli Chandra Kiran  
This repository contains the full work submitted individually for the project.
