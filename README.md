# DyFAIP_UAI2026

## DyFAIP: End-to-End Imputation and Prediction via Adaptive Frequency Learning in Irregular Time Series

This repository contains the implementation for **DyFAIP**, a dual-path framework for time-series imputation and downstream tasks. The model leverages frequency-driven gating and feature-hidden representations to handle missing data in complex temporal datasets.

## 🛠 Prerequisites

Ensure your environment meets the following version requirements to maintain compatibility with the tensor operations and broadcasting logic used in the framework.

*   **Python:** 3.10+
*   **PyTorch:** `2.5.1`
*   **NumPy:** `2.0.1`
*   **Additional Dependencies:** `matplotlib`, `scipy`, `pandas`, `scikit-learn`, `jupyter`
*   
## 🚀 Getting Started

The execution pipeline consists of two primary phases: data preparation and model execution.

### Step 1: Data Preprocessing
The model utilizes the **Electricity Transformer Dataset (ETDataset)**. Before training, the raw data must be cleaned, normalized, and formatted into temporal windows.

1.  Open the Jupyter Notebook:
    `Electricity Transformer Dataset (ETDataset) Data preprocessing UAI-2026.ipynb`
2.  Run all cells to generate the processed `.npz` files required by the runner.
3.  Ensure the output directory matches the path defined in `Runner.py`.

### Step 2: Model Training & Evaluation
Once the data is preprocessed, use the `Runner.py` script to initialize the DyFAIP architecture, execute the training loop, and perform imputation evaluation.

```bash
python Runner.py
```

## 📁 Repository Structure

*   `UAI-2026.ipynb`: Complete preprocessing pipeline for ETDataset.
*   `helpers/Runner.py`: Main entry point for training and evaluation.
*   `models/GTACM.py`: Core architecture including the `gtacm` and frequency gating modules.

---
