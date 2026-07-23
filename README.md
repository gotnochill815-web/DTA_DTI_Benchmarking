# DTA & DTI Benchmarking

Benchmarking and reproducible implementation of deep learning models for Drug–Target Affinity (DTA) prediction.

This repository provides end-to-end pipelines for training, evaluation, prediction, and benchmarking of two popular DTA models:

- **GraphDTA** (Graph Neural Networks for molecular graphs)
- **MolTrans** (Transformer-based Drug–Target Interaction model)

The project focuses on reproducibility, multi-seed evaluation, and robust benchmarking.

---

## Repository Structure

```
DTA_DTI_Benchmarking/
│
├── GraphDTA_Benchmark/
│   ├── models/
│   ├── utils/
│   ├── data/
│   ├── train.py
│   ├── predict.py
│   └── README.md
│
├── MolTrans_Benchmark/
│   ├── models/
│   ├── configs/
│   ├── train.py
│   ├── predict.py
│   └── README.md
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Overview

Drug–Target Affinity prediction estimates the binding strength between a small molecule and a target protein.

This repository benchmarks two widely used architectures:

## GraphDTA

GraphDTA represents

- Drug as a molecular graph
- Protein as an amino acid sequence

Drug graphs are encoded using Graph Neural Networks while proteins are encoded using a CNN.

Implemented GraphDTA variant:

- Graph Convolution Network (GCN)

---

## MolTrans

MolTrans models drug–protein interactions using transformer-based representations.

The implementation includes

- Tokenization
- Model training
- Prediction
- Evaluation
- Multi-seed benchmarking

---

# Features

- Reproducible training pipeline
- Multi-seed experiments
- Bootstrap evaluation
- Prediction pipeline
- Standard evaluation metrics
- Easy dataset preprocessing
- Modular project structure

---

# Evaluation Metrics

The benchmarking pipeline reports

- RMSE
- Pearson Correlation
- Spearman Correlation
- Concordance Index (CI)
- R² Score

---

# GraphDTA Results

| Seed | RMSE | R² | Pearson |
|------|------|------|---------|
| 42 | 1.0661 | 0.5439 | 0.7400 |
| 52 | 1.0515 | 0.5505 | 0.7437 |
| 62 | **1.0416** | **0.5711** | **0.7569** |
| 72 | 1.1217 | 0.5099 | 0.7164 |

---

Average Performance

| Metric | Value |
|---------|-------|
| RMSE | 1.109 ± 0.092 |
| Pearson | 0.715 ± 0.056 |
| Spearman | 0.673 ± 0.054 |
| CI | 0.749 ± 0.024 |
| R² | 0.510 ± 0.079 |

---

# Reproducibility

Experiments were performed using multiple random seeds to ensure stable and reproducible evaluation.

Seeds used:

```
32
42
52
62
72
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/gotnochill815-web/DTA_DTI_Benchmarking.git
cd DTA_DTI_Benchmarking
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Training

## GraphDTA

```bash
cd GraphDTA_Benchmark

python train.py
```

## MolTrans

```bash
cd MolTrans_Benchmark

python train.py
```

---

# Prediction

GraphDTA

```bash
python predict.py
```

MolTrans

```bash
python predict.py
```

---

# Datasets

This repository does **not** include processed datasets or trained checkpoints due to their large size.

Users should prepare datasets following the preprocessing scripts included in each benchmark.

---

# Repository Contents

Included

- Source code
- Training pipeline
- Evaluation scripts
- Prediction scripts
- Configuration files
- Benchmarking utilities

Excluded

- Processed datasets
- Model checkpoints
- Generated outputs
- Cached graph files

---

# Citation

If you use this repository in your work, please cite the original GraphDTA and MolTrans papers.

---

# Acknowledgements

This repository builds upon the original implementations of

- GraphDTA
- MolTrans

and extends them with reproducible benchmarking, multi-seed evaluation, and standardized performance reporting.
