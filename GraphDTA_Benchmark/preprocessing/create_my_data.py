"""
Convert CSV files into GraphDTA PyTorch Geometric datasets.

Usage:

python -m preprocessing.create_my_data --seed 32
python -m preprocessing.create_my_data --seed 42
python -m preprocessing.create_my_data --seed 52
python -m preprocessing.create_my_data --seed 62
python -m preprocessing.create_my_data --seed 72
"""

import os
import sys
import pickle
import argparse

import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from preprocessing.graph_utils import smile_to_graph
from preprocessing.protein_encoder import encode_protein
from graph_dataset.dataset import GraphDTADataset


# ==========================================================
# Arguments
# ==========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--seed",
    type=int,
    required=True,
    help="Dataset number (32, 42, 52, 62, 72)"
)

args = parser.parse_args()

SEED = args.seed

print("=" * 60)
print(f"Preparing Dataset : {SEED}")
print("=" * 60)


# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = "/content/GraphDTA-Benchmark"

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)


# ==========================================================
# Load CSVs
# ==========================================================

print("\nLoading CSV files...")

train_df = pd.read_csv(
    os.path.join(RAW_DATA_DIR, f"train_{SEED}.csv")
)

val_df = pd.read_csv(
    os.path.join(RAW_DATA_DIR, f"val_{SEED}.csv")
)

test_df = pd.read_csv(
    os.path.join(RAW_DATA_DIR, f"test_{SEED}.csv")
)

# Remove rows having missing labels
train_df = train_df.dropna(subset=["drug_seq", "target_seq", "pKd"]).reset_index(drop=True)
val_df = val_df.dropna(subset=["drug_seq", "target_seq", "pKd"]).reset_index(drop=True)
test_df = test_df.dropna(subset=["drug_seq", "target_seq", "pKd"]).reset_index(drop=True)

print("Train :", train_df.shape)
print("Val   :", val_df.shape)
print("Test  :", test_df.shape)


# ==========================================================
# Build SMILES Graph Dictionary
# ==========================================================

print("\nCollecting unique SMILES...")

all_smiles = set()

for df in [train_df, val_df, test_df]:
    all_smiles.update(df["drug_seq"].tolist())

print(f"Unique molecules : {len(all_smiles)}")

smile_graph = {}

for i, smile in enumerate(all_smiles):

    if (i + 1) % 500 == 0 or i == len(all_smiles):
        print(f"Building graph {i+1}/{len(all_smiles)}")

    smile_graph[smile] = smile_to_graph(smile)

graph_path = os.path.join(
    METADATA_DIR,
    f"smile_graph_{SEED}.pkl"
)

with open(graph_path, "wb") as f:
    pickle.dump(smile_graph, f)

print(f"Saved graph dictionary -> {graph_path}")


# ==========================================================
# Helper
# ==========================================================

def build_dataset(df, dataset_name):

    smiles = df["drug_seq"].values

    proteins = np.array(
        [encode_protein(seq) for seq in df["target_seq"]]
    )

    labels = df["pKd"].astype(np.float32).values

    GraphDTADataset(
        root=os.path.join(PROJECT_ROOT, "data"),
        dataset_name=dataset_name,
        smiles=smiles,
        proteins=proteins,
        labels=labels,
        smile_graph=smile_graph,
    )


# ==========================================================
# Create PyG Datasets
# ==========================================================

print("\nCreating Train Dataset...")
build_dataset(
    train_df,
    f"train_{SEED}"
)

print("\nCreating Validation Dataset...")
build_dataset(
    val_df,
    f"val_{SEED}"
)

print("\nCreating Test Dataset...")
build_dataset(
    test_df,
    f"test_{SEED}"
)

print("\n" + "=" * 60)
print(f"Finished Dataset {SEED}")
print("=" * 60)