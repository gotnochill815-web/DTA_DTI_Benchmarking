import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from training.config import (
    DEVICE,
    DATA_ROOT,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    seed_everything,
)

from training.dataloader import get_dataloaders
from training.early_stopping import EarlyStopping
from training.engine import train_one_epoch, validate

from models.gcn import GCNNet


def main():

    # -------------------------------------------------------
    # Arguments
    # -------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=32,
        help="Dataset seed (32,42,52,62,72)",
    )

    args = parser.parse_args()

    dataset_seed = args.seed

    # -------------------------------------------------------
    # Seed
    # -------------------------------------------------------
    seed_everything(dataset_seed)

    print(f"Using device: {DEVICE}")
    print(f"Dataset Seed: {dataset_seed}")

    # -------------------------------------------------------
    # Output folders
    # -------------------------------------------------------
    OUTPUT_DIR = os.path.join("outputs", f"seed_{dataset_seed}")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # -------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        dataset_seed=dataset_seed,
    )

    print(f"Train Samples : {len(train_loader.dataset)}")
    print(f"Val Samples   : {len(val_loader.dataset)}")
    print(f"Test Samples  : {len(test_loader.dataset)}")

    # -------------------------------------------------------
    # Model
    # -------------------------------------------------------
    model = GCNNet().to(DEVICE)

    print(model)

    # -------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -------------------------------------------------------
    # Loss
    # -------------------------------------------------------
    criterion = nn.MSELoss()

    # -------------------------------------------------------
    # Early stopping
    # -------------------------------------------------------
    early_stopping = EarlyStopping(
        patience=10,
        min_delta=0.001,
        mode="min",
        save_path=os.path.join(
            CKPT_DIR,
            "best_graphdta.pt",
        ),
    )

    # -------------------------------------------------------
    # Logs
    # -------------------------------------------------------
    logs = []

    # -------------------------------------------------------
    # Training
    # -------------------------------------------------------
    for epoch in range(100):

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
        )

        val_rmse = validate(
            model=model,
            loader=val_loader,
            device=DEVICE,
        )

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val RMSE: {val_rmse:.4f}"
        )

        logs.append({

            "epoch": epoch + 1,

            "train_loss": float(train_loss),

            "val_rmse": float(val_rmse),

            "best_rmse":
                float(
                    early_stopping.best_score
                    if early_stopping.best_score is not None
                    else val_rmse
                ),
        })

        early_stopping(val_rmse, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("\nTraining Complete.")

    # -------------------------------------------------------
    # Save Logs
    # -------------------------------------------------------
    log_df = pd.DataFrame(logs)

    log_path = os.path.join(
        LOG_DIR,
        "training_logs.csv",
    )

    log_df.to_csv(
        log_path,
        index=False,
    )

    print(f"Training logs saved to {log_path}")


if __name__ == "__main__":
    main()