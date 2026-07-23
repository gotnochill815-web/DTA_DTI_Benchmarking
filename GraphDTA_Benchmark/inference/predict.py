import os
import argparse

import torch
import numpy as np
import pandas as pd

from models.gcn import GCNNet
from training.config import DEVICE, DATA_ROOT, BATCH_SIZE
from training.dataloader import get_dataloaders


def load_model(model_path):

    model = GCNNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    model.eval()

    return model


@torch.no_grad()
def predict(model, loader):

    y_true = []
    y_pred = []

    model.eval()

    for batch in loader:

        batch = batch.to(DEVICE)

        outputs = model(batch)

        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(outputs.squeeze(1).cpu().numpy())

    return np.array(y_true), np.array(y_pred)


def main():

    # --------------------------------------------------
    # Arguments
    # --------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Dataset seed (32,42,52,62,72)",
    )

    args = parser.parse_args()

    dataset_seed = args.seed

    print("=" * 60)
    print(f"Predicting Seed {dataset_seed}")
    print("=" * 60)

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    OUTPUT_DIR = os.path.join(
        "outputs",
        f"seed_{dataset_seed}",
    )

    CKPT_DIR = os.path.join(
        OUTPUT_DIR,
        "checkpoints",
    )

    PRED_DIR = os.path.join(
        OUTPUT_DIR,
        "predictions",
    )

    os.makedirs(PRED_DIR, exist_ok=True)

    # --------------------------------------------------
    # Load Data
    # --------------------------------------------------

    _, _, test_loader = get_dataloaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        dataset_seed=dataset_seed,
    )

    print("Test Samples :", len(test_loader.dataset))

    # --------------------------------------------------
    # Load Model
    # --------------------------------------------------

    model = load_model(
        os.path.join(
            CKPT_DIR,
            "best_graphdta.pt",
        )
    )

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------

    y_true, y_pred = predict(
        model,
        test_loader,
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    np.save(
        os.path.join(
            PRED_DIR,
            "y_true.npy",
        ),
        y_true,
    )

    np.save(
        os.path.join(
            PRED_DIR,
            "y_pred.npy",
        ),
        y_pred,
    )

    pd.DataFrame(
        {
            "Actual": y_true,
            "Predicted": y_pred,
        }
    ).to_csv(
        os.path.join(
            PRED_DIR,
            "test_predictions.csv",
        ),
        index=False,
    )

    print("\nSaved:")
    print(os.path.join(PRED_DIR, "y_true.npy"))
    print(os.path.join(PRED_DIR, "y_pred.npy"))
    print(os.path.join(PRED_DIR, "test_predictions.csv"))


if __name__ == "__main__":
    main()