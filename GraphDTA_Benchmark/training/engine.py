import torch
import numpy as np

from math import sqrt


# ==========================================================
# Train
# ==========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    running_loss = 0.0

    for batch in loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        predictions = model(batch)
        

        loss = criterion(
            predictions.view(-1),
            batch.y.view(-1)
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(loader)

    return epoch_loss


# ==========================================================
# Validation
# ==========================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
):

    model.eval()

    predictions = []

    labels = []

    for batch in loader:

        batch = batch.to(device)

        outputs = model(batch)

        predictions.extend(
            outputs.view(-1).cpu().numpy()
        )

        labels.extend(
            batch.y.view(-1).cpu().numpy()
        )

    predictions = np.array(predictions)
    labels = np.array(labels)

    rmse = sqrt(
        np.mean(
            (labels - predictions) ** 2
        )
    )

    return rmse


# ==========================================================
# Prediction
# ==========================================================

@torch.no_grad()
def predict(
    model,
    loader,
    device,
):

    model.eval()

    predictions = []

    labels = []

    for batch in loader:

        batch = batch.to(device)

        outputs = model(batch)

        predictions.extend(
            outputs.view(-1).cpu().numpy()
        )

        labels.extend(
            batch.y.view(-1).cpu().numpy()
        )

    predictions = np.array(predictions)
    labels = np.array(labels)

    return labels, predictions