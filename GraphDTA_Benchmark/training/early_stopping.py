import os
import torch


class EarlyStopping:
    """
    Early stops training if validation metric doesn't improve.

    Parameters
    ----------
    patience : int
        Number of validation epochs to wait before stopping.

    min_delta : float
        Minimum improvement required to reset patience.

    mode : str
        "min" for RMSE/Loss
        "max" for Accuracy/AUC

    save_path : str
        Path to save the best model.

    verbose : bool
        Print improvement messages.
    """

    def __init__(
        self,
        patience=5,
        min_delta=0.001,
        mode="min",
        save_path="checkpoints/best_graphdta.pt",
        verbose=True,
    ):

        assert mode in ["min", "max"]

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.early_stop = False

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def __call__(self, metric, model):

        if self.best_score is None:

            self.best_score = metric
            self.save_checkpoint(metric, model)
            return

        if self.mode == "min":
            improved = metric < (self.best_score - self.min_delta)
        else:
            improved = metric > (self.best_score + self.min_delta)

        if improved:

            self.best_score = metric
            self.counter = 0

            self.save_checkpoint(metric, model)

        else:

            self.counter += 1

            if self.verbose:
                print(
                    f"No improvement "
                    f"({self.counter}/{self.patience})"
                )

            if self.counter >= self.patience:

                self.early_stop = True

                if self.verbose:
                    print("Early stopping triggered.")

    def save_checkpoint(self, metric, model):
        torch.save(model.state_dict(), self.save_path)

        if self.verbose:
            if self.mode == "min":
                print(f"Validation RMSE improved to {metric:.4f}")
            else:
                print(f"Validation metric improved to {metric:.4f}")

            print(f"Saved model -> {self.save_path}")