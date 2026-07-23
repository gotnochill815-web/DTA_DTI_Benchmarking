import os
import json
import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)

from scipy.stats import (
    spearmanr,
)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def ci(y, f):

    ind = np.argsort(y)

    y = y[ind]
    f = f[ind]

    i = len(y) - 1
    j = i - 1

    z = 0.0
    S = 0.0

    while i > 0:

        while j >= 0:

            if y[i] > y[j]:

                z += 1

                u = f[i] - f[j]

                if u > 0:
                    S += 1

                elif u == 0:
                    S += 0.5

            j -= 1

        i -= 1
        j = i - 1

    return S / z


# --------------------------------------------------
# Bootstrap
# --------------------------------------------------

def bootstrap_metrics(
    y_true,
    y_pred,
    n_bootstrap=1000,
    seed=32,
):

    rng = np.random.default_rng(seed)

    n = len(y_true)

    history = []

    for i in range(n_bootstrap):

        if (i + 1) % 100 == 0:
            print(f"{i+1}/{n_bootstrap}")

        idx = rng.integers(0, n, n)

        yt = y_true[idx]
        yp = y_pred[idx]

        history.append({

            "Iteration": i + 1,

            "RMSE": np.sqrt(mean_squared_error(yt, yp)),

            "MSE": mean_squared_error(yt, yp),

            "R2": r2_score(yt, yp),

            "Pearson": np.corrcoef(yt, yp)[0, 1],

        })

    return pd.DataFrame(history)


# --------------------------------------------------
# Summary
# --------------------------------------------------

def summarize(df):

    summary = {}

    for col in df.columns[1:]:

        summary[col] = {

            "Mean": float(df[col].mean()),

            "Std": float(df[col].std()),

            "Lower95": float(df[col].quantile(0.025)),

            "Upper95": float(df[col].quantile(0.975)),
        }

    return summary


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    dataset_seed = args.seed

    OUTPUT_DIR = os.path.join(
        "outputs",
        f"seed_{dataset_seed}",
    )

    PRED_DIR = os.path.join(
        OUTPUT_DIR,
        "predictions",
    )

    RESULT_DIR = os.path.join(
        OUTPUT_DIR,
        "results",
    )

    os.makedirs(RESULT_DIR, exist_ok=True)

    y_true = np.load(
        os.path.join(
            PRED_DIR,
            "y_true.npy",
        )
    )

    y_pred = np.load(
        os.path.join(
            PRED_DIR,
            "y_pred.npy",
        )
    )

    print("Running Bootstrap...")

    results = bootstrap_metrics(
        y_true,
        y_pred,
        n_bootstrap=1000,
        seed=dataset_seed,
    )

    results.to_csv(
        os.path.join(
            RESULT_DIR,
            "bootstrap_results.csv",
        ),
        index=False,
    )

    summary = summarize(results)

    summary["Spearman"] = float(
        spearmanr(y_true, y_pred)[0]
    )

    summary["CI"] = float(
        ci(y_true, y_pred)
    )

    with open(
        os.path.join(
            RESULT_DIR,
            "bootstrap_summary.json",
        ),
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    print("\nSummary\n")

    for k, v in summary.items():

        print(k, ":", v)

    print("\nSaved to")

    print(RESULT_DIR)


if __name__ == "__main__":
    main()