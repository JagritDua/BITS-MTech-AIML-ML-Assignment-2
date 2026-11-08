"""Download the official UCI Dry Bean dataset and save it locally."""
from ucimlrepo import fetch_ucirepo
import pandas as pd


def load_uci_dry_bean():
    bean = fetch_ucirepo(id=602)
    X = bean.data.features.copy()
    y = bean.data.targets.copy()
    target = y.columns[0]
    data = X.copy()
    data[target] = y[target].astype(str)
    data.to_csv("dataset.csv", index=False)
    return data, target


if __name__ == "__main__":
    data, target = load_uci_dry_bean()
    print(f"Saved dataset.csv: {data.shape[0]} rows, {data.shape[1]-1} features; target={target}")
