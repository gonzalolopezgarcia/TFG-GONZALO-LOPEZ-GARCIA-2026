from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def describe_scale(name, x):
    x = np.asarray(x, dtype=float)
    print(f"\n{name}")
    print("min:", np.nanmin(x))
    print("max:", np.nanmax(x))
    print("mean:", np.nanmean(x))
    print("p95:", np.nanpercentile(x, 95))
    print("p99:", np.nanpercentile(x, 99))
    print("p99.9:", np.nanpercentile(x, 99.9))


def read_csv(name):
    return pd.read_csv(DATA / name, index_col=0, parse_dates=True)


series = {
    "MLP-VaR w=20": (read_csv("nn_var_predictions_20.csv"), "VaR_pred"),
    "GARCH-t": (read_csv("garch_t_var_predictions.csv"), "VaR_GARCH"),
    "GARCH-Normal": (read_csv("garch_n_var_predictions.csv"), "VaR_GARCH_n"),
    "HS": (read_csv("hs_var_predictions.csv"), "VaR_HS"),
    "CAViaR-AS": (read_csv("caviar_var_predictions.csv"), "VaR_CAViaR"),
}

idx = None
for df, _ in series.values():
    idx = df.index if idx is None else idx.intersection(df.index)

print(f"Common evaluation dates: {idx.min().date()} to {idx.max().date()} | n={len(idx)}")

base_loss = series["MLP-VaR w=20"][0].loc[idx, "loss_real"]
describe_scale("realized losses | common sample", base_loss)

for name, (df, col) in series.items():
    aligned = df.loc[idx]
    loss_delta = np.nanmax(np.abs(aligned["loss_real"].to_numpy() - base_loss.to_numpy()))
    assert loss_delta < 1e-8, f"{name}: loss_real differs from MLP loss_real, max delta={loss_delta}"
    describe_scale(f"{name} VaR | common sample", aligned[col])

dataset = pd.read_pickle(DATA / "dataset_tfg.pkl").sort_index()
eval_dataset = dataset.loc[idx]
assert np.allclose(eval_dataset["target"].to_numpy(), base_loss.to_numpy(), atol=1e-6)

describe_scale("dataset target | common sample", eval_dataset["target"])
describe_scale("dataset rp_lag_0 | common sample", eval_dataset["rp_lag_0"])

mlp = series["MLP-VaR w=20"][0].loc[idx]
covid_mlp = mlp.loc["2020-01-01":"2020-12-31"]
print("\nMLP-VaR w=20 largest COVID predictions")
print(covid_mlp.nlargest(10, "VaR_pred")[["VaR_pred", "loss_real", "viol", "year"]])

print("\nMLP-VaR w=20 mean VaR check")
print("mean VaR:", mlp["VaR_pred"].mean())
print("mean VaR if interpreted as percent and converted to fraction:", mlp["VaR_pred"].mean() / 100.0)
