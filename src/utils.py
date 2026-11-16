import json

import joblib
import matplotlib.pyplot as plt

from .config import FIGURES, MODELS, OUTPUTS, TABLES


def _prepare(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, name):
    fig.savefig(_prepare(FIGURES / f"{name}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_table(table, name, index=True):
    table.to_csv(_prepare(TABLES / f"{name}.csv"), index=index)


def save_json(data, name):
    with open(_prepare(OUTPUTS / f"{name}.json"), "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=lambda value: value.item())


def load_json(name):
    with open(OUTPUTS / f"{name}.json", encoding="utf-8") as file:
        return json.load(file)


def save_model(model, name):
    joblib.dump(model, _prepare(MODELS / f"{name}.joblib"))


def load_model(name):
    return joblib.load(MODELS / f"{name}.joblib")
