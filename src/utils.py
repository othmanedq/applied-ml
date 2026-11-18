import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = ROOT / "models"


def load_config(path=ROOT / "config.yaml"):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def _prepare(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def figure_path(name):
    return FIGURES / f"{name}.png"


def model_path(name):
    return MODELS / f"{name}.joblib"


def save_figure(fig, name):
    fig.savefig(_prepare(figure_path(name)), dpi=130, bbox_inches="tight")
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
    joblib.dump(model, _prepare(model_path(name)))


def load_model(name):
    return joblib.load(model_path(name))
