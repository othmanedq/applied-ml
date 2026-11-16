from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = ROOT / "models"


def load_config(path=ROOT / "config.yaml"):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)
