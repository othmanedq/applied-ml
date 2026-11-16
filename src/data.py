import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

TARGET = "subscribed"

COLUMN_NAMES = {
    "V1": "age", "V2": "job", "V3": "marital", "V4": "education", "V5": "default",
    "V6": "balance", "V7": "housing", "V8": "loan", "V9": "contact", "V10": "day",
    "V11": "month", "V12": "duration", "V13": "campaign", "V14": "pdays",
    "V15": "previous", "V16": "poutcome",
}


def load_raw(config):
    """All the columns of the dataset, including duration."""
    # The default pandas parser fails on this dataset with pandas 3, so we use liac-arff.
    bank = fetch_openml(data_id=config["data"]["openml_id"], as_frame=True, parser="liac-arff")
    data = bank.data.rename(columns=COLUMN_NAMES)
    categorical = data.select_dtypes("category").columns
    # Plain strings, the same type as the API input.
    data[categorical] = data[categorical].astype(str)
    # OpenML codes the target as "1" (no) and "2" (yes).
    data[TARGET] = (bank.target == "2").astype(int)
    return data


def load_data(config):
    # duration is the length of the call: it is only known after the call, so it is a leak.
    return load_raw(config).drop(columns="duration")


def split_data(data, config):
    """Stratified split into train, validation and test sets."""
    seed = config["seed"]
    test_size = config["data"]["test_size"]
    # validation_size is a share of the full data, so we rescale it for the second split.
    validation_share = config["data"]["validation_size"] / (1 - test_size)
    development, test = train_test_split(
        data, test_size=test_size, stratify=data[TARGET], random_state=seed
    )
    train, validation = train_test_split(
        development, test_size=validation_share, stratify=development[TARGET], random_state=seed
    )
    return {"train": train, "validation": validation, "test": test}


def features_and_target(data):
    return data.drop(columns=TARGET), data[TARGET]


def age_group(data, config):
    groups = config["age_groups"]
    return (
        pd.cut(data["age"], bins=groups["bins"], labels=groups["labels"])
        .astype(str)
        .rename("age_group")
    )
