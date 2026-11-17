from functools import lru_cache
from typing import Annotated, Literal

import pandas as pd
from fastapi import Body, FastAPI
from pydantic import BaseModel, Field

from .utils import load_model

app = FastAPI(title="Term deposit subscription model")

YesNo = Literal["yes", "no"]


class Client(BaseModel):
    """One client, with the columns of the training data. Unknown categories are rejected."""

    age: int = Field(ge=18)
    job: Literal[
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
        "self-employed", "services", "student", "technician", "unemployed", "unknown",
    ]
    marital: Literal["divorced", "married", "single"]
    education: Literal["primary", "secondary", "tertiary", "unknown"]
    default: YesNo
    balance: float
    housing: YesNo
    loan: YesNo
    contact: Literal["cellular", "telephone", "unknown"]
    day: int = Field(ge=1, le=31)
    month: Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    campaign: int = Field(ge=1)
    pdays: int = Field(ge=-1)
    previous: int = Field(ge=0)
    poutcome: Literal["failure", "other", "success", "unknown"]


@lru_cache
def final_model():
    return load_model("final_model")


@app.get("/health")
def health():
    model = final_model()
    return {"status": "ok", "model": model["name"], "threshold": model["threshold"]}


@app.post("/predict")
def predict(clients: Annotated[list[Client], Body(min_length=1)]):
    model = final_model()
    X = pd.DataFrame([client.model_dump() for client in clients])
    proba = model["pipeline"].predict_proba(X)[:, 1]
    return [{"probability": float(p), "call": bool(p >= model["threshold"])} for p in proba]
