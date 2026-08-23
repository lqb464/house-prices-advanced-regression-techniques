"""API serving model tối giản cho case study DS/MLE StocKast."""

from __future__ import annotations

import json
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import load_config, resolve_project_path
from src.models.artifact import sha256
from src.pipeline import predict_records

CONFIG = load_config()
ARTIFACT_PATH = resolve_project_path(CONFIG["artifact_path"])
METADATA_PATH = ARTIFACT_PATH.with_suffix(".metadata.json")

app = FastAPI(
    title="StocKast — API dự báo VIC",
    description="Framework nghiên cứu dự báo stock time-series với quality gate minh bạch.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class MarketRecord(BaseModel):
    date: str
    open: float | str | None
    high: float | str | None
    low: float | str | None
    close: float | str | None
    volume: float | str | None


class PredictionRequest(BaseModel):
    records: list[MarketRecord] = Field(min_length=70, max_length=5000)


def metadata() -> dict:
    if not METADATA_PATH.exists():
        raise HTTPException(503, "Model artifact chưa được train. Hãy chạy `make train`.")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@app.get("/api/health", tags=["system"])
def health() -> dict:
    available = ARTIFACT_PATH.exists() and METADATA_PATH.exists()
    return {
        "status": "healthy" if available else "degraded",
        "artifact_available": available,
        "deployment_status": (
            metadata().get("deployment_status") if available else "missing"
        ),
    }


@app.get("/api/model", tags=["model"])
def model_info() -> dict:
    info = metadata()
    info["checksum_verified"] = sha256(ARTIFACT_PATH) == info["checksum_sha256"]
    return info


@app.post("/api/predict", tags=["model"])
def predict(body: PredictionRequest) -> dict:
    try:
        return predict_records(
            pd.DataFrame([record.model_dump() for record in body.records]),
            ARTIFACT_PATH,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
