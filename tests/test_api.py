import json

from fastapi.testclient import TestClient

import backend.main as api
from src.models.artifact import save_artifact
from tests.test_model_pipeline import artifact_bundle


def test_health_model_and_predict_endpoints(tmp_path, monkeypatch, prices):
    artifact = tmp_path / "model.pt"
    metadata = tmp_path / "model.metadata.json"
    bundle = artifact_bundle(prices)
    checksum = save_artifact(bundle, artifact)
    metadata.write_text(
        json.dumps(
            {
                "model_version": "test-v1",
                "deployment_status": "deployment_blocked",
                "checksum_sha256": checksum,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "ARTIFACT_PATH", artifact)
    monkeypatch.setattr(api, "METADATA_PATH", metadata)
    client = TestClient(api.app)

    assert client.get("/api/health").json()["status"] == "healthy"
    assert client.get("/api/model").json()["checksum_verified"] is True
    payload = prices.copy()
    payload["date"] = payload["date"].dt.strftime("%Y-%m-%d")
    response = client.post("/api/predict", json={"records": payload.to_dict("records")})
    assert response.status_code == 200
    assert response.json()["model_version"] == "test-v1"
