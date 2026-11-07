
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import app


def test_health_check():
    client = app.test_client()

    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "ok"
    assert "message" in data
