import pytest
from fastapi.testclient import TestClient
from src.api.main import app

# wrapping the client in a fixture to ensure lifespan runs for every test
@pytest.fixture
def client():
    """
    Fixture that triggers the FastAPI lifespan (startup/shutdown) 
    before running tests.
    """
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """Test the health endpoint returns 200 and correct status."""
    response = client.get("/health")
    # If the model isn't loaded, main.py raises a 503. 
    # This test ensures it is actually healthy.
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_single(client):
    """Test single prediction endpoint."""
    payload = {"text": "I love this new architecture!"}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)

def test_predict_empty_string(client):
    """Test that pydantic validation catches empty strings."""
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422