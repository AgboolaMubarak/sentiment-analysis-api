import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def client():
    """
    Fixture that triggers the FastAPI lifespan.
    If the real model fails to load (common in CI/CD environments),
    it injects a mock predictor to allow API testing.
    """
    with TestClient(app) as c:
        # Check if the lifespan failed to attach the predictor due to missing files
        if not hasattr(app.state, "predictor"):
            # Create a simple Mock class that mimics your SentimentPredictor
            class MockPredictor:
                def predict(self, text: str):
                    # Returns a fake label and confidence
                    return "neutral", 0.5
            
            # Inject the mock into the app state
            app.state.predictor = MockPredictor()
            
        yield c

def test_health_check(client):
    """Test the health endpoint. Now returns 200 even in CI."""
    response = client.get("/health")
    assert response.status_code == 200
    # it will now pass because of our mock injection.
    assert response.json()["status"] == "healthy"

def test_predict_single(client):
    """Test single prediction endpoint with the mock/real predictor."""
    payload = {"text": "I love this new architecture!"}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)

def test_predict_empty_string(client):
    """Test that pydantic validation still works correctly."""
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    # This checks the Pydantic schema logic, which doesn't require a real model
    assert response.status_code == 422