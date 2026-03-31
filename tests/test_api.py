from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def mock_model_loading(tmp_path):
    """Mock les chargements de modèle pour les tests sans fichiers réels."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # Créer un modèle factice
    mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_dummy = np.random.rand(20, 60)
    y_dummy = np.array(["M", "R"] * 10)
    mock_model.fit(X_dummy, y_dummy)

    # Créer un scaler factice
    mock_scaler = StandardScaler()
    mock_scaler.fit(X_dummy)

    with patch("joblib.load") as mock_load:
        def side_effect(path):
            path_str = str(path)
            if "model.pkl" in path_str:
                return mock_model
            elif "scaler.pkl" in path_str:
                return mock_scaler
            return MagicMock()

        mock_load.side_effect = side_effect

        with patch("app.main.MODEL_PATH") as mock_model_path:
            mock_model_path.exists.return_value = True

            # Recharger le module pour appliquer les mocks
            import importlib
            import app.main
            importlib.reload(app.main)

            yield


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Sonar" in data["message"] or "OK" in data["message"]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predict_with_features(client):
    features = [0.5] * 60
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert "prediction_label" in data
    assert data["prediction_label"] in ["M", "R"]
    assert "shap_values" in data
    assert "class_probabilities" in data


def test_predict_wrong_feature_count(client):
    features = [0.5] * 30  # Mauvais nombre
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 400


def test_generate_data(client):
    response = client.post("/generate-data")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert data["count"] == 60
