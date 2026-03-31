import json
import joblib
import os
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import shap
from evidently import Report
from evidently.presets import DataDriftPreset
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.generate_data import GENERATED_DATA_PATH, generate_and_save_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
REFERENCE_DATA_PATH = BASE_DIR / "data" / "sonar_data.csv"
REPORT_PATH = BASE_DIR / "report.html"

if not MODEL_PATH.exists():
    subprocess.run(["dvc", "pull"], cwd=BASE_DIR, check=False)

app = FastAPI(
    title="Sonar Classifier API",
    description="API de classification sonar avec monitoring Evidently",
    version="1.0.0",
)

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "https://python-community.com,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
EXPECTED_FEATURE_COUNT = int(getattr(scaler, "n_features_in_", 60))
FEATURE_NAMES = (
    list(scaler.feature_names_in_)
    if hasattr(scaler, "feature_names_in_")
    else [f"feature_{index}" for index in range(1, EXPECTED_FEATURE_COUNT + 1)]
)
CLASS_NAMES = list(getattr(model, "classes_", []))

explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")


class InputData(BaseModel):
    features: List[float]


def load_generated_data() -> dict:
    if not GENERATED_DATA_PATH.exists():
        return generate_and_save_data(EXPECTED_FEATURE_COUNT)

    with GENERATED_DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_features(features: List[float]) -> pd.DataFrame:
    if len(features) != EXPECTED_FEATURE_COUNT:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Le modèle attend {EXPECTED_FEATURE_COUNT} features, "
                f"mais {len(features)} ont été reçues."
            ),
        )
    try:
        array = np.array(features, dtype=float).reshape(1, -1)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Les features doivent être numériques."
        ) from exc
    return pd.DataFrame(array, columns=FEATURE_NAMES)


def extract_predicted_class_shap_values(
    raw_shap_values, class_index: int
) -> list[float]:
    if isinstance(raw_shap_values, list):
        if class_index >= len(raw_shap_values):
            raise HTTPException(status_code=500, detail="Sortie SHAP inattendue.")
        selected = raw_shap_values[class_index]
        return np.asarray(selected, dtype=float).reshape(-1).tolist()

    shap_array = np.asarray(raw_shap_values)

    if shap_array.ndim == 3 and shap_array.shape[-1] == max(len(CLASS_NAMES), 1):
        selected = shap_array[0, :, class_index]
    elif shap_array.ndim == 3 and shap_array.shape[0] == max(len(CLASS_NAMES), 1):
        selected = shap_array[class_index, 0, :]
    elif shap_array.ndim == 2:
        selected = shap_array[0]
    elif shap_array.ndim == 1:
        selected = shap_array
    else:
        raise HTTPException(
            status_code=500, detail="Impossible d'interpréter les valeurs SHAP."
        )

    return np.asarray(selected, dtype=float).reshape(-1).tolist()


def load_reference_features() -> pd.DataFrame:
    ref_df = pd.read_csv(REFERENCE_DATA_PATH)
    ref_df.columns = FEATURE_NAMES + ["target"]
    return ref_df[FEATURE_NAMES]


def prepare_uploaded_features(file: UploadFile) -> pd.DataFrame:
    current_df = pd.read_csv(file.file)

    if "target" in current_df.columns:
        current_df = current_df.drop(columns=["target"])

    if current_df.shape[1] != EXPECTED_FEATURE_COUNT:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Le fichier doit contenir {EXPECTED_FEATURE_COUNT} colonnes de features, "
                f"mais {current_df.shape[1]} ont été trouvées."
            ),
        )

    current_df.columns = FEATURE_NAMES
    return current_df


def get_class_index(label: str) -> int:
    if CLASS_NAMES and label in CLASS_NAMES:
        return CLASS_NAMES.index(label)
    return 0


def build_feature_importance_summary(
    feature_names: list[str], shap_values: list[float], limit: int = 3
) -> list[dict]:
    ranked_features = sorted(
        (
            {
                "feature": feature_name,
                "value": float(shap_value),
                "direction": "towards_prediction"
                if shap_value >= 0
                else "away_from_prediction",
            }
            for feature_name, shap_value in zip(feature_names, shap_values)
        ),
        key=lambda item: abs(item["value"]),
        reverse=True,
    )
    return ranked_features[:limit]


def predict_from_features(features: List[float], source: str) -> dict:
    data = normalize_features(features)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    prediction_label = str(prediction[0])
    class_index = get_class_index(prediction_label)
    probabilities = (
        model.predict_proba(data_scaled)[0]
        if hasattr(model, "predict_proba")
        else None
    )
    shap_values = extract_predicted_class_shap_values(
        explainer.shap_values(data_scaled), class_index
    )

    class_probabilities = (
        {
            str(label): float(probability)
            for label, probability in zip(CLASS_NAMES, probabilities)
        }
        if probabilities is not None and CLASS_NAMES
        else {}
    )

    return {
        "prediction": prediction.tolist(),
        "prediction_label": prediction_label,
        "predicted_probability": float(
            class_probabilities.get(prediction_label, 0.0)
        ),
        "class_probabilities": class_probabilities,
        "shap_values": shap_values,
        "source": source,
        "features": features,
        "feature_names": FEATURE_NAMES,
        "explained_class": prediction_label,
        "feature_importance_summary": build_feature_importance_summary(
            FEATURE_NAMES, shap_values
        ),
    }


@app.get("/")
def home():
    return {"message": "API Sonar classifier OK"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL_PATH.exists()}


@app.get("/generated-data")
def get_generated_data():
    payload = load_generated_data()
    return {
        "features": payload["features"],
        "count": len(payload["features"]),
        "source": str(GENERATED_DATA_PATH.relative_to(BASE_DIR)),
    }


@app.post("/generate-data")
def generate_data():
    payload = generate_and_save_data(EXPECTED_FEATURE_COUNT)
    return {
        "message": "Nouvelles données générées",
        "features": payload["features"],
        "count": len(payload["features"]),
        "source": str(GENERATED_DATA_PATH.relative_to(BASE_DIR)),
    }


@app.post("/predict")
def predict(data: InputData | None = None):
    if data is None:
        payload = load_generated_data()
        return predict_from_features(payload["features"], "data/data_gen.json")

    return predict_from_features(data.features, "request")


@app.post("/predict/generated")
def predict_generated():
    payload = load_generated_data()
    return predict_from_features(payload["features"], "data/data_gen.json")


@app.post("/predict_file")
def predict_file(file: UploadFile = File(...)):
    df = prepare_uploaded_features(file)
    data_scaled = scaler.transform(df)
    prediction = model.predict(data_scaled)
    prediction_label = str(prediction[0])
    class_index = get_class_index(prediction_label)
    shap_values = extract_predicted_class_shap_values(
        explainer.shap_values(data_scaled), class_index
    )
    return {
        "prediction": prediction.tolist(),
        "prediction_label": prediction_label,
        "shap_values": shap_values,
        "feature_names": FEATURE_NAMES,
        "explained_class": prediction_label,
        "feature_importance_summary": build_feature_importance_summary(
            FEATURE_NAMES, shap_values
        ),
    }


def monitoring(ref_df, current_df):
    report = Report([DataDriftPreset()])
    snapshot = report.run(current_df, ref_df)
    snapshot.save_html(str(REPORT_PATH))

    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Le rapport de monitoring n'a pas pu être écrit sur le disque.",
        )


@app.post("/monitor")
def monitor(file: UploadFile = File(...)):
    current_df = prepare_uploaded_features(file)
    ref_df = load_reference_features()
    monitoring(ref_df, current_df)
    return {
        "message": "Rapport généré",
        "report_path": str(REPORT_PATH.relative_to(BASE_DIR)),
    }


@app.post("/monitor/generated")
def monitor_generated():
    payload = load_generated_data()
    current_df = normalize_features(payload["features"])
    ref_df = load_reference_features()
    monitoring(ref_df, current_df)
    return {
        "message": "Rapport de monitoring généré à partir de data/data_gen.json",
        "report_path": str(REPORT_PATH.relative_to(BASE_DIR)),
        "report_url": "/monitor/report",
        "current_rows": len(current_df),
        "reference_rows": len(ref_df),
    }


@app.get("/monitor/report")
def get_monitor_report():
    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Aucun rapport de monitoring n'a encore été généré.",
        )

    return FileResponse(REPORT_PATH, media_type="text/html", filename="report.html")
