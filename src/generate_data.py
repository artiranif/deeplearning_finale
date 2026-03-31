import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
GENERATED_DATA_PATH = BASE_DIR / "data" / "data_gen.json"
DEFAULT_FEATURE_COUNT = 60


def generate_random_features(feature_count: int = DEFAULT_FEATURE_COUNT) -> list[float]:
    return [round(random.uniform(0.0001, 0.9), 6) for _ in range(feature_count)]


def save_generated_data(
    features: list[float], output_path: Path = GENERATED_DATA_PATH
) -> dict:
    payload = {"features": features}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)

    return payload


def generate_and_save_data(feature_count: int = DEFAULT_FEATURE_COUNT) -> dict:
    return save_generated_data(generate_random_features(feature_count))


if __name__ == "__main__":
    generate_and_save_data()
