"""
Model Loader — Load trained detectors from MLflow Model Registry.

Provides a cached loader so the API process loads each well's model once
at startup and keeps it in memory, rather than hitting the registry on
every telemetry reading.

Usage (in Kafka consumer or API route):
    from src.ml.model_loader import DetectorRegistry

    registry = DetectorRegistry()
    await registry.load_all()          # loads all Production-stage models at startup

    # Per reading:
    model = registry.get("15/9-F-12")
    scores = model.predict(telemetry_df)
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow.pyfunc
import pandas as pd

from src.ml.model_registry import (
    TELEMETRY_FEATURES,
    get_production_model_uri,
    list_all_model_versions,
)

logger = logging.getLogger(__name__)

KNOWN_WELLS = [
    "15/9-F-1 C",
    "15/9-F-11",
    "15/9-F-12",
    "15/9-F-14",
    "15/9-F-15 D",
    "15/9-F-5",
]


class DetectorRegistry:
    """
    In-memory cache of loaded MLflow pyfunc models, one per well.

    Loads the Production-stage model for each well at startup.
    Falls back to Staging if Production is not available.
    Provides thread-safe predict() interface.
    """

    def __init__(self, tracking_uri: str = "http://localhost:5001") -> None:
        self._tracking_uri = tracking_uri
        self._models: dict[str, mlflow.pyfunc.PyFuncModel] = {}

    def load_all(self, wells: list[str] | None = None) -> dict[str, str]:
        """
        Load Production (or Staging) model for each well.

        Returns dict of {well_id: model_uri} for loaded models.
        """
        targets = wells or KNOWN_WELLS
        loaded: dict[str, str] = {}

        for well_id in targets:
            uri = get_production_model_uri(well_id, self._tracking_uri)
            if uri is None:
                logger.warning("No registered model for %s — skipping", well_id)
                continue
            try:
                model = mlflow.pyfunc.load_model(uri)
                self._models[well_id] = model
                loaded[well_id] = uri
                logger.info("Loaded model for %s from %s", well_id, uri)
            except Exception as e:
                logger.error("Failed to load model for %s: %s", well_id, e)

        logger.info("DetectorRegistry loaded %d/%d models", len(loaded), len(targets))
        return loaded

    def get(self, well_id: str) -> mlflow.pyfunc.PyFuncModel | None:
        """Return the loaded model for a well, or None if not loaded."""
        return self._models.get(well_id)

    def predict(self, well_id: str, reading: dict[str, Any]) -> pd.DataFrame | None:
        """
        Score a single telemetry reading for a well.

        Returns a one-row DataFrame with anomaly_score, anomaly_type,
        severity, economic_impact_usd — or None if model not loaded.
        """
        model = self.get(well_id)
        if model is None:
            return None

        row = {f: float(reading.get(f, 0.0)) for f in TELEMETRY_FEATURES}
        input_df = pd.DataFrame([row])
        return model.predict(input_df)

    def reload(self, well_id: str) -> bool:
        """Hot-reload the model for a specific well (e.g. after rollback)."""
        uri = get_production_model_uri(well_id, self._tracking_uri)
        if uri is None:
            return False
        try:
            self._models[well_id] = mlflow.pyfunc.load_model(uri)
            logger.info("Hot-reloaded model for %s from %s", well_id, uri)
            return True
        except Exception as e:
            logger.error("Hot-reload failed for %s: %s", well_id, e)
            return False

    def summary(self) -> list[dict[str, Any]]:
        """Return registry status — loaded models and their URIs."""
        return list_all_model_versions(self._tracking_uri)
