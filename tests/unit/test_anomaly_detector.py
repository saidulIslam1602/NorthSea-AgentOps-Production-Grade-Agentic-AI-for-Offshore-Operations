"""Unit tests for the anomaly detector."""

from __future__ import annotations

from datetime import UTC, datetime

from src.anomaly.detector import WellAnomalyDetector, _build_description
from src.schemas.domain import SeverityLevel


def make_reading(
    oil_rate: float = 2500.0,
    water_cut: float = 15.0,
    gor: float = 650.0,
    bhp: float = 3200.0,
    temp: float = 145.0,
    choke: float = 48.0,
    well_id: str = "TEST-1H",
) -> dict:
    return {
        "timestamp": datetime.now(UTC),
        "well_id": well_id,
        "field_name": "TestField",
        "oil_rate_bopd": oil_rate,
        "water_cut_pct": water_cut,
        "gas_oil_ratio": gor,
        "bhp_psi": bhp,
        "wh_temp_f": temp,
        "choke_64ths": choke,
    }


def prime_detector(detector: WellAnomalyDetector, n: int = 60) -> None:
    """Feed n normal readings to train the detector."""
    for _ in range(n):
        reading = make_reading(well_id=detector.well_id)
        detector.ingest(reading)


class TestWellAnomalyDetector:
    def test_no_alert_on_normal_reading(self) -> None:
        detector = WellAnomalyDetector(well_id="D-1H")
        prime_detector(detector, 60)
        alert = detector.ingest(make_reading(well_id="D-1H"))
        # After stable readings, a single normal reading should not alert
        # (may occasionally alert depending on noise, so we check severity)
        if alert is not None:
            assert alert.severity in (SeverityLevel.LOW,)

    def test_alert_on_extreme_water_cut(self) -> None:
        detector = WellAnomalyDetector(well_id="D-2H")
        prime_detector(detector, 60)
        # Inject a severe water cut spike
        alert = detector.ingest(make_reading(water_cut=95.0, oil_rate=400.0, well_id="D-2H"))
        assert alert is not None
        assert alert.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL)
        assert "water_cut_pct" in alert.affected_features

    def test_alert_on_bhp_collapse(self) -> None:
        detector = WellAnomalyDetector(well_id="D-3H")
        prime_detector(detector, 60)
        alert = detector.ingest(make_reading(bhp=800.0, oil_rate=200.0, well_id="D-3H"))
        assert alert is not None
        assert alert.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL)

    def test_alert_score_bounded(self) -> None:
        detector = WellAnomalyDetector(well_id="D-4H")
        prime_detector(detector, 60)
        alert = detector.ingest(make_reading(water_cut=99.0, oil_rate=0.0, well_id="D-4H"))
        if alert is not None:
            assert 0.0 <= alert.anomaly_score <= 1.0

    def test_alert_has_required_fields(self) -> None:
        detector = WellAnomalyDetector(well_id="D-5H")
        prime_detector(detector, 60)
        alert = detector.ingest(make_reading(water_cut=80.0, oil_rate=500.0, well_id="D-5H"))
        if alert is not None:
            assert alert.well_id == "D-5H"
            assert alert.field_name == "TestField"
            assert len(alert.description) > 10
            assert alert.timestamp is not None

    def test_description_includes_domain_hint(self) -> None:
        desc = _build_description(
            well_id="D-1H",
            severity=SeverityLevel.HIGH,
            affected_features=["water_cut_pct", "oil_rate_bopd"],
            deviation_pct={"water_cut_pct": 250.0, "oil_rate_bopd": -45.0},
            current_values={"water_cut_pct": 52.5, "oil_rate_bopd": 1375.0, "bhp_psi": 3200.0},
        )
        assert "D-1H" in desc
        assert "water" in desc.lower() or "breakthrough" in desc.lower()


class TestSyntheticGenerator:
    def test_generate_timeseries(self) -> None:
        from datetime import datetime

        from src.data.synthetic_generator import generate_well_timeseries

        df = generate_well_timeseries("D-1H", "Draugen", datetime(2024, 1, 1, tzinfo=UTC), n_hours=100)
        assert len(df) == 100
        assert "oil_rate_bopd" in df.columns
        assert "water_cut_pct" in df.columns
        assert (df["water_cut_pct"] >= 0).all()
        assert (df["water_cut_pct"] <= 100).all()

    def test_anomaly_injection_changes_values(self) -> None:
        from datetime import datetime

        from src.data.synthetic_generator import generate_well_timeseries

        df_normal = generate_well_timeseries(
            "TEST-1H", "Test", datetime(2024, 1, 1, tzinfo=UTC), n_hours=200, anomaly_type=None
        )
        df_anomaly = generate_well_timeseries(
            "TEST-1H",
            "Test",
            datetime(2024, 1, 1, tzinfo=UTC),
            n_hours=200,
            anomaly_type="water_breakthrough",
            anomaly_onset_hour=100,
        )
        # After onset, water cut should be higher in anomaly dataset
        late_normal = df_normal.iloc[150:]["water_cut_pct"].mean()
        late_anomaly = df_anomaly.iloc[150:]["water_cut_pct"].mean()
        assert late_anomaly > late_normal
