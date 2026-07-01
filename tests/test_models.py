import json
import pickle

import numpy as np
import pytest

from coughkit import models


class _PickleScaler:
    def transform(self, x):
        return x


def test_load_scaler_prefers_json_artifact(tmp_path, monkeypatch):
    scaler_json = tmp_path / "cough_classification_scaler.json"
    scaler_json.write_text(json.dumps({
        "mean_": [1.0, 2.0],
        "scale_": [2.0, 4.0],
        "var_": [4.0, 16.0],
        "with_mean": True,
        "with_std": True,
        "n_features": 2,
    }))
    legacy_scaler = tmp_path / "cough_classification_scaler"
    legacy_scaler.write_bytes(pickle.dumps(_PickleScaler()))

    monkeypatch.setattr(models, "SCALER_JSON_PATH", scaler_json)
    monkeypatch.setattr(models, "SCALER_PATH", legacy_scaler)

    scaler = models.load_scaler()

    assert isinstance(scaler, models.NumpyStandardScaler)
    np.testing.assert_allclose(
        scaler.transform(np.array([[3.0, 10.0]])),
        np.array([[1.0, 2.0]]),
    )


def test_numpy_standard_scaler_validates_feature_count():
    scaler = models.NumpyStandardScaler(
        mean=[1.0, 2.0],
        scale=[1.0, 1.0],
        n_features=2,
    )

    with pytest.raises(ValueError, match="Expected 2 features"):
        scaler.transform(np.array([[1.0]]))
