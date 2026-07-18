import math

from scripts.smoke_train import run_smoke


def test_cpu_smoke_training_reduces_loss():
    result = run_smoke(seed=42, steps=5)

    assert result["device"] == "cpu"
    assert result["steps"] == 5
    assert math.isfinite(result["initial_loss"])
    assert math.isfinite(result["final_loss"])
    assert result["final_loss"] < result["initial_loss"]
