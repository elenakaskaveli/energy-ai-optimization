import pytest

from src.evaluation.metrics import compare_models, evaluate, mae, mape, rmse, wape


def test_rmse_mae_perfect_prediction():
    y_true = [1, 2, 3, 4]
    assert rmse(y_true, y_true) == 0
    assert mae(y_true, y_true) == 0
    assert mape(y_true, y_true) == 0
    assert wape(y_true, y_true) == 0


def test_rmse_mae_known_error():
    y_true = [10, 10, 10, 10]
    y_pred = [12, 8, 12, 8]
    assert rmse(y_true, y_pred) == pytest.approx(2.0)
    assert mae(y_true, y_pred) == pytest.approx(2.0)
    assert mape(y_true, y_pred) == pytest.approx(20.0)
    assert wape(y_true, y_pred) == pytest.approx(20.0)


def test_wape_is_robust_to_near_zero_actuals():
    # A single near-zero actual would blow up MAPE, but WAPE stays sane
    # because it weights by total magnitude instead of per-point ratios.
    y_true = [0.01, 10, 10, 10]
    y_pred = [0.5, 10, 10, 10]
    assert mape(y_true, y_pred) > 1000  # dominated by the near-zero point
    assert wape(y_true, y_pred) == pytest.approx(0.49 / 30.01 * 100)


def test_evaluate_returns_all_metrics():
    result = evaluate([1, 2, 3], [1, 2, 4])
    assert set(result.keys()) == {"RMSE", "MAE", "MAPE", "WAPE"}


def test_compare_models_sorts_by_rmse():
    results = {
        "worse": {"RMSE": 5.0, "MAE": 4.0, "MAPE": 10.0},
        "better": {"RMSE": 1.0, "MAE": 0.8, "MAPE": 2.0},
    }
    table = compare_models(results)
    assert table.index[0] == "better"
    assert table.index[1] == "worse"
