"""Audit the committed experiments without fitting or changing any models.

The script checks saved model predictions, nested-fold coverage, grouped-fold
isolation for var2, search-record accounting, pruning bounds, and file hashes.
It writes a fresh completion_verification.json only after all checks pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9)


def check_search(directory, maximum, folds=3):
    rows = []
    for degree in range(1, maximum + 1):
        current = load(directory / f"degree_{degree:02d}.json")
        assert all(row["degree"] == degree for row in current)
        assert {row["model"] for row in current} == {"Ridge", "LASSO", "ElasticNet"}
        for row in current:
            if row.get("pruned", False):
                assert row["mean_mse"] is None
                assert len(row["fold_mse"]) < folds
                close(row["mse_lower_bound"], sum(row["fold_mse"]) / folds)
            else:
                assert len(row["fold_mse"]) == folds
                close(row["mean_mse"], np.mean(row["fold_mse"]))
        rows.extend(current)
    for name in ("convergence_retries", "refine_Ridge", "refine_LASSO", "refine_ElasticNet"):
        rows.extend(load(directory / f"{name}.json"))
    eligible = [row for row in rows if row["converged"] and not row.get("pruned", False)]
    winner = load(directory / "winner.json")
    assert winner["converged"] and not winner.get("pruned", False)
    assert winner["max_kkt"] <= 5e-5
    close(winner["mean_mse"], min(row["mean_mse"] for row in eligible))
    table = pd.read_csv(directory / "search_results.csv")
    assert len(table) == len(rows)
    return {
        "candidate_rows": len(rows),
        "fully_evaluated_converged_rows": len(eligible),
        "pruned_rows": sum(bool(row.get("pruned", False)) for row in rows),
        "fully_evaluated_unconverged_rows": sum(
            not row["converged"] and not row.get("pruned", False) for row in rows
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output", type=Path, default=Path("completion_verification.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    results = root / "results"
    before = {}
    for path in [root / "README.md", root / "train_polynomial.py", root / "predict.py", root / "requirements.txt", root / "verify_results.py"]:
        before[str(path.relative_to(root))] = digest(path)
    for path in sorted(results.rglob("*")):
        if path.is_file() and path.name != "completion_verification.json":
            before[str(path.relative_to(root))] = digest(path)

    summaries = load(results / "summary.json")
    assert {summary["variant"] for summary in summaries} == {1, 2}
    report = {"status": "passed", "script": "verify_results.py", "no_models_retrained": True, "variants": []}

    for variant, maximum in ((1, 10), (2, 20)):
        train = pd.read_csv(args.data_dir / f"BT2024175_train_var{variant}.csv")
        test = pd.read_csv(args.data_dir / f"BT2024175_test_var{variant}.csv")
        X = train.drop(columns="y").to_numpy(float)
        y = train["y"].to_numpy(float)
        summary = load(results / f"var{variant}" / "summary.json")
        assert summary == next(item for item in summaries if item["variant"] == variant)
        model_dir = results / f"var{variant}"
        model = joblib.load(model_dir / "best_model.joblib")
        prediction_file = results / f"BT2024175_pred_var{variant}.csv"
        prediction = pd.read_csv(prediction_file)
        assert list(prediction.columns) == ["y"] and len(prediction) == len(test)
        close(model.predict(test.to_numpy(float)), prediction["y"].to_numpy())
        coef = model[-1].coef_
        assert np.isfinite(coef).all()
        stage_checks = {"final_search": check_search(model_dir / "final_search", maximum)}
        oof = pd.read_csv(model_dir / "outer_predictions.csv")
        assert len(oof) == len(train) and set(oof.columns) == {"y_true", "outer_cv_prediction"}
        close(oof["y_true"], y)
        groups = np.unique(X, axis=0, return_inverse=True)[1] if variant == 2 else None
        splitter = (GroupKFold(5, shuffle=True, random_state=176) if groups is not None
                    else KFold(5, shuffle=True, random_state=176))
        coverage = np.zeros(len(y), dtype=int)
        fold_mse, fold_r2 = [], []
        for number, (tr, va) in enumerate(splitter.split(X, y, groups), 1):
            stage = model_dir / f"outer_{number}"
            evaluation = load(stage / "outer_evaluation.json")
            stage_checks[f"outer_{number}"] = check_search(stage, maximum)
            coverage[va] += 1
            if groups is not None:
                assert not set(groups[tr]) & set(groups[va])
                for inner_tr, inner_va in GroupKFold(3, shuffle=True, random_state=175).split(X[tr], y[tr], groups[tr]):
                    assert not set(groups[tr][inner_tr]) & set(groups[tr][inner_va])
            mse = mean_squared_error(y[va], oof["outer_cv_prediction"].to_numpy()[va])
            r2 = r2_score(y[va], oof["outer_cv_prediction"].to_numpy()[va])
            close(mse, evaluation["mse"])
            close(r2, evaluation["r2"])
            fold_mse.append(mse)
            fold_r2.append(r2)
        assert np.all(coverage == 1)
        close(np.mean(fold_mse), summary["outer_mean_mse"])
        close(np.std(fold_mse, ddof=1), summary["outer_std_mse"])
        close(np.mean(fold_r2), summary["outer_mean_r2"])
        close(np.std(fold_r2, ddof=1), summary["outer_std_r2"])
        report["variants"].append({"variant": variant, "training_rows": len(train), "test_rows": len(test),
                                    "outer_folds_verified": 5, "all_training_rows_evaluated_once": True,
                                    "polynomial_terms": len(coef), "nonzero_coefficients": int(np.count_nonzero(coef)),
                                    "searches": stage_checks})

    report["tracked_file_sha256"] = before
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verification passed for both variants; wrote {args.output}")


if __name__ == "__main__":
    main()
