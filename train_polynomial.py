"""Polynomial regression for BT2024175; run --help for usage.

Every CV training fold fits its own two scalers. Model family, total degree,
alpha and Elastic Net ratio are selected jointly by mean validation MSE.
The independent outer loop evaluates the entire selection procedure.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import skglm
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

SEED = 175
RATIOS = (0.1, 0.3, 0.5, 0.7, 0.9, 0.95)
BASE_ALPHAS = np.logspace(-6, 6, 13)
TOL = 1e-5
LOG = logging.getLogger("polynomial")
POOL = None


def initialize_worker():
    threadpool_limits(limits=2)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def degree_job(X, y, cv, degree, candidates, checkpoint):
    start = time.monotonic()
    rows = evaluate_degree(X, y, cv, degree, candidates)
    json_save(checkpoint, rows)
    return rows, time.monotonic() - start


def json_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def preprocess(degree):
    return Pipeline([
        ("input_scaler", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
        ("term_scaler", StandardScaler()),
    ])


def splits(X, y, groups, count, seed):
    if groups is None:
        return list(KFold(count, shuffle=True, random_state=seed).split(X, y))
    return list(GroupKFold(count, shuffle=True, random_state=seed).split(X, y, groups))


def sparse_estimator(family, alpha, ratio, strict=False, thorough=False):
    common = dict(alpha=float(alpha), fit_intercept=False, warm_start=True,
                  tol=TOL / 10 if strict else TOL,
                  max_iter=100 if strict else (50 if thorough else 10),
                  max_epochs=50000 if strict else (10000 if thorough else 500))
    return skglm.Lasso(**common) if family == "LASSO" else skglm.ElasticNet(l1_ratio=ratio, **common)


def kkt_error(A, centered_y, coef, alpha, ratio):
    """Independent first-order optimality check for the L1/L2 objective."""
    gradient = A.T @ (A @ coef - centered_y) / len(centered_y)
    gradient += alpha * (1 - ratio) * coef
    active = coef != 0
    errors = np.maximum(np.abs(gradient) - alpha * ratio, 0)
    errors[active] = np.abs(gradient[active] + alpha * ratio * np.sign(coef[active]))
    return float(np.max(errors, initial=0))


def ridge_predictions(A, B, y, alphas):
    """One eigendecomposition for all Ridge alphas; matches sklearn Ridge."""
    yc = y - y.mean()
    if A.shape[1] <= A.shape[0]:
        values, vectors = eigh(A.T @ A, check_finite=False)
        values = np.maximum(values, 0)
        rhs = vectors.T @ (A.T @ yc)
        return (B @ vectors) @ (rhs[:, None] / (values[:, None] + alphas)) + y.mean()
    values, vectors = eigh(A @ A.T, check_finite=False)
    values = np.maximum(values, 0)
    rhs = vectors.T @ yc
    return ((B @ A.T) @ vectors) @ (rhs[:, None] / (values[:, None] + alphas)) + y.mean()


def evaluate_degree(X, y, cv, degree, candidates, thorough=False, prune=True):
    """Cache transformations in memory within each fold; never across folds."""
    accum = {tuple(c): {"mse": [], "r2": [], "kkt": [], "warnings": 0} for c in candidates}
    ridge_alphas = np.array(sorted({c[1] for c in candidates if c[0] == "Ridge"}))
    sparse_paths = sorted({(c[0], c[2]) for c in candidates if c[0] != "Ridge"})
    bounds = {family: float("inf") for family, _, _ in candidates}

    def finished(item, family):
        # MSE is nonnegative. This lower bound proves a candidate cannot beat
        # the fully evaluated leader for the SAME degree and model family.
        return len(item["mse"]) == len(cv) or (
            prune and item["mse"] and sum(item["mse"]) / len(cv) >= bounds[family])

    for fold, (train, valid) in enumerate(cv, 1):
        transformer = preprocess(degree)
        A = np.asfortranarray(transformer.fit_transform(X[train]))
        B = transformer.transform(X[valid])
        yt, yv = y[train], y[valid]
        yc = yt - yt.mean()
        if len(ridge_alphas):
            predictions = ridge_predictions(A, B, yt, ridge_alphas)
            for j, alpha in enumerate(ridge_alphas):
                item = accum[("Ridge", float(alpha), 0.0)]
                if finished(item, "Ridge"):
                    continue
                item["mse"].append(float(mean_squared_error(yv, predictions[:, j])))
                item["r2"].append(float(r2_score(yv, predictions[:, j])))
                item["kkt"].append(0.0)
        for family, ratio in sparse_paths:
            alphas = sorted({c[1] for c in candidates if c[0] == family and c[2] == ratio}, reverse=True)
            estimator = sparse_estimator(family, alphas[0], ratio, thorough=thorough)
            for alpha in alphas:
                item = accum[(family, alpha, ratio)]
                if finished(item, family):
                    continue
                estimator.set_params(alpha=alpha)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    estimator.fit(A, yc)
                error = kkt_error(A, yc, estimator.coef_, alpha, ratio)
                item["warnings"] += len(caught)
                item["kkt"].append(error)
                pred = estimator.predict(B) + yt.mean()
                item["mse"].append(float(mean_squared_error(yv, pred)))
                item["r2"].append(float(r2_score(yv, pred)))
        LOG.info("  degree %d: inner fold %d/%d finished (%d terms)", degree, fold, len(cv), A.shape[1])
        if fold == 1 and len(cv) > 1 and prune:
            # Fully evaluate one promising candidate per family to establish
            # valid upper bounds. This uses inner data only, never outer data.
            seeds = []
            for family in bounds:
                keys = [k for k, v in accum.items() if k[0] == family and max(v["kkt"]) <= 5*TOL]
                if keys:
                    seeds.append(min(keys, key=lambda k: accum[k]["mse"][0]))
            extra = evaluate_degree(X, y, cv[1:], degree, seeds, thorough=thorough, prune=False)
            for result in extra:
                key = (result["model"], result["alpha"], result["l1_ratio"])
                item = accum[key]
                item["mse"].extend(result["fold_mse"])
                item["r2"].extend(result["fold_r2"])
                item["kkt"].append(result["max_kkt"])
                item["warnings"] += result["warning_count"]
                if max(item["kkt"]) <= 5*TOL:
                    bounds[key[0]] = float(np.mean(item["mse"]))
    result = []
    for (family, alpha, ratio), item in accum.items():
        # An unconverged numerical result cannot win the model search.
        complete = len(item["mse"]) == len(cv)
        converged = complete and max(item["kkt"]) <= 5 * TOL
        result.append(dict(degree=degree, model=family, alpha=alpha, l1_ratio=ratio,
                           mean_mse=float(np.mean(item["mse"])) if complete else None,
                           std_mse=float(np.std(item["mse"], ddof=1 if len(cv)>1 else 0)) if complete else None,
                           mean_r2=float(np.mean(item["r2"])) if complete else None,
                           fold_mse=item["mse"], fold_r2=item["r2"],
                           evaluated_folds=len(item["mse"]), pruned=not complete,
                           mse_lower_bound=float(sum(item["mse"])/len(cv)), max_kkt=max(item["kkt"]),
                           converged=converged, warning_count=item["warnings"]))
    return result


def best_row(rows):
    eligible = [r for r in rows if r["converged"] and np.isfinite(r["mean_mse"])]
    if not eligible:
        raise RuntimeError("No converged candidates; increase solver budgets before proceeding.")
    return min(eligible, key=lambda r: (r["mean_mse"], r["degree"]))


def search(X, y, groups, max_degree, directory, folds=3):
    directory.mkdir(parents=True, exist_ok=True)
    cv = splits(X, y, groups, folds, SEED)
    rows = []
    families = [("Ridge", 0.0), ("LASSO", 1.0)] + [("ElasticNet", r) for r in RATIOS]
    candidates = [(m, float(a), r) for m, r in families for a in BASE_ALPHAS]
    pending = {}
    for degree in range(1, max_degree + 1):
        checkpoint = directory / f"degree_{degree:02d}.json"
        if checkpoint.exists():
            degree_rows = json.loads(checkpoint.read_text())
            LOG.info("Resuming completed degree %d", degree)
        else:
            pending[POOL.submit(degree_job, X, y, cv, degree, candidates, checkpoint)] = degree
            continue
        rows.extend(degree_rows)
    for future in as_completed(pending):
        degree = pending[future]
        degree_rows, elapsed = future.result()
        rows.extend(degree_rows)
        winner = best_row(degree_rows)
        LOG.info("%s Degree %d: best %s alpha=%.5g MSE=%.7g [%.1fs]", directory.name,
                 degree, winner["model"], winner["alpha"], winner["mean_mse"], elapsed)
    # Give the three most promising unfinished candidates per family longer
    # retries when their provisional error is within 5% of that family's leader.
    # These provisional errors never qualify an unconverged candidate to win.
    retry_path = directory / "convergence_retries.json"
    if retry_path.exists():
        rows.extend(json.loads(retry_path.read_text()))
    else:
        retries = []
        for family in ("LASSO", "ElasticNet"):
            family_best = best_row([r for r in rows if r["model"] == family])
            promising = sorted([r for r in rows if r["model"] == family and not r["converged"]
                                and r["mean_mse"] is not None
                                and r["mean_mse"] <= 1.05*family_best["mean_mse"]],
                               key=lambda r: r["mean_mse"])[:3]
            for degree in sorted({r["degree"] for r in promising}):
                candidates = [(r["model"], r["alpha"], r["l1_ratio"])
                              for r in promising if r["degree"] == degree]
                retries.extend(evaluate_degree(X, y, cv, degree, candidates, thorough=True))
        json_save(retry_path, retries)
        rows.extend(retries)
    # Refine the best coarse configuration in each family, using inner CV only.
    # Extend a boundary winner by two decades; otherwise refine +/- one decade.
    for family in ("Ridge", "LASSO", "ElasticNet"):
        coarse = best_row([r for r in rows if r["model"] == family])
        checkpoint = directory / f"refine_{family}.json"
        if checkpoint.exists():
            rows.extend(json.loads(checkpoint.read_text()))
            continue
        exponent = np.log10(coarse["alpha"])
        low = exponent - (2 if exponent == -6 else 1)
        high = exponent + (2 if exponent == 6 else 1)
        alphas = np.logspace(low, high, 9)
        ratios = RATIOS if family == "ElasticNet" else (coarse["l1_ratio"],)
        candidates = [(family, float(a), r) for r in ratios for a in alphas]
        refined = evaluate_degree(X, y, cv, coarse["degree"], candidates, thorough=True)
        json_save(checkpoint, refined)
        rows.extend(refined)
    frame = pd.DataFrame(rows)
    frame.drop(columns=["fold_mse", "fold_r2"], errors="ignore").sort_values(["converged", "mean_mse"], ascending=[False, True]).to_csv(directory / "search_results.csv", index=False)
    winner = best_row(rows)
    json_save(directory / "winner.json", winner)
    LOG.info("SEARCH WINNER: %s degree=%d alpha=%.8g ratio=%.3g inner MSE=%.8g; %d/%d converged",
             winner["model"], winner["degree"], winner["alpha"], winner["l1_ratio"], winner["mean_mse"],
             sum(r["converged"] for r in rows), len(rows))
    return winner


def fit_pipeline(X, y, winner):
    prep = preprocess(winner["degree"])
    A = np.asfortranarray(prep.fit_transform(X))
    if winner["model"] == "Ridge":
        model = Ridge(alpha=winner["alpha"], solver="svd", fit_intercept=True)
        model.fit(A, y)
    else:
        model = sparse_estimator(winner["model"], winner["alpha"], winner["l1_ratio"], strict=True)
        model.fit(A, y - y.mean())
        residual = kkt_error(A, y-y.mean(), model.coef_, winner["alpha"], winner["l1_ratio"])
        if residual > 5*TOL:
            raise RuntimeError(f"Final model did not converge: KKT error {residual}")
        model.intercept_ = float(y.mean())
    return Pipeline(prep.steps + [("regression", model)])


def read_data(data_dir, roll, variant):
    train_path = data_dir / f"{roll}_train_var{variant}.csv"
    test_path = data_dir / f"{roll}_test_var{variant}.csv"
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    features = [f"x{i}" for i in range(1, 7 if variant == 1 else 4)]
    if list(train.columns) != features + ["y"] or list(test.columns) != features:
        raise ValueError("Unexpected CSV columns or feature order.")
    X, y, Xtest = train[features].to_numpy(float), train.y.to_numpy(float), test.to_numpy(float)
    if not all(np.isfinite(v).all() for v in (X, y, Xtest)):
        raise ValueError("Missing or nonfinite data must be addressed first.")
    _, groups = np.unique(X, axis=0, return_inverse=True)
    groups = groups if variant == 2 else None
    digest = hashlib.sha256(train_path.read_bytes() + test_path.read_bytes()).hexdigest()
    return X, y, Xtest, groups, features, digest


def run_variant(args, variant):
    X, y, Xtest, groups, features, digest = read_data(args.data_dir, args.roll, variant)
    directory = args.output_dir / f"var{variant}"
    directory.mkdir(parents=True, exist_ok=True)
    max_degree = 10 if variant == 1 else 20
    manifest = dict(data_sha256=digest, max_degree=max_degree, inner_folds=args.inner_folds,
                    outer_folds=args.outer_folds, seed=SEED, ratios=RATIOS,
                    alphas=BASE_ALPHAS.tolist(), tolerance=TOL,
                    python=platform.python_version(), sklearn=sklearn.__version__,
                    scipy=scipy.__version__, numpy=np.__version__, skglm=skglm.__version__,
                    algorithm_version=3, coarse_max_iter=10, coarse_max_epochs=500,
                    retry_max_iter=50, retry_max_epochs=10000)
    manifest = json.loads(json.dumps(manifest))
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        # Version 3 only adds mathematically safe lower-bound pruning, so
        # exhaustive version-2 checkpoints remain valid and can be retained.
        if previous.get("algorithm_version") == 2:
            previous["algorithm_version"] = 3
        if previous != manifest:
            raise ValueError("Data/settings/version changed; use a new --output-dir.")
    json_save(manifest_path, manifest)
    LOG.info("VAR%d: %d rows, %d inputs, degrees 1-%d; %s folds", variant, len(y), X.shape[1], max_degree,
             "grouped identical coordinates" if groups is not None else "shuffled")
    # Final selection is independent of all outer-fold evaluation scores.
    winner = search(X, y, groups, max_degree, directory / "final_search", args.inner_folds)
    fitted = fit_pipeline(X, y, winner)
    joblib.dump(fitted, directory / "best_model.joblib", compress=3)
    predictions = fitted.predict(Xtest)
    assert len(predictions) == len(Xtest) and np.isfinite(predictions).all()
    prediction_path = args.output_dir / f"{args.roll}_pred_var{variant}.csv"
    pd.DataFrame({"y": predictions}).to_csv(prediction_path, index=False)
    names = fitted.named_steps["polynomial"].get_feature_names_out(features)
    coefficients = pd.DataFrame({"term_in_standardized_inputs": names,
                                 "coefficient_of_standardized_term": fitted[-1].coef_,
                                 "term_mean": fitted.named_steps["term_scaler"].mean_,
                                 "term_scale": fitted.named_steps["term_scaler"].scale_})
    coefficients.to_csv(directory / "polynomial_coefficients.csv", index=False)
    LOG.info("\nBEST MODEL FOR VAR%d (full-data CV selection)\n%s\nDegree: %d\nAlpha: %.10g\nL1 ratio: %s\nMean inner-CV MSE: %.10g\nMean inner-CV R2: %.10g\nIntercept: %.10g\nPolynomial terms: %d\nPredictions: %s\n",
             variant, fitted, winner["degree"], winner["alpha"],
             winner["l1_ratio"] if winner["model"] == "ElasticNet" else "not applicable",
             winner["mean_mse"], winner["mean_r2"], fitted[-1].intercept_, len(names), prediction_path)
    if args.selection_only:
        selection = dict(variant=variant, selected_model=winner, outer_evaluation="not_run")
        json_save(directory / "selection_summary.json", selection)
        return selection
    outer_results = []
    oof = np.full(len(y), np.nan)
    outer_cv = splits(X, y, groups, args.outer_folds, SEED + 1)
    for number, (train, valid) in enumerate(outer_cv, 1):
        stage = directory / f"outer_{number}"
        outer_winner = search(X[train], y[train], None if groups is None else groups[train],
                              max_degree, stage, args.inner_folds)
        outer_model = fit_pipeline(X[train], y[train], outer_winner)
        oof[valid] = outer_model.predict(X[valid])
        result = dict(fold=number, train_rows=len(train), validation_rows=len(valid),
                      mse=float(mean_squared_error(y[valid], oof[valid])),
                      r2=float(r2_score(y[valid], oof[valid])), winner=outer_winner)
        outer_results.append(result)
        json_save(stage / "outer_evaluation.json", result)
        LOG.info("OUTER %d/%d VAR%d: MSE=%.8g R2=%.8g", number, args.outer_folds, variant, result["mse"], result["r2"])
    assert np.isfinite(oof).all()
    pd.DataFrame({"y_true": y, "outer_cv_prediction": oof}).to_csv(directory / "outer_predictions.csv", index=False)
    summary = dict(variant=variant, selected_model=winner,
                   outer_mean_mse=float(np.mean([r["mse"] for r in outer_results])),
                   outer_std_mse=float(np.std([r["mse"] for r in outer_results], ddof=1)),
                   outer_mean_r2=float(np.mean([r["r2"] for r in outer_results])),
                   outer_std_r2=float(np.std([r["r2"] for r in outer_results], ddof=1)),
                   pooled_outer_mse=float(mean_squared_error(y, oof)),
                   pooled_outer_r2=float(r2_score(y, oof)), outer_folds=outer_results)
    json_save(directory / "summary.json", summary)
    LOG.info("COMPLETED VAR%d BEST: %s degree=%d alpha=%.10g ratio=%s | inner MSE=%.8g | outer MSE=%.8g +/- %.8g | outer R2=%.8g +/- %.8g",
             variant, winner["model"], winner["degree"], winner["alpha"], winner["l1_ratio"], winner["mean_mse"],
             summary["outer_mean_mse"], summary["outer_std_mse"], summary["outer_mean_r2"], summary["outer_std_r2"])
    return summary


def self_check():
    """Verify optimized Ridge math and group isolation before expensive fitting."""
    rng = np.random.default_rng(SEED)
    for shape in ((40, 5), (15, 30)):
        X = rng.normal(size=shape)
        A = StandardScaler().fit_transform(X)
        B = rng.normal(size=(9, shape[1]))
        y = rng.normal(size=shape[0])
        alphas = np.array([1e-6, .1, 10., 1e6])
        fast = ridge_predictions(A, B, y, alphas)
        for j, alpha in enumerate(alphas):
            reference = Ridge(alpha=alpha, solver="svd").fit(A, y).predict(B)
            np.testing.assert_allclose(fast[:, j], reference, rtol=1e-5, atol=1e-6)
    X = rng.normal(size=(30, 3))
    groups = np.repeat(np.arange(10), 3)
    for train, valid in splits(X, np.zeros(30), groups, 3, SEED):
        assert not set(groups[train]) & set(groups[valid])
    LOG.info("Self-check passed: Ridge path matches sklearn; grouped folds are disjoint.")


def main():
    global POOL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--roll", default="BT2024175")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--variants", nargs="+", type=int, choices=[1, 2], default=[1, 2])
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--show-results", action="store_true", help="Print saved winners without retraining.")
    parser.add_argument("--selection-only", action="store_true", help="Select, fit, save and print winners; omit independent outer evaluation.")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent degree searches; each uses two BLAS threads.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(args.output_dir / "training.log", encoding="utf-8")])
    if args.show_results:
        for variant in args.variants:
            directory = args.output_dir / f"var{variant}"
            winner_path = directory / "final_search" / "winner.json"
            model_path = directory / "best_model.joblib"
            if not winner_path.exists() or not model_path.exists():
                print(f"var{variant}: final model is not ready yet.")
                continue
            winner = json.loads(winner_path.read_text())
            print(f"\nBEST MODEL FOR VAR{variant}")
            print(joblib.load(model_path))
            print(json.dumps(winner, indent=2))
            summary_path = directory / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                print(f"Nested outer MSE: {summary['outer_mean_mse']:.8g} +/- {summary['outer_std_mse']:.8g}")
                print(f"Nested outer R2: {summary['outer_mean_r2']:.8g} +/- {summary['outer_std_r2']:.8g}")
            else:
                print("Nested outer evaluation has not finished yet.")
        return
    with threadpool_limits(limits=2):
        self_check()
        if not args.check_only:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker) as executor:
                POOL = executor
                summaries = [run_variant(args, v) for v in args.variants]
            if args.selection_only:
                json_save(args.output_dir / "selection_summary.json", summaries)
                print("\nSELECTED WINNERS (outer evaluation not run)", flush=True)
                for summary in summaries:
                    w = summary["selected_model"]
                    print(f"var{summary['variant']}: {w['model']}, degree={w['degree']}, "
                          f"alpha={w['alpha']:.10g}, l1_ratio={w['l1_ratio']}, "
                          f"inner CV MSE={w['mean_mse']:.8g}, inner CV R2={w['mean_r2']:.8g}", flush=True)
                return
            # Separate --variants runs may finish independently; retain both
            # completed variant summaries in the combined terminal report.
            summaries = [json.loads((args.output_dir / f"var{v}" / "summary.json").read_text())
                         for v in (1, 2) if (args.output_dir / f"var{v}" / "summary.json").exists()]
            json_save(args.output_dir / "summary.json", summaries)
            print("\nFINAL WINNERS", flush=True)
            for summary in summaries:
                w = summary["selected_model"]
                print(f"var{summary['variant']}: {w['model']}, degree={w['degree']}, alpha={w['alpha']:.10g}, "
                      f"l1_ratio={w['l1_ratio']}, inner CV MSE={w['mean_mse']:.8g}, "
                      f"nested outer MSE={summary['outer_mean_mse']:.8g}, "
                      f"nested outer R2={summary['outer_mean_r2']:.8g}", flush=True)


if __name__ == "__main__":
    main()
