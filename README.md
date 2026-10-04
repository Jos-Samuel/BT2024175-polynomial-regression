# BT2024175 polynomial regression


Repository: https://github.com/Jos-Samuel/BT2024175-polynomial-regression

The five-page [assignment report](BT2024175_Polynomial_Regression_Report.pdf) includes this repository link.
Completed nested CV selected degree-5 LASSO for var1 and degree-11 Elastic Net for var2.
Submission files are [var1](results/BT2024175_pred_var1.csv) and [var2](results/BT2024175_pred_var2.csv).
The personalized input datasets are not distributed here; place your supplied CSVs in a local data directory.

## Inference with the saved models

After installing the pinned requirements, run these commands from this repository:

```powershell
.\.venv\Scripts\python.exe predict.py --variant 1 --input "data/BT2024175_test_var1.csv" --output "predictions/BT2024175_pred_var1.csv"
.\.venv\Scripts\python.exe predict.py --variant 2 --input "data/BT2024175_test_var2.csv" --output "predictions/BT2024175_pred_var2.csv"
.\.venv\Scripts\python.exe train_polynomial.py --show-results
```

Inference validates feature names/order and finite values, prints the loaded model, and exports a single `y` column without an index. It refuses to overwrite an existing prediction file. No fitting is performed. Load only trusted model files.

## Run

Use Python 3.12 or another version compatible with `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_polynomial.py --data-dir "$env:USERPROFILE\Downloads" --output-dir rerun_results
```

The script prints progress and the selected model for each dataset in the terminal.
The final `FINAL WINNERS` section prints both winners and their nested CV scores.
The command above writes a fresh run to `rerun_results`, preserving the included completed experiments. Without `--output-dir`, outputs default to `results` beside the script; do not retrain into that directory if you want to preserve the archived results. Run `--help` for options.
`--check-only` verifies the optimized Ridge calculation and grouped fold isolation.
`--show-results` prints the saved best models and available scores without retraining.
`verify_results.py` performs a read-only audit of the saved models, predictions, nested-fold
coverage, grouped isolation, search records and pruning bounds:

```powershell
.\.venv\Scripts\python.exe verify_results.py --data-dir "$env:USERPROFILE\Downloads"
```
For an explicitly shorter run, `--selection-only` performs the model search,
fits and prints the winners, and exports predictions, but omits outer evaluation.
Its scores are model-selection scores and must not be described as independent
nested-CV performance estimates. The default still runs the complete nested plan.

## Method

- var1: total polynomial degrees 1 through 10, with all interactions.
- var2: total polynomial degrees 1 through 20, with all interactions.
- Each training fold separately fits input standardization, polynomial expansion,
  and polynomial-column standardization. Validation data only uses `transform`.
- Compare Ridge, LASSO, and Elastic Net on identical folds within each search.
- Initial alpha values are 10 raised to every integer from -6 through 6.
- Elastic Net ratios are 0.1, 0.3, 0.5, 0.7, 0.9 and 0.95.
- Select by mean 3-fold validation MSE. Keep each model's fold average separate.
- For each degree and family, fully evaluate a promising candidate first. Stop
  evaluating another candidate only when its accumulated squared error divided
  by the total fold count already exceeds that leader's full CV error. Remaining
  fold errors cannot be negative, so that candidate cannot win. Search tables
  flag such rows as `pruned` and report their lower bound rather than inventing a
  full CV mean. The best fully evaluated model per degree/family is retained.
- Refine alpha around the best coarse configuration in each family using nine
  logarithmically spaced points. Extend boundary winners by two decades in the
  boundary direction. For Elastic Net, recheck all six ratios at that degree.
- Evaluate the entire selection procedure using five independent outer folds.
  Outer scores never choose the final model or alter the search.
- Fit the final selected pipeline on all labelled rows and predict every test row.
- var1 uses shuffled KFold; var2 uses shuffled GroupKFold, grouping exact duplicate
  input coordinates. Seed 175 is used internally; outer folds use seed 176.

The full-data search is run before outer evaluation to make the final fitted model
available early. This does not leak outer scores into model selection: each outer
fold independently repeats the complete search on only its own training rows.
The final reported outer scores evaluate the selection procedure, not a single
fixed configuration fitted to all rows.

## Computation and numerical reliability

Ridge predictions use a shared eigendecomposition across alpha values, verified
against scikit-learn Ridge. LASSO/Elastic Net use the accelerated `skglm` solvers,
which optimize the usual L1 and L1/L2 objectives. Warm starts are reused only
between alpha values with the same training fold, degree and ratio.

Sparse fits start with at most 10 working-set iterations and 500 coordinate
epochs per working set, and are checked independently using first-order
optimality (KKT) conditions. Candidates with maximum residual above 0.00005
are recorded but excluded from selection. Up to three promising unfinished
candidates per family (within 5% of that family's leading MSE) receive longer
retries: 50 working-set iterations and 10,000 epochs. Alpha refinement also uses
this longer budget. Final sparse models use tighter solver settings and
must also pass the KKT check. Thus "best" means the lowest mean CV MSE among the
converged candidates evaluated, not a guarantee of the global best predictor.
The search tables explicitly identify any excluded candidates and warnings.
The supplied run retained already-converged var1 degree-1 through degree-5
checkpoints from an initial pass with the longer solver budget; those results
passed the same KKT threshold. A clean rerun may differ slightly within solver
tolerances.
Exhaustive checkpoints created before lower-bound pruning was enabled can also
be resumed: their full scores remain valid. The pruning optimization was checked
against exhaustive CV for all three model families on a controlled dataset.

Three degree searches run concurrently, each with two numerical-library threads
to limit memory/CPU pressure. Set `--workers 1` on a smaller machine. Degree-level JSON
checkpoints allow the same command to resume without repeating completed degrees.
Input hashes, search settings, package versions and an algorithm version are
recorded in `manifest.json`; changed inputs/settings require a new output directory.
Do not share a results directory between concurrent runs of the same variant.

## Outputs

- `BT2024175_pred_var1.csv`, `BT2024175_pred_var2.csv`: single `y` column, original
  test-row order, no index. These match the supplied sample submission schema.
- `var1/best_model.joblib`, `var2/best_model.joblib`: fitted complete pipelines.
- `var*/final_search/search_results.csv`: all final-search model scores.
- `var*/outer_*/search_results.csv`: independent inner searches for outer folds.
- `var*/summary.json`: selected settings and nested MSE/R2 results.
- `var*/outer_predictions.csv`: out-of-fold predictions for all labelled rows.
- `var*/polynomial_coefficients.csv`: all fitted polynomial coefficients.
- `training.log`: terminal output saved to disk.

The coefficient names refer to standardized input variables. Coefficients multiply
standardized polynomial columns; use the saved pipeline for prediction rather than
applying those coefficients directly to raw inputs. Scalers and the intercept are
stored in the pipeline. Only load model files from trusted sources.

Example inference after training:

```python
import joblib
import pandas as pd

model = joblib.load("results/var1/best_model.joblib")
test = pd.read_csv("BT2024175_test_var1.csv")
prediction = model.predict(test.to_numpy(dtype=float))
print(model)
```

The supplied test sets contain no targets, so their true MSE/R2 cannot be computed.
Full-data inner-CV scores are selection scores; nested outer scores are the
held-out performance estimates appropriate for the assignment report.
