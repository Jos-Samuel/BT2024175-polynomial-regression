# BT2024175 polynomial regression


Repository: https://github.com/Jos-Samuel/BT2024175-polynomial-regression

The five-page [assignment report](BT2024175_Polynomial_Regression_Report.pdf) includes this repository link.
Completed nested CV selected degree-5 LASSO for var1 and degree-11 Elastic Net for var2.
Submission files are [var1](results/BT2024175_pred_var1.csv) and [var2](results/BT2024175_pred_var2.csv).
The personalized input datasets are not distributed here; place your supplied CSVs in a local data directory.

## Quick start (Windows PowerShell)

Follow steps 1–6 in order. Run each command on its own line, and continue only if it succeeds.
The saved models are included, so you can generate predictions without waiting for training.
These commands are for Windows PowerShell, not Command Prompt or a Python prompt.

### 1. Check the prerequisites

Install Git and 64-bit Python 3.12, including the Windows Python launcher, before starting.
Open a new PowerShell window after installation and check:

```powershell
git --version
py -3.12 --version
```

The second command must print Python 3.12.x. If either command is not found, finish installing
that prerequisite before continuing. This project uses the versions pinned in `requirements.txt`.

### 2. Clone the repository and enter its folder

Run this from a directory where you want to keep the project:

```powershell
git clone https://github.com/Jos-Samuel/BT2024175-polynomial-regression.git
cd .\BT2024175-polynomial-regression
Get-Item .\requirements.txt, .\predict.py, .\train_polynomial.py
```

If you already cloned the repository, open PowerShell in that existing repository folder
and skip the clone command. All remaining commands must run from the folder containing
`README.md`, `predict.py`, and `requirements.txt`.

### 3. Create the Python environment and install packages

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Wait for installation to finish. The final command should report no broken requirements.
Every command below uses this environment explicitly; activation is unnecessary.

### 4. Copy the supplied test datasets into the project

The input datasets are not included in GitHub. Download the two original test CSVs supplied
for roll number BT2024175 into your Windows `Downloads` folder, keeping their filenames.
Then run:

```powershell
New-Item -ItemType Directory -Force .\data | Out-Null
Copy-Item -LiteralPath "$env:USERPROFILE\Downloads\BT2024175_test_var1.csv" -Destination .\data\
Copy-Item -LiteralPath "$env:USERPROFILE\Downloads\BT2024175_test_var2.csv" -Destination .\data\
Get-Item .\data\BT2024175_test_var1.csv, .\data\BT2024175_test_var2.csv
```

If your Downloads folder is redirected or the files are elsewhere, replace the source paths
in the two `Copy-Item` commands with their actual locations. var1 must have columns
`x1,x2,x3,x4,x5,x6`; var2 must have `x1,x2,x3`, in that order. Test files must not contain `y`.

### 5. Generate predictions for both datasets

Run all three lines in the same PowerShell window:

```powershell
$predictionDir = Join-Path "predictions" (Get-Date -Format "yyyyMMdd_HHmmss_fff")
.\.venv\Scripts\python.exe predict.py --variant 1 --input ".\data\BT2024175_test_var1.csv" --output "$predictionDir\BT2024175_pred_var1.csv"
.\.venv\Scripts\python.exe predict.py --variant 2 --input ".\data\BT2024175_test_var2.csv" --output "$predictionDir\BT2024175_pred_var2.csv"
```

Each command prints the loaded model and the output path. Both CSVs go into the same dated
subfolder under `predictions`, with the required submission filenames and one `y` column.
For the supplied test sets, each file has 1,000 predictions in the original row order.
Run all three lines again to generate another pair in a new subfolder.
`predict.py` refuses to overwrite an existing output. Only load trusted model files.

### 6. Print the saved best models and scores

```powershell
.\.venv\Scripts\python.exe train_polynomial.py --show-results
```

Expected selections: degree-5 LASSO with alpha 0.01 for var1, and degree-11 Elastic Net
with alpha 0.001 and L1 mixing ratio 0.3 for var2. This command displays the saved results;
it does not train the models. The completed submission CSVs also remain available in `results`.

## Optional: train both models from scratch

Complete steps 1–4 first. Download the original training CSVs for BT2024175 to Downloads,
then copy them into `data`:

```powershell
Copy-Item -LiteralPath "$env:USERPROFILE\Downloads\BT2024175_train_var1.csv" -Destination .\data\
Copy-Item -LiteralPath "$env:USERPROFILE\Downloads\BT2024175_train_var2.csv" -Destination .\data\
.\.venv\Scripts\python.exe train_polynomial.py --data-dir .\data --output-dir .\rerun_results
```

Training files must contain the corresponding input columns followed by `y`.
This command runs the complete search and five-fold outer evaluation for both datasets.
It can take substantially longer than prediction. Progress appears in the terminal and
`rerun_results/training.log`; the final `FINAL WINNERS` section prints both winners and scores.
Use `--workers 1` on a machine with limited memory.

The training script performs two kinds of fitting. During cross-validation, temporary
models are fitted separately inside each training fold to compare candidate settings.
After the winning degree and regularization settings are selected, the script fits one
final model on all 1,000 labelled training rows for that variant and saves it as
`rerun_results/var1/best_model.joblib` or `rerun_results/var2/best_model.joblib`.
The prediction script loads those saved models and does not fit them again.

The new models, prediction CSVs and validation records are written to `rerun_results`.
Keep that separate from the included `results` directory. To display the new results after
training finishes:

```powershell
.\.venv\Scripts\python.exe train_polynomial.py --output-dir .\rerun_results --show-results
```

To use these newly trained models with the prediction commands in step 5, add
`--model-dir .\rerun_results` to each command. Without that option, prediction uses the
original included models in `results`.

`--selection-only` skips outer evaluation for a shorter training run. Its tuning scores
are not independent nested-CV performance estimates. `--check-only` checks the optimized
Ridge calculation and grouped-fold isolation without running the full search.

## Optional: verify the included results

This requires all four original CSVs in `data`, including the training files copied above.
It checks the included `results` directory, not `rerun_results`:

```powershell
.\.venv\Scripts\python.exe verify_results.py --data-dir .\data --output .\verification_local.json
```

The script checks saved predictions, outer-fold coverage and scores, group isolation,
search-record accounting, and the arithmetic of recorded pruning bounds. It writes a local
verification report without retraining. This check does not rerun the complete experiment
or reproduce an exhaustive-versus-pruned search comparison.

## Troubleshooting

- **`.venv\Scripts\python.exe` is not recognized:** make sure you are in the repository
  folder from step 2 and have completed step 3. Check `Test-Path .\.venv\Scripts\python.exe`.
- **`predict.py` cannot be found:** return to the repository folder, not its parent directory.
- **A CSV cannot be found:** check step 4 and the actual download location. Input data is
  supplied separately from the repository.
- **An output already exists:** rerun all three lines in step 5 to choose a new output folder.
- **Missing Python packages:** repeat installation in step 3 and use the explicit
  `.\.venv\Scripts\python.exe` interpreter for subsequent commands.

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

The supplied test sets contain no targets, so their true MSE/R2 cannot be computed.
Full-data inner-CV scores are selection scores; nested outer scores are the
held-out performance estimates appropriate for the assignment report.
