"""Predict with a trusted saved pipeline; no training or model selection."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', type=int, choices=[1, 2], required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new path to preserve saved predictions.')
    data = pd.read_csv(args.input)
    columns = [f'x{i}' for i in range(1, 7 if args.variant == 1 else 4)]
    if list(data.columns) != columns or data.empty:
        parser.error(f'Expected nonempty input with columns in order: {columns}')
    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        parser.error('Input must contain only finite numeric values.')
    model = joblib.load(args.model_dir / f'var{args.variant}' / 'best_model.joblib')
    predictions = model.predict(values)
    if predictions.shape != (len(data),) or not np.isfinite(predictions).all():
        raise ValueError('Model returned invalid predictions.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', newline='', encoding='utf-8') as handle:
        pd.DataFrame({'y': predictions}).to_csv(handle, index=False)
    print(model)
    print(f'Saved {len(data)} predictions to {args.output}')


if __name__ == '__main__':
    main()
