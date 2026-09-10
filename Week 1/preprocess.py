"""Week 1 Auto MPG acquisition, audit and preprocessing. Run: python preprocess.py."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import urllib.request
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parent
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
COLS = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
        'acceleration', 'model_year', 'origin', 'car_name']
NUM = ['displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
CAT = ['cylinders', 'origin']
# Conservative first-token corrections; raw names are retained for review.
ALIASES = {'chevroelt': 'chevrolet', 'chevy': 'chevrolet',
           'vokswagen': 'volkswagen', 'vw': 'volkswagen',
           'toyouta': 'toyota', 'maxda': 'mazda', 'mercedes': 'mercedes-benz'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Refresh raw file from UCI')
    args = parser.parse_args()
    raw_path = ROOT / 'data/raw/auto-mpg.data'
    out = ROOT / 'outputs'
    out.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if args.download or not raw_path.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
        # Validate before replacing the existing local copy.
        if len(payload.splitlines()) != 398:
            raise ValueError('Unexpected source response; existing raw file preserved')
        raw_path.write_bytes(payload)
    raw = pd.read_csv(raw_path, sep=r'\s+', names=COLS, quotechar='"', dtype=str)
    if raw.shape != (398, 9):
        raise ValueError(f'Unexpected source shape: {raw.shape}')
    raw.to_csv(ROOT / 'data/auto_mpg_raw.csv', index=False)
    df = raw.replace('?', np.nan).copy()
    audit = {'rows_raw': len(raw), 'columns_raw': len(COLS),
             'source_url': URL, 'sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest()}
    invalid = {}
    for col in COLS[:-1]:
        converted = pd.to_numeric(df[col], errors='coerce')
        invalid[col] = int((df[col].notna() & converted.isna()).sum())
        df[col] = converted
    audit['invalid_numeric_tokens'] = invalid
    audit['missing_before'] = df.isna().sum().to_dict()
    audit['duplicates'] = int(df.duplicated().sum())
    # Identical records may be separate vehicles; require review rather than silent removal.
    if audit['duplicates']:
        raise ValueError('Duplicate records found; review identity before proceeding')
    df.insert(0, 'row_id', np.arange(1, len(df) + 1))
    df['car_name_raw'] = df['car_name']
    df['car_name'] = df['car_name'].str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    df['car_name'] = df['car_name'].str.replace(r'^mercedes benz ', 'mercedes-benz ', regex=True)
    words = df['car_name'].str.split(' ', n=1, expand=True)
    df['car_name'] = words[0].replace(ALIASES) + ' ' + words[1].fillna('')
    df['car_name'] = df['car_name'].str.strip()
    audit['names_changed'] = int((df.car_name != df.car_name_raw).sum())
    df.loc[df.car_name != df.car_name_raw, ['row_id', 'car_name_raw', 'car_name']].to_csv(out / 'name_corrections.csv', index=False)
    checks = {c: int((df[c].notna() & (df[c] <= 0)).sum())
              for c in ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']}
    for c, allowed in [('cylinders', [3, 4, 5, 6, 8]), ('origin', [1, 2, 3]),
                       ('model_year', list(range(70, 83)))]:
        checks[c] = int((~df[c].isin(allowed)).sum())
    audit['domain_violations'] = checks
    if any(checks.values()) or any(invalid.values()) or df['mpg'].isna().any():
        raise ValueError('Unexpected invalid values; inspect source instead of guessing repairs')
    df['horsepower_missing'] = df['horsepower'].isna().astype(int)
    df[COLS[:-1]].describe().round(4).to_csv(out / 'initial_statistics.csv')
    train_ids, test_ids = train_test_split(df.index, test_size=0.20, random_state=42)
    train, test = df.loc[train_ids].copy(), df.loc[test_ids].copy()
    audit['train_rows'], audit['test_rows'] = len(train), len(test)
    audit['train_missing_hp'], audit['test_missing_hp'] = int(train.horsepower.isna().sum()), int(test.horsepower.isna().sum())
    # Learn outlier fences only from observed training predictors.
    continuous = ['displacement', 'horsepower', 'weight', 'acceleration']
    fences, flags = [], pd.DataFrame({'row_id': df.row_id})
    for col in continuous:
        q1, q3 = train[col].quantile([.25, .75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        flagged = (df[col] < lo) | (df[col] > hi)
        flags[col + '_outlier'] = flagged
        fences.append({'feature': col, 'lower': lo, 'upper': hi,
                       'train_flagged': int(flagged.loc[train_ids].sum()),
                       'test_flagged': int(flagged.loc[test_ids].sum()),
                       'total_flagged': int(flagged.sum())})
    pd.DataFrame(fences).to_csv(out / 'outlier_summary.csv', index=False)
    flags.to_csv(out / 'outlier_flags.csv', index=False)
    audit['outlier_rows'] = int(flags.drop(columns='row_id').any(axis=1).sum())
    audit['outliers'] = fences
    # Missingness remains visible in the human-readable cleaned data.
    hp_median = float(train.horsepower.median())
    audit['horsepower_train_median'] = hp_median
    cleaned = df.copy()
    cleaned['horsepower'] = cleaned['horsepower'].fillna(hp_median)
    cleaned['split'] = np.where(cleaned.index.isin(train_ids), 'train', 'test')
    cleaned.to_csv(out / 'auto_mpg_cleaned.csv', index=False)
    transformer = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')),
                              ('scaler', StandardScaler())]), NUM),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT),
        ('missing', 'passthrough', ['horsepower_missing'])])
    x_train = transformer.fit_transform(train)
    x_test = transformer.transform(test)
    feature_names = transformer.get_feature_names_out()
    for split, frame, matrix in [('train', train, x_train), ('test', test, x_test)]:
        result = pd.DataFrame(matrix, columns=feature_names)
        result.insert(0, 'row_id', frame.row_id.to_numpy())
        result.to_csv(out / f'X_{split}.csv', index=False)
        frame[['row_id', 'mpg']].to_csv(out / f'y_{split}.csv', index=False)
    joblib.dump(transformer, out / 'preprocessor.joblib')
    assert set(train.row_id).isdisjoint(test.row_id)
    assert len(train) + len(test) == len(df)
    assert np.isfinite(x_train).all() and np.isfinite(x_test).all()
    assert cleaned[COLS].isna().sum().sum() == 0
    assert np.allclose(joblib.load(out / 'preprocessor.joblib').transform(test), x_test)
    assert np.allclose(x_train[:, :len(NUM)].mean(axis=0), 0, atol=1e-10)
    audit['features_prepared'] = len(feature_names)
    audit['missing_after'] = int(cleaned[COLS].isna().sum().sum())
    audit['validation'] = 'PASS: split separation, row preservation, finite matrices, no missing cleaned values, saved transformer round trip, training scaling'
    audit['python_version'] = platform.python_version()
    audit['versions'] = {p: importlib.metadata.version(p) for p in ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'joblib', 'python-docx']}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].hist(df.horsepower.dropna(), bins=18, color='#376780', edgecolor='white')
    axes[0].axvline(hp_median, color='#ac4c33', linestyle='--', label=f'Training median {hp_median:g}')
    axes[0].set(xlabel='Horsepower', ylabel='Vehicles', title='Observed horsepower before imputation')
    axes[0].legend(fontsize=8)
    axes[1].scatter(df.weight, df.mpg, s=13, alpha=.65, color='#376780')
    axes[1].set(xlabel='Weight in source units', ylabel='Miles per gallon', title='Weight and fuel efficiency')
    fig.tight_layout()
    fig.savefig(out / 'exploration.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, col in zip(axes, ['horsepower', 'acceleration']):
        ax.boxplot(train[col].dropna(), vert=False)
        ax.set(xlabel=col, title=f'Training {col} with 1.5 IQR whiskers')
    fig.tight_layout()
    fig.savefig(out / 'outliers.png', dpi=180)
    plt.close(fig)
    (out / 'audit_summary.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    message = (f'Raw data: {len(raw)} rows x {len(COLS)} columns\n'
               f'Missing horsepower: {audit["missing_before"]["horsepower"]}\n'
               f'Names standardized: {audit["names_changed"]}\n'
               f'Outlier rows flagged and retained: {audit["outlier_rows"]}\n'
               f'Train/test: {len(train)}/{len(test)}; features: {len(feature_names)}\n'
               f'Training horsepower median: {hp_median}\n{audit["validation"]}\n')
    (out / 'run_summary.txt').write_text(message, encoding='utf-8')
    print(message)

if __name__ == '__main__':
    main()
