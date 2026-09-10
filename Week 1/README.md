# Week 1 submission — Auto MPG

Open **Week1_Report.docx** for the complete report. The Word document is in modern .docx format. No personal name, college, roll number or unverified work hours have been invented; add any institution-required identity details before submitting.

## Run

Use Python 3.12 (tested with 3.12.14). Extract the ZIP, then open a terminal in this folder.

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
python preprocess.py
python build_report.py
```

The default workflow uses the included raw dataset and needs no network after dependency installation. To reacquire it from the original publisher:

```bash
python preprocess.py --download
python build_report.py
```

The report describes this specific source snapshot. If refreshed data change, review the generated audit and narrative before submission. The report builder updates numerical tables but some explanatory text is specific to the verified snapshot. The script intentionally stops for an unexpected shape, invalid numerical tokens, domain violations or duplicate records requiring review.

## What to submit

- Week1_Report.docx — detailed report with actual results, code snippets, figures, reasoning, limitations and references.
- preprocess.py — acquisition, exploration, cleaning, preprocessing and verification.
- requirements.txt — tested dependencies.
- Include the complete project ZIP if supporting datasets and outputs are accepted.

## Data provenance

Dataset: Auto MPG, credited to R. Quinlan (1993), UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C5859H
Source page: https://archive.ics.uci.edu/dataset/9/auto+mpg
Raw download: https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data
Kaggle listing: https://www.kaggle.com/datasets/uciml/autompg-dataset
License listed by UCI: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
Acquired: 10 September 2026.

The download used UCI, not Kaggle. Raw source files are unmodified. The original 406-row variant is included only as supplied source context; **auto-mpg.data**, the 398-row variant, is the input. The raw CSV preserves question marks. Cleaned and prepared outputs are derived adaptations. No synthetic records or artificial errors were added.

## Output guide

- data/raw/auto-mpg.data: exact input text.
- data/raw/auto-mpg.names: source documentation.
- data/auto_mpg_raw.csv: parsed raw data.
- outputs/auto_mpg_cleaned.csv: all 398 records; imputed horsepower; original and corrected names; missingness flag; split label.
- outputs/X_train.csv and X_test.csv: 14 prepared predictors **plus row_id**. Drop row_id before fitting a model.
- outputs/y_train.csv and y_test.csv: target mpg and row_id. Use row_id to verify alignment.
- outputs/preprocessor.joblib: fitted scikit-learn ColumnTransformer, not a trained prediction model. Load only trusted joblib files, using the recorded package versions.
- outputs/name_corrections.csv: every changed name and its original.
- outputs/outlier_summary.csv: training-derived fences and partition counts.
- outputs/outlier_flags.csv: row-level flags, not predictors.
- outputs/initial_statistics.csv: numerical exploration before imputation.
- outputs/audit_summary.json: checksum, quality counts, package versions and validation status.
- outputs/run_summary.txt: actual console summary.
- outputs/exploration.png and outliers.png: generated report figures.
- build_report.py: regenerates the Word document from outputs.

Use original parsed data with the documented deterministic cleaning before future cross-validation and refit preprocessing within each fold. Do not fit new cross-validation preprocessing on the already imputed full cleaned CSV or the prepared matrices. The supplied train/test files are for the fixed Week 1 split. The horsepower_missing field must be computed from unfilled horsepower when applying the saved transformer to new input. Supply NUM, CAT and horsepower_missing columns with the same schema.

## Verified result

398 rows retained; 6 horsepower values imputed using training median 92; 15 names standardized; no exact duplicates or observed domain violations; 20 distinct outlier records flagged and retained; 318/80 training/test split; 14 prepared predictors. Assertions passed for disjoint partitions, preserved rows, finite arrays, complete cleaned fields, training scaling and saved-transformer equivalence. No model performance is claimed.
