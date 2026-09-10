# Week 2 EDA submission

## Submit
- `Week2_EDA_Report.docx`: complete Word report with interpretations, code excerpts and eight annotated figures.
- `Week2_EDA.ipynb`: executed notebook with the full analysis and saved outputs.

The ZIP contains all supporting materials. Add an institutional cover sheet only if your course requires one; no unprovided roll number or institution has been invented.

## Files
- `analysis.py`: equivalent runnable Python analysis.
- `data/penguins.csv`: source-value CSV snapshot, 344 rows and 8 columns.
- `figures/`: eight PNG charts used in the report.
- `tables/`: statistics, quality audit, correlations, flags and sensitivity results.
- `requirements.txt`: analysis versions plus optional JupyterLab.
- `DATA_SOURCE.md`: source, acquisition method, license and checksum.

## Run
Use Python 3.12 or newer compatible with the pinned packages.
Open a terminal in this extracted folder:

```bash
python -m pip install -r requirements.txt
python analysis.py
```

Alternatively launch JupyterLab from this folder, open `Week2_EDA.ipynb` and choose Restart Kernel and Run All. Keep the data folder next to the notebook. The original CSV is never overwritten. Re-running replaces the generated figures and tables. Network access is not needed for the analysis after dependencies are installed.

## Method
All 344 records contribute to categorical counts; 342 have complete measurements; 333 have observed sex and mass. No imputation or permanent outlier deletion. Pearson and Spearman correlations, within-species IQR screens, missingness sensitivity and annual summaries are descriptive, not causal. Notebook cells were executed sequentially in a shared Python namespace and their text and chart outputs saved. No measured claim of 30–35 hours spent is made; that is the assignment's suggested effort budget.

## Report format
Both `Week2_EDA_Report.docx` (modern Word) and `Week2_EDA_Report.doc` (legacy Word 97–2003) are supplied. Use the format your submission portal accepts. The report contains your first name, Mayank; add your roll number or institution if your course requires them.
