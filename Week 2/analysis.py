"""Reproduce the Week 2 analysis."""
import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import hashlib, json, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value.to_string() if hasattr(value, 'to_string') else value)
ROOT = Path.cwd()
if not (ROOT / 'data/penguins.csv').exists():
    raise FileNotFoundError('Run from the extracted Week2_EDA folder.')
for folder in ['figures', 'tables']:
    (ROOT / folder).mkdir(exist_ok=True)
sns.set_theme(style='whitegrid', context='notebook')
ORDER = ['Adelie', 'Chinstrap', 'Gentoo']
COLORS = dict(zip(ORDER, ['#287D8E', '#C06A32', '#74529B']))
NUM = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
LABELS = ['Bill length (mm)', 'Bill depth (mm)', 'Flipper length (mm)', 'Body mass (g)']
df = pd.read_csv(ROOT / 'data/penguins.csv', na_values=['NA'])
def savefig(name):
    plt.tight_layout()
    plt.savefig(ROOT / 'figures' / f'{name}.png', dpi=180, bbox_inches='tight')
    plt.show()
    plt.close()
print('Shape:', df.shape)
print('SHA256:', hashlib.sha256((ROOT / 'data/penguins.csv').read_bytes()).hexdigest())
display(df.head())
df.info()

quality = pd.DataFrame({'dtype': df.dtypes.astype(str), 'missing': df.isna().sum(),
                        'missing_pct': df.isna().mean() * 100})
quality.to_csv(ROOT / 'tables/data_quality.csv')
display(quality)
print('Exact duplicate rows:', df.duplicated().sum())
print('Nonpositive measurements:', int((df[NUM] <= 0).sum().sum()))
for col in ['species', 'island', 'sex', 'year']:
    print(col, df[col].dropna().unique())
assert df.shape == (344, 8)
assert df['species'].isin(ORDER).all()
assert df['island'].isin(['Biscoe', 'Dream', 'Torgersen']).all()
assert df['sex'].dropna().isin(['female', 'male']).all()
assert df['year'].isin([2007, 2008, 2009]).all()
assert (df[NUM].dropna() > 0).all().all()
measurements = df.dropna(subset=NUM).copy()
sex_data = df.dropna(subset=['sex', 'body_mass_g']).copy()
print('All records:', len(df), '| complete measurements:', len(measurements),
      '| sex and mass observed:', len(sex_data))
print('Missing sex by species:')
display(df.assign(missing_sex=df.sex.isna()).groupby('species').missing_sex.agg(['sum', 'mean']))

summary = df[NUM].describe().T
summary['IQR'] = summary['75%'] - summary['25%']
summary.to_csv(ROOT / 'tables/descriptive_statistics.csv')
display(summary.round(2))
species_stats = df.groupby('species')['body_mass_g'].agg(['count','mean','median','std','min','max']).reindex(ORDER)
species_stats.to_csv(ROOT / 'tables/species_mass_summary.csv')
display(species_stats.round(2))

counts = df.species.value_counts().reindex(ORDER)
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(ORDER, counts, color=[COLORS[s] for s in ORDER])
ax.bar_label(bars, labels=[f'{n} ({n/len(df):.1%})' for n in counts], padding=4)
ax.set(xlabel='Species', ylabel='Number of sampled penguins', ylim=(0, 180),
       title='Species representation in the sample (n = 344)')
savefig('01_species_counts')
island = pd.crosstab(df.island, df.species).reindex(columns=ORDER)
island.to_csv(ROOT / 'tables/island_species_counts.csv')
display(island)
fig, ax = plt.subplots(figsize=(8, 4))
island.plot.bar(stacked=True, ax=ax, color=[COLORS[s] for s in ORDER], rot=0)
ax.set(xlabel='Island', ylabel='Number of sampled penguins',
       title='Species and island are intertwined in this sample')
ax.legend(title='Species', loc='upper right')
for container in ax.containers:
    ax.bar_label(container, label_type='center', labels=[str(int(v)) if v else '' for v in container.datavalues], color='white')
savefig('02_island_species')

fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for col, label, ax in zip(NUM, LABELS, axes.flat):
    sns.histplot(data=df, x=col, bins=18, color='#287D8E', ax=ax)
    ax.axvline(df[col].median(), color='#9D452F', linestyle='--', label=f'Median: {df[col].median():.1f}')
    ax.set(xlabel=label, ylabel='Count')
    ax.legend(fontsize=8)
fig.suptitle('Distributions of measured body dimensions (n = 342 each)', y=1.02)
savefig('03_distributions')

sex_summary = sex_data.groupby(['species','sex']).body_mass_g.agg(['count','mean','median','std'])
sex_summary.to_csv(ROOT / 'tables/species_sex_mass.csv')
display(sex_summary.round(2))
fig, ax = plt.subplots(figsize=(8, 4.4))
sns.boxplot(data=sex_data, x='species', y='body_mass_g', hue='sex', order=ORDER,
            hue_order=['female','male'], palette=['#73A9B2','#334D70'], ax=ax)
ax.set(xlabel='Species', ylabel='Body mass (g)', title='Male penguins have higher median mass within each species (n = 333)')
ax.legend(title='Recorded sex')
savefig('04_mass_species_sex')

pearson = measurements[NUM].corr(method='pearson')
spearman = measurements[NUM].corr(method='spearman')
pearson.to_csv(ROOT / 'tables/pearson_correlations.csv')
spearman.to_csv(ROOT / 'tables/spearman_correlations.csv')
fig, ax = plt.subplots(figsize=(7.6, 5))
sns.heatmap(pearson.set_axis(LABELS).set_axis(LABELS, axis=1), annot=True, fmt='.2f',
            cmap='vlag', vmin=-1, vmax=1, center=0, square=True,
            cbar_kws={'label':'Pearson r'}, ax=ax)
ax.set_title('Pooled correlations of body measurements (n = 342)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
savefig('05_correlation')
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.scatterplot(data=measurements, x='flipper_length_mm', y='body_mass_g',
                hue='species', style='species', hue_order=ORDER, palette=COLORS, alpha=.8, ax=ax)
ax.set(xlabel='Flipper length (mm)', ylabel='Body mass (g)',
       title='Longer flippers are associated with greater body mass')
ax.legend(title='Species')
savefig('06_flipper_mass')
rows=[]
for name, group in [('Pooled', measurements)] + list(measurements.groupby('species')):
    rows.append({'group': name, 'n':len(group),
        'flipper_mass_r':group.flipper_length_mm.corr(group.body_mass_g),
        'depth_mass_r':group.bill_depth_mm.corr(group.body_mass_g),
        'flipper_mass_spearman':group.flipper_length_mm.corr(group.body_mass_g, method='spearman')})
corr_groups = pd.DataFrame(rows)
corr_groups.to_csv(ROOT / 'tables/group_correlations.csv', index=False)
display(corr_groups.round(3))
fig, ax = plt.subplots(figsize=(8, 4))
bars=ax.bar(corr_groups.group, corr_groups.depth_mass_r, color=['#666666']+[COLORS[s] for s in corr_groups.group[1:]])
ax.bar_label(bars, fmt='%.3f', padding=4)
ax.axhline(0, color='black', linewidth=.8)
ax.set(ylim=(-.65,.9), xlabel='Analysis group', ylabel='Pearson r',
       title='Bill depth and mass: pooled association reverses within species')
savefig('07_correlation_reversal')

flags = pd.DataFrame(False, index=measurements.index, columns=NUM)
thresholds=[]
for species, group in measurements.groupby('species'):
    for col in NUM:
        q1, q3 = group[col].quantile([.25,.75])
        lower, upper = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
        flags.loc[group.index,col] = (group[col] < lower) | (group[col] > upper)
        thresholds.append([species,col,lower,upper,int(flags.loc[group.index,col].sum())])
thresholds = pd.DataFrame(thresholds, columns=['species','variable','lower','upper','flagged_count'])
thresholds.to_csv(ROOT / 'tables/iqr_thresholds.csv',index=False)
flagged = measurements.loc[flags.any(axis=1)].copy()
flagged.insert(0,'source_row',flagged.index+1)
flagged['flagged_fields'] = flags.loc[flagged.index].apply(lambda r: ', '.join(r.index[r]), axis=1)
flagged.to_csv(ROOT / 'tables/flagged_observations.csv',index=False)
display(flagged)
sensitivity = pd.concat([measurements.groupby('species').body_mass_g.mean().rename('all_mean_g'),
    measurements.loc[~flags.any(axis=1)].groupby('species').body_mass_g.mean().rename('without_flagged_mean_g')],axis=1)
sensitivity['change_g'] = sensitivity.without_flagged_mean_g-sensitivity.all_mean_g
sensitivity.to_csv(ROOT / 'tables/outlier_sensitivity.csv')
display(sensitivity.round(2))
print('Flagged unique records:', len(flagged))

missing_sensitivity = pd.concat([df.groupby('species').body_mass_g.mean().rename('available_mass_mean_g'),
    df.dropna().groupby('species').body_mass_g.mean().rename('complete_case_mean_g')],axis=1)
missing_sensitivity['change_g'] = missing_sensitivity.complete_case_mean_g-missing_sensitivity.available_mass_mean_g
missing_sensitivity.to_csv(ROOT / 'tables/missingness_sensitivity.csv')
display(missing_sensitivity.round(2))
annual = df.groupby(['species','year']).body_mass_g.agg(['count','mean','median']).reset_index()
annual.to_csv(ROOT / 'tables/annual_mass_summary.csv',index=False)
display(annual.round(2))
fig, ax = plt.subplots(figsize=(8, 4))
for species in ORDER:
    g=annual[annual.species==species]
    ax.plot(g.year,g['mean'],marker='o',label=species,color=COLORS[species])
    for _, row in g.iterrows():
        ax.annotate(f'n={int(row["count"])}',(row.year,row['mean']),xytext=(0,-17 if species == 'Adelie' else 9),textcoords='offset points',ha='center',fontsize=8)
ax.set(xlabel='Sampling year',ylabel='Mean body mass (g)',xticks=[2007,2008,2009],
       ylim=(3350,5400), title='Annual sample means vary without a common species trend')
ax.legend(title='Species',loc='center left',bbox_to_anchor=(1, .5))
savefig('08_annual_mass')

versions = {'python':platform.python_version(),'pandas':pd.__version__,
            'numpy':np.__version__,'matplotlib':plt.matplotlib.__version__,'seaborn':sns.__version__}
(ROOT / 'tables/software_versions.json').write_text(json.dumps(versions,indent=2))
print(versions)
print('Analysis complete. Figures and tables saved locally; original CSV unchanged.')
