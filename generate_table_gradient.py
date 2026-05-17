"""
generate_table_gradient.py

Lit results_section2.csv, extrait les méthodes gradient-based
(GradBoost, XGBoost, LGBM, RFF) en mode 'vanilla'
et génère le code LaTeX d'un tableau de comparaison avec rang moyen.

Usage : python generate_table_gradient.py
Sortie : table_gradient.tex  (à \input{} dans le rapport)
"""

import pandas as pd
import numpy as np

CSV_PATH   = 'results_section2.csv'
OUTPUT_TEX = 'table_gradient.tex'

METHODS = ['GradBoost', 'XGBoost', 'LGBM', 'RFF_binary', 'RFF_OvR']
# Noms affichés dans le tableau
DISPLAY = {
    'GradBoost':  'GradBoost',
    'XGBoost':    'XGBoost',
    'LGBM':       'LGBM',
    'RFF_binary': 'RFF-GB',
    'RFF_OvR':    'RFF-GB (OvR)',
}

df = pd.read_csv(CSV_PATH)
df = df[df['Mode'] == 'vanilla']

# Garder uniquement les méthodes gradient
df = df[df['Algorithm'].isin(METHODS)].copy()

# Pour chaque dataset, conserver la meilleure variante RFF
# (binary si binaire, OvR si multiclasse) afin d'éviter les doublons
rows_out = []
for ds, grp in df.groupby('Dataset'):
    for m in METHODS:
        sub = grp[grp['Algorithm'] == m]
        if not sub.empty:
            rows_out.append(sub.iloc[0])
df_filt = pd.DataFrame(rows_out)

# Pivot : datasets en lignes, méthodes en colonnes
pivot_f1   = df_filt.pivot_table(index='Dataset', columns='Algorithm',
                                  values='F1_mean',  aggfunc='first')
pivot_std  = df_filt.pivot_table(index='Dataset', columns='Algorithm',
                                  values='F1_std',   aggfunc='first')
pivot_time = df_filt.pivot_table(index='Dataset', columns='Algorithm',
                                  values='Time_s',   aggfunc='first')

# Colonnes présentes (LGBM ou XGBoost peuvent être absents selon l'env)
cols_present = [c for c in METHODS if c in pivot_f1.columns]

# Calcul des rangs (rang 1 = meilleur F1)
pivot_rank = pivot_f1[cols_present].rank(axis=1, ascending=False, method='min')

# Génération LaTeX
n_methods = len(cols_present)
col_spec   = 'l' + 'r' * n_methods

header_methods = ' & '.join([f'\\textbf{{{DISPLAY.get(c,c)}}}' for c in cols_present])

lines = []
lines.append(r'\begin{table}[ht]')
lines.append(r'\centering')
lines.append(r'\small')
lines.append(r'\caption{F1-score moyen ($\pm$ std) sur 5 folds pour les méthodes'
             r' gradient-based (mode vanilla). \textbf{Gras} = meilleur score par dataset.}')
lines.append(r'\label{tab:gradient_methods}')
lines.append(r'\begin{tabular}{' + col_spec + r'}')
lines.append(r'\toprule')
lines.append(r'\textbf{Dataset} & ' + header_methods + r' \\')
lines.append(r'\midrule')

for ds in sorted(pivot_f1.index):
    row_vals = []
    # Récupérer le IR du dataset pour l'affichage
    ir_vals = df_filt[df_filt['Dataset'] == ds]['IR'].values
    ir_str  = f'{ir_vals[0]:.1f}' if len(ir_vals) > 0 else '?'
    ds_label = f'{ds} ({ir_str}\\%)'

    best_f1 = pivot_f1.loc[ds, cols_present].max()

    for c in cols_present:
        if c not in pivot_f1.columns or pd.isna(pivot_f1.loc[ds, c]):
            row_vals.append('---')
            continue
        f1  = pivot_f1.loc[ds, c]
        std = pivot_std.loc[ds, c] if c in pivot_std.columns else 0.0
        cell = f'{f1*100:.1f} $\\pm$ {std*100:.1f}'
        if abs(f1 - best_f1) < 1e-4:
            cell = f'\\textbf{{{cell}}}'
        row_vals.append(cell)

    lines.append(f'  {ds_label} & ' + ' & '.join(row_vals) + r' \\')

lines.append(r'\midrule')

# Ligne moyenne
avg_row = []
for c in cols_present:
    if c not in pivot_f1.columns:
        avg_row.append('---')
        continue
    m   = pivot_f1[c].mean()
    std = pivot_f1[c].std()
    avg_row.append(f'{m*100:.1f} $\\pm$ {std*100:.1f}')
lines.append(r'  \textbf{Moyenne} & ' + ' & '.join(avg_row) + r' \\')

# Ligne rang moyen
rank_row = []
for c in cols_present:
    if c not in pivot_rank.columns:
        rank_row.append('---')
        continue
    rank_row.append(f'{pivot_rank[c].mean():.2f}')
lines.append(r'  \textbf{Rang moyen} & ' + ' & '.join(rank_row) + r' \\')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

tex_content = '\n'.join(lines)

with open(OUTPUT_TEX, 'w', encoding='utf-8') as f:
    f.write(tex_content)

print(f"Tableau LaTeX sauvegardé → {OUTPUT_TEX}")
print(tex_content)
