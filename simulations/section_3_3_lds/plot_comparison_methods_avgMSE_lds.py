import math
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

parser = argparse.ArgumentParser(description="Plot Section 3.3 LDS MSE comparison.")
parser.add_argument(
    "--input_csv",
    default=os.path.join("sim_results", "sim_LDSdiag_cmp", "summary", "ALL_runs_combined.csv"),
    help="Combined CSV produced by summarize_results_lds_diag.py.",
)
parser.add_argument(
    "--output_png",
    default=os.path.join("sim_results", "sim_LDSdiag_cmp", "summary", "MSE_Comparison_Standard_lds_J10.png"),
    help="Path for the output figure.",
)
args = parser.parse_args()

df = pd.read_csv(args.input_csv)

# Clean method names and set plotting order.
method_map = {
    "HPPCA": "HPPCA",
    "PPCA_iid": "PPCA",
    "mFPCA_covariance": "mFPCA (cov)",
    "mFPCA_inner-product": "mFPCA (inner)"
}
df['method'] = df['method'].map(method_map)
method_order = ["HPPCA", "PPCA", "mFPCA (cov)", "mFPCA (inner)"]
df['method'] = pd.Categorical(df['method'], categories=method_order, ordered=True)

# Sort to keep facet order stable.
df = df.sort_values(['r', 'J', 'rho'])
df = df[~df['J'].isin([5])] ## remove too high missing

def format_label(x):
    return f"$p_{{miss}}$={x['r']}, rho={x['rho']}"

df['facet_label'] = df.apply(format_label, axis=1)
facets = df['facet_label'].unique()

n_rows = 3
n_cols = math.ceil(len(facets) / n_rows)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 5), sharey=True)
axes = axes.flatten()

for i, label in enumerate(facets):
    ax = axes[i]
    subset = df[df['facet_label'] == label]
    
    sns.boxplot(data=subset, x='method', y='mse_missing', ax=ax, 
                order=method_order, palette="Set2", fliersize=1)
    
    ax.set_title(label, fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    plt.setp(ax.get_xticklabels(), ha='right')

fig.text(0.02, 0.5, 'MSE (Missing)', va='center', rotation='vertical', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])

output_dir = os.path.dirname(args.output_png)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
plt.savefig(args.output_png, bbox_inches='tight', dpi=600)
plt.show()
