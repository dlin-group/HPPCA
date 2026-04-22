import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

parser = argparse.ArgumentParser(description="Plot Section 3.2 average MSE comparison.")
parser.add_argument(
    "--input_csv",
    default=os.path.join("sim_results", "sim_comparison_section_3_2", "summary", "ALL_runs_combined.csv"),
    help="Combined CSV produced by summarize_comparison_export.R.",
)
parser.add_argument(
    "--output_png",
    default=os.path.join("sim_results", "sim_comparison_section_3_2", "summary", "MSE_Comparison_Broken_Axis.png"),
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
df = df[~df['r'].isin([0.7, 0.9])] ## remove too high missing
# For gp_rbf_single_ell, keep the ell=10 scenario only.
mask = (df['km'] != "gp_rbf_single_ell") | (df['ell'] == 10)
df = df[mask].copy()
df['method'] = df['method'].map(method_map)
method_order = ["HPPCA", "PPCA", "mFPCA (cov)", "mFPCA (inner)"]
df['method'] = pd.Categorical(df['method'], categories=method_order, ordered=True)
df = df.sort_values(['r', 'J', 'km'])
def format_label(x):
    base = f"J={int(x['J'])}, $p_{{miss}}$={x['r']}"
    
    if x['km'] == "gp_iid":
        return f"{base}\nKernel=IID"
    else:
        return f"{base}\nKernel=RBF, $\ell$={int(x['ell'])}"

df['facet_label'] = df.apply(format_label, axis=1)
facets = df['facet_label'].unique()
n_cols = 4
n_rows = (len(facets) + n_cols - 1) // n_cols

fig = plt.figure(figsize=(15, 6 * n_rows))
y_low_lim = [0.25, 0.4]
y_high_lim = [0.8, 5.5]

for i, label in enumerate(facets):
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.3, hspace=0.5)
    inner_gs = gs[i // n_cols, i % n_cols].subgridspec(2, 1, height_ratios=[1, 2.5], hspace=0.08)
    
    ax_top = fig.add_subplot(inner_gs[0])
    ax_bottom = fig.add_subplot(inner_gs[1], sharex=ax_top)
    
    subset = df[df['facet_label'] == label]
    
    sns.boxplot(data=subset, x='method', y='mse_missing', ax=ax_top, 
                order=method_order, palette="Set2", fliersize=1)
    sns.boxplot(data=subset, x='method', y='mse_missing', ax=ax_bottom, 
                order=method_order, palette="Set2", fliersize=1)
    
    ax_top.set_xlabel("")
    ax_top.xaxis.set_visible(False) 
    
    ax_bottom.set_xlabel("") 
    
    ax_top.set_ylim(y_high_lim)
    ax_bottom.set_ylim(y_low_lim)
    
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    
    ax_top.tick_params(labelbottom=False, bottom=False)
    
    d = .015
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, lw=1)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    ax_top.set_title(label, fontsize=14)
    ax_top.set_ylabel("")
    ax_bottom.set_ylabel("")
    ax_bottom.set_xlabel("")

    ax_top.tick_params(axis='y', labelsize=12)
    ax_bottom.tick_params(axis='y', labelsize=12)
    ax_bottom.tick_params(axis='x', labelsize=12)

    plt.xticks(rotation=45)

fig.text(0.04, 0.5, 'MSE (Missing)', va='center', rotation='vertical', fontsize=14, fontweight='bold')

output_dir = os.path.dirname(args.output_png)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
plt.savefig(args.output_png, bbox_inches='tight')
plt.show()
