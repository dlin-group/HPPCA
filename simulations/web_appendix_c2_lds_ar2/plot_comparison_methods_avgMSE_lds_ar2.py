import math
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

parser = argparse.ArgumentParser(description="Plot Web Appendix C.2 AR(2) LDS MSE comparison.")
parser.add_argument(
    "--input_csv",
    default=os.path.join("sim_results", "sim_0318_results_LDSdiag_cmp_ar2", "summary", "ALL_runs_combined.csv"),
    help="Combined CSV produced by summarize_results_lds_ar2.py.",
)
parser.add_argument(
    "--output_png",
    default=os.path.join(
        "sim_results",
        "sim_0318_results_LDSdiag_cmp_ar2",
        "summary",
        "MSE_Comparison_Standard_lds_ar2_phi1_0.8_phi2_0.1.png",
    ),
    help="Path for the output figure.",
)
args = parser.parse_args()

df = pd.read_csv(args.input_csv)

# 2. Keep only AR(2) scenario (phi1, phi2) = (0.8, 0.1)
df = df[np.isclose(df["phi1"], 0.8) & np.isclose(df["phi2"], 0.1)].copy()

# 3. Data cleaning and ordering
method_map = {
    "HPPCA": "HPPCA",
    "PPCA_iid": "PPCA",
    "mFPCA_covariance": "mFPCA (cov)",
    "mFPCA_inner-product": "mFPCA (inner)"
}
df["method"] = df["method"].map(method_map)
method_order = ["HPPCA", "PPCA", "mFPCA (cov)", "mFPCA (inner)"]
df = df[df["method"].notna()].copy()
df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)

# Sort by r, J, d1 to keep facet arrangement stable
df = df.sort_values(["r", "J", "d1"])

# 4. Facet labels and layout
def format_label(x):
    return f"$p_{{miss}}$={x['r']}, J={int(x['J'])}"

df["facet_label"] = df.apply(format_label, axis=1)
facets = df["facet_label"].unique()

n_rows = 3
n_cols = math.ceil(len(facets) / n_rows)

# 5. Plot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 5), sharey=True)
axes = np.array(axes).reshape(-1)

# Draw each facet
for i, label in enumerate(facets):
    ax = axes[i]
    subset = df[df["facet_label"] == label]

    sns.boxplot(data=subset, x="method", y="mse_missing", ax=ax,
                order=method_order, palette="Set2", fliersize=1)

    ax.set_title(label, fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    plt.setp(ax.get_xticklabels(), ha="right")

# Hide unused panels, if any
for j in range(len(facets), len(axes)):
    axes[j].axis("off")

#fig.suptitle("Standard LDS AR(2): phi1=0.8, phi2=0.1", fontsize=12, fontweight="bold", y=0.995)
fig.text(0.02, 0.5, "MSE (Missing)", va="center", rotation="vertical", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])

output_dir = os.path.dirname(args.output_png)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
plt.savefig(args.output_png, bbox_inches="tight", dpi=600)
plt.show()
