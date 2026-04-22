import os
import pandas as pd
import re
import argparse

parser = argparse.ArgumentParser(description="Summarize Web Appendix C.2 AR(2) LDS simulation result CSVs.")
parser.add_argument(
    "--results_folder",
    default=os.path.join("sim_results", "sim_0318_results_LDSdiag_cmp_ar2"),
    help="Folder containing per-run results_*.csv files.",
)
parser.add_argument(
    "--output_file",
    default=None,
    help="Output CSV. Defaults to <results_folder>/summary/ALL_runs_combined.csv.",
)
args = parser.parse_args()

results_folder = args.results_folder
summary_folder = os.path.join(results_folder, "summary")
output_file = args.output_file or os.path.join(summary_folder, "ALL_runs_combined.csv")


def parse_filename(filename):
    """
    Parse AR2 LDS comparison result filenames such as:
    results_LDSdiagAR2_J10_p50_n1000_d14_d24_phi10.5_phi20.0_r0.1_sd1_kmgp_rbf_single_ell_initalgo2_cs_iter50000.csv
    """
    float_pat = r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?"
    pattern = (
        r"^results_LDSdiagAR2_"
        r"J(?P<J>\d+)_"
        r"p(?P<p>\d+)_"
        r"n(?P<n>\d+)_"
        r"d1(?P<d1>\d+)_"
        r"d2(?P<d2>\d+)_"
        r"phi1(?P<phi1>" + float_pat + r")_"
        r"phi2(?P<phi2>" + float_pat + r")_"
        r"r(?P<r>" + float_pat + r")_"
        r"sd(?P<sd>\d+)_"
        r"km(?P<km>.+?)_"
        r"init(?P<init>.+?)_"
        r"iter(?P<iter>\d+)\.csv$"
    )

    match = re.match(pattern, filename)
    if not match:
        return None

    params = match.groupdict()
    int_keys = ["J", "p", "n", "d1", "d2", "sd", "iter"]
    float_keys = ["phi1", "phi2", "r"]
    for key in int_keys:
        params[key] = int(params[key])
    for key in float_keys:
        params[key] = float(params[key])
    return params


def normalize_method(method_name):
    m_raw = str(method_name)
    if "HPPCA" in m_raw:
        return "HPPCA"
    if "PPCA_iid" in m_raw:
        return "PPCA_iid"
    if "mFPCA_covariance" in m_raw:
        return "mFPCA_covariance"
    if "mFPCA_inner-product" in m_raw:
        return "mFPCA_inner-product"
    return m_raw


# --- 2. Iterate and combine ---
all_rows = []
os.makedirs(summary_folder, exist_ok=True)

print("Processing files...")
for filename in sorted(os.listdir(results_folder)):
    if not (filename.endswith(".csv") and filename.startswith("results_")):
        continue

    file_params = parse_filename(filename)
    if not file_params:
        print(f"Skipping unmatched filename: {filename}")
        continue

    file_path = os.path.join(results_folder, filename)
    try:
        # Each CSV contains method, mse_missing, time_sec.
        df_content = pd.read_csv(file_path)
        for _, row in df_content.iterrows():
            combined_row = {**file_params, **row.to_dict()}
            if "method" in combined_row:
                combined_row["method"] = normalize_method(combined_row["method"])
            all_rows.append(combined_row)
    except Exception as e:
        print(f"Error parsing file {filename}: {e}")

# --- 3. Save combined results ---
final_df = pd.DataFrame(all_rows)
final_df.to_csv(output_file, index=False)

print(f"Done! Combined {len(final_df)} rows.")
print(f"Saved to: {output_file}")
