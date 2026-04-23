import os
import pandas as pd
import re
import argparse

parser = argparse.ArgumentParser(description="Summarize Section 3.3 LDS simulation result CSVs.")
parser.add_argument(
    "--results_folder",
    default=os.path.join("sim_results", "sim_LDSdiag_cmp"),
    help="Folder containing per-run results_*.csv files.",
)
parser.add_argument(
    "--output_file",
    default=None,
    help="Output CSV. Defaults to <results_folder>/summary/ALL_runs_combined.csv.",
)
args = parser.parse_args()

results_folder = args.results_folder
output_file = args.output_file or os.path.join(results_folder, "summary", "ALL_runs_combined.csv")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

def parse_filename(filename):
    """
    Parse LDS result filenames such as:
    results_LDSdiag_J5_p50_n1000_d12_d22_rho0.3_slow0.99_r0.1_sd1_kmgp_rbf_single_ell_initalgo2_cs_iter50000.csv
    """
    pattern = r"J(?P<J>\d+)_p(?P<p>\d+)_n(?P<n>\d+)_d1(?P<d1>\d+)_d2(?P<d2>\d+)_rho(?P<rho>[\d\.]+)_slow(?P<slow>[\d\.]+)_r(?P<r>[\d\.]+)_sd(?P<sd>\d+)_km(?P<km>.+?)_initalgo"
    
    match = re.search(pattern, filename)
    if match:
        params = match.groupdict()
        for key in ['J', 'p', 'n', 'd1', 'd2', 'rho', 'slow', 'r', 'sd']:
            params[key] = float(params[key]) if '.' in params[key] else int(params[key])
        return params
    return None

all_rows = []

print("Processing files...")
for filename in os.listdir(results_folder):
    if filename.endswith(".csv") and filename.startswith("results_"):
        file_params = parse_filename(filename)
        if not file_params:
            continue
            
        file_path = os.path.join(results_folder, filename)
        try:
            df_content = pd.read_csv(file_path)
            
            for _, row in df_content.iterrows():
                combined_row = {**file_params, **row.to_dict()}
                
                m_raw = str(combined_row['method'])
                if "HPPCA" in m_raw:
                    combined_row['method'] = "HPPCA"
                elif "PPCA_iid" in m_raw:
                    combined_row['method'] = "PPCA_iid"
                elif "mFPCA_covariance" in m_raw:
                    combined_row['method'] = "mFPCA_covariance"
                elif "mFPCA_inner-product" in m_raw:
                    combined_row['method'] = "mFPCA_inner-product"
                
                all_rows.append(combined_row)
        except Exception as e:
            print(f"Error parsing file {filename}: {e}")

final_df = pd.DataFrame(all_rows)
final_df.to_csv(output_file, index=False)

print(f"Done! Combined {len(final_df)} rows.")
print(f"Saved to: {output_file}")
