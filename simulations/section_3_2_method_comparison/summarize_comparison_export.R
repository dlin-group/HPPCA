#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(purrr)
  library(stringr)
})

# Root results directory holding CSVs from comparison runs.
results_dir <- file.path("sim_results", "sim_comparison_section_3_2")
summary_dir <- file.path(results_dir, "summary")
dir.create(summary_dir, showWarnings = FALSE, recursive = TRUE)

out_raw     <- file.path(summary_dir, "ALL_runs_combined.csv")

message("Reading CSVs from: ", results_dir)

# Collect all result CSVs
csv_files <- list.files(results_dir, pattern = "\\.csv$", full.names = TRUE, recursive = TRUE)
if (length(csv_files) == 0) {
  stop("No result CSVs found in ", results_dir)
}

read_one <- function(fp) {
  message("Reading file: ", fp)
  df <- suppressMessages(readr::read_csv(fp, show_col_types = FALSE))
  # Force consistent types across files before binding
  ## get n, J, p, d1, d2, r, s, km, ell, sd from filename
  # km should be gp_iid or gp_rbf_single_ell
  fn <- basename(fp)
    pattern <- paste0(
        "compare_metrics_partial_",
        "n(\\d+)_",
        "J(\\d+)_",
        "p(\\d+)_",
        "d1(\\d+)_",
        "d2(\\d+)_",
        "r([0-9\\.]+)_",
        "s([0-9\\.]+)_",
        "km(|gp_iid|gp_rbf_single_ell)_",
        "ell([0-9\\.]+)_",
        "sd([0-9\\.]+)\\.csv"
    )
    matches <- str_match(fn, pattern)
    if (is.na(matches[1,1])) {
      stop("Filename does not match expected pattern: ", fn)
    }
    n <- matches[1,2]
    J <- matches[1,3]
    p <- matches[1,4]
    d1 <- matches[1,5]
    d2 <- matches[1,6]
    r <- matches[1,7]
    s <- matches[1,8]
    km <- matches[1,9]
    ell <- matches[1,10]
    sd <- matches[1,11]
    print(paste("Extracted params:", n, J, p, d1, d2, r, s, km, ell, sd))
    df <- df %>%
        mutate(
        n = as.integer(n),
        J = as.integer(J),
        p = as.integer(p),
        d1 = as.integer(d1),
        d2 = as.integer(d2),
        r = as.numeric(r),
        s = as.numeric(s),
        km = as.character(km),
        ell = as.numeric(ell),
        sd = as.numeric(sd)
    )
}
df_all <- purrr::map_dfr(csv_files, read_one)

# Write combined raw rows
readr::write_csv(df_all, out_raw)
message("Wrote combined rows to: ", out_raw)


## find out the missing seeds for r = 0.7, J = 10 for each kernel method (suppose the seed should exist from 1 to 100)
missing_seeds <- df_all %>%
  filter(r == 0.7, J == 10) %>%
  group_by(n, p, d1, d2, s, km, ell, method) %>%
  summarize(
    existing_seeds = list(unique(seed)),
    .groups = "drop"
  ) %>%
  mutate(
    missing_seeds = map(existing_seeds, ~ setdiff(1:100, .x))
  ) %>%
  select(-existing_seeds)

print("Missing seeds for r = 0.7, J = 10:")
print(missing_seeds)


summary_dir <- file.path(results_dir, "summary")
out_raw     <- file.path(summary_dir, "ALL_runs_combined.csv")

## read this to plot comparison of avg MSE across methods

all_results_df <- read.csv(out_raw)

hppca_summary_file <- file.path(results_dir, "summary", "ALL_hppca_runs_combined_hppca_correct.csv")
hppca_mse_all_runs <- read.csv(hppca_summary_file)
hppca_mse_all_runs <- hppca_mse_all_runs %>%
  mutate(
    method = "HPPCA"
  ) %>%
  filter(missing_rate != 0) %>%
  select(method, seed, num_missing_entries, mse_missing_hppca, n_orig, J, p, d1, d2, missing_rate, sigma2_true, kernel_method, ell_true_sim)

## in hppca_mse_all_runs, if ell_true_sim is not none, it is a vector like number;number;... replace it with the number before the first ;
hppca_mse_all_runs <- hppca_mse_all_runs %>%
  mutate(
    ell_true_sim = ifelse(grepl(";", ell_true_sim), as.numeric(sub(";.*", "", ell_true_sim)), as.numeric(ell_true_sim))
  )

colnames(hppca_mse_all_runs) <- colnames(all_results_df)

## in all_results_df, if km is gp_iid, replace ell with NA
all_results_df <- all_results_df %>%
  select(-sd) %>%
  mutate(
    ell = ifelse(km == "gp_iid", NA, ell)
  )

all_results_df <- rbind(all_results_df, hppca_mse_all_runs)
# Write combined raw rows
readr::write_csv(all_results_df, out_raw)
message("Wrote combined rows to: ", out_raw)

### add how many rows per group
summary <- all_results_df %>%
  group_by(n, J, p, d1, d2, r, s, km, ell, method) %>%
  summarize(
    across(
      where(is.numeric),
      list(
        n = ~ n(),
        mean = ~ mean(.x, na.rm = TRUE),
        sd = ~ sd(.x, na.rm = TRUE)
      ),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  ) %>%
  readr::write_csv(file.path(summary_dir, "ALL_runs_summary.csv"))
