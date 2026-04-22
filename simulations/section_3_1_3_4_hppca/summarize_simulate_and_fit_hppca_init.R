#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(purrr)
  library(stringr)
})

# Root results directory holding CSVs from both IID and kernel runs.
results_dir <- file.path("sim_results", "results_hppca_init")
summary_dir <- file.path(results_dir, "summary")
dir.create(summary_dir, showWarnings = FALSE, recursive = TRUE)

out_raw     <- file.path(summary_dir, "ALL_hppca_runs_combined.csv")
out_summary <- file.path(summary_dir, "SUMMARY_by_kernel_nJpd_mr_sigma2_ell_tol.csv")

message("Reading CSVs from: ", results_dir)

# Collect all result CSVs
csv_files <- list.files(results_dir, pattern = "_results\\.csv$", full.names = TRUE, recursive = TRUE)
if (length(csv_files) == 0) {
  stop("No result CSVs found in ", results_dir)
}

read_one <- function(fp) {
  message("Reading file: ", fp)
  df <- suppressMessages(readr::read_csv(fp, show_col_types = FALSE))
  # Force consistent types across files before binding
  if ("init_method" %in% names(df))  df$init_method  <- as.character(df$init_method)
  if ("kernel_method" %in% names(df)) df$kernel_method <- as.character(df$kernel_method)
  if ("ell_init" %in% names(df))      df$ell_init      <- as.character(df$ell_init)
  if ("ell_final" %in% names(df))     df$ell_final     <- as.character(df$ell_final)
  if ("ell_true_sim" %in% names(df))  df$ell_true_sim  <- as.character(df$ell_true_sim)
  if ("converged" %in% names(df)) {
    if (is.numeric(df$converged)) {
      df$converged <- df$converged != 0
    } else {
      df$converged <- tolower(as.character(df$converged)) %in% c("true","t","1")
    }
  }
  df
}
df_all <- purrr::map_dfr(csv_files, read_one)

# Coerce/clean columns used below
num_cols <- c(
  "n_orig","n_eff","J","p","d1","d2","missing_rate",
  "sigma2_final","sigma2_true","iteration_num","total_time","seed","tolerance","n_cpus",
  "hppca_matrix_rank",
  "sum_angles_W1_vs_true_deg","max_angle_W1_vs_true_deg",
  "sum_angles_W2_vs_true_deg","max_angle_W2_vs_true_deg"
)

bool_cols <- c("converged")

df_all <- df_all %>%
  mutate(
    across(intersect(names(.), num_cols), ~ suppressWarnings(as.numeric(.))),
    across(intersect(names(.), bool_cols), ~ as.logical(.)),
    kernel_method = as.character(kernel_method),
    ell_init = as.character(ell_init),
    ell_final = as.character(ell_final),
    ell_true_sim = as.character(ell_true_sim)
  )

# Helper to parse semicolon-separated ell strings; return sorted numeric vector or numeric(0) for None
parse_ell_sorted <- function(x) {
  if (is.null(x) || is.na(x) || x == "None" || trimws(x) == "") return(numeric(0))
  parts <- strsplit(x, ";", fixed = TRUE)[[1]]
  vals <- suppressWarnings(as.numeric(parts))
  vals <- vals[!is.na(vals)]
  if (length(vals) == 0) return(numeric(0))
  sort(vals)
}

# Attach list-columns for vector parameters and per-row bias vectors
df_all <- df_all %>%
  mutate(
    ell_final_vec = map(ell_final, parse_ell_sorted),
    ell_true_vec  = map(ell_true_sim, parse_ell_sorted),
    ell_bias_vec  = map2(ell_final_vec, ell_true_vec, function(a, b) {
      if (length(a) == 0 || length(b) == 0 || length(a) != length(b)) return(numeric(0))
      a - b
    })
  )

# Write combined raw rows
readr::write_csv(df_all, out_raw)
message("Wrote combined rows to: ", out_raw)


## read this file

df_all <- readr::read_csv(out_raw, show_col_types = FALSE)

# Vector stats helpers: return semicolon-joined strings or "None"
fmt_vec <- function(v, digits = 6) paste0(formatC(v, format = "f", digits = digits), collapse = ";")
parse_ell_scalar <- function(x) {
  vals <- parse_ell_sorted(x)
  if (length(vals) == 0) return(NA_real_)
  vals[[1]]
}
safe_mean <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_real_)
  mean(x)
}
safe_sd <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) <= 1) return(NA_real_)
  stats::sd(x)
}
vec_mean_str <- function(lst) {
  good <- lengths(lst) > 0
  if (!any(good)) return("None")
  M <- do.call(rbind, lst[good])
  fmt_vec(colMeans(M, na.rm = TRUE))
}
vec_sd_str <- function(lst) {
  good <- lengths(lst) > 0
  if (!any(good)) return("None")
  M <- do.call(rbind, lst[good])
  fmt_vec(apply(M, 2, sd, na.rm = TRUE))
}

# Reconstruct ell columns from the combined CSV after reloading.
df_all <- df_all %>%
  mutate(
    ell_init = as.character(ell_init),
    ell_final = as.character(ell_final),
    ell_true_sim = as.character(ell_true_sim),
    ell_init_num = map_dbl(ell_init, parse_ell_scalar),
    ell_final_num = map_dbl(ell_final, parse_ell_scalar),
    ell_init_vec = map(ell_init, parse_ell_sorted),
    ell_final_vec = map(ell_final, parse_ell_sorted),
    ell_true_vec = map(ell_true_sim, parse_ell_sorted),
    ell_true_first = map_dbl(ell_true_sim, parse_ell_scalar),
    ell_final_bias_num = ell_final_num - ell_true_first,
    ell_init_bias_num = ell_init_num - ell_true_first,
    ell_bias_vec = map2(ell_final_vec, ell_true_vec, function(a, b) {
      if (length(a) == 0 || length(b) == 0 || length(a) != length(b)) return(numeric(0))
      a - b
    })
  )

# Group and summarise
group_keys <- c("init_method","kernel_method","n_orig","J","p","d1","d2","missing_rate","sigma2_true","ell_true_sim","tolerance")

df_summary <- df_all %>%
  group_by(across(all_of(group_keys))) %>%
  summarise(
    n_runs  = n(),
    conv_rate      = mean(converged, na.rm = TRUE),

    sigma2_final_mean = mean(sigma2_final, na.rm = TRUE),
    sigma2_final_sd   = sd(sigma2_final, na.rm = TRUE),
    sigma2_bias_mean  = mean(sigma2_final - sigma2_true, na.rm = TRUE),

    sigma2_init_mean = mean(sigma2_init, na.rm = TRUE),
    sigma2_init_sd   = sd(sigma2_init, na.rm = TRUE),

    iteration_num_mean = mean(iteration_num, na.rm = TRUE),
    iteration_num_sd   = sd(iteration_num, na.rm = TRUE),
    total_time_mean    = mean(total_time, na.rm = TRUE),
    total_time_sd      = sd(total_time, na.rm = TRUE),

    hppca_rank_mean = mean(hppca_matrix_rank, na.rm = TRUE),
    hppca_rank_sd   = sd(hppca_matrix_rank, na.rm = TRUE),

    sum_angles_W1_mean = mean(sum_angles_W1_vs_true_deg, na.rm = TRUE),
    sum_angles_W1_sd   = sd(sum_angles_W1_vs_true_deg, na.rm = TRUE),
    max_angle_W1_mean  = mean(max_angle_W1_vs_true_deg, na.rm = TRUE),
    max_angle_W1_sd    = sd(max_angle_W1_vs_true_deg, na.rm = TRUE),
    sum_angles_W2_mean = mean(sum_angles_W2_vs_true_deg, na.rm = TRUE),
    sum_angles_W2_sd   = sd(sum_angles_W2_vs_true_deg, na.rm = TRUE),
    max_angle_W2_mean  = mean(max_angle_W2_vs_true_deg, na.rm = TRUE),
    max_angle_W2_sd    = sd(max_angle_W2_vs_true_deg, na.rm = TRUE),

    sum_angles_W1_init_mean = mean(sum_angles_W1_init_vs_true_deg, na.rm = TRUE),
    sum_angles_W1_init_sd   = sd(sum_angles_W1_init_vs_true_deg, na.rm = TRUE),
    max_angle_W1_init_mean  = mean(max_angle_W1_init_vs_true_deg, na.rm = TRUE),
    max_angle_W1_init_sd    = sd(max_angle_W1_init_vs_true_deg, na.rm = TRUE),
    sum_angles_W2_init_mean = mean(sum_angles_W2_init_vs_true_deg, na.rm = TRUE),
    sum_angles_W2_init_sd   = sd(sum_angles_W2_init_vs_true_deg, na.rm = TRUE),
    max_angle_W2_init_mean  = mean(max_angle_W2_init_vs_true_deg, na.rm = TRUE),
    max_angle_W2_init_sd    = sd(max_angle_W2_init_vs_true_deg, na.rm = TRUE),

    # Ell summaries
    .kernel = dplyr::first(kernel_method),
    .ell_true_first = dplyr::first(ell_true_first),
    .is_single = grepl("single_ell", .kernel),

    ell_final_mean_single = if (.is_single) safe_mean(ell_final_num) else NA_real_,
    ell_final_sd_single   = if (.is_single) safe_sd(ell_final_num) else NA_real_,
    ell_bias_single       = if (.is_single) safe_mean(ell_final_bias_num) else NA_real_,

    ell_final_init_mean = if (.is_single) safe_mean(ell_init_num) else NA_real_,
    ell_final_init_sd   = if (.is_single) safe_sd(ell_init_num) else NA_real_,
    ell_bias_init       = if (.is_single) safe_mean(ell_init_bias_num) else NA_real_,

    ell_final_mean_vec = if (.is_single) NA_character_ else vec_mean_str(ell_final_vec),
    ell_final_sd_vec   = if (.is_single) NA_character_ else vec_sd_str(ell_final_vec),
    ell_bias_vec       = if (.is_single) NA_character_ else vec_mean_str(ell_bias_vec),

    .groups = "drop"
  ) %>%
  mutate(
    ell_final_mean = ifelse(grepl("single_ell", kernel_method),
                            as.character(ell_final_mean_single), ell_final_mean_vec),
    ell_final_sd   = ifelse(grepl("single_ell", kernel_method),
                            as.character(ell_final_sd_single),   ell_final_sd_vec),
    ell_bias       = ifelse(grepl("single_ell", kernel_method),
                            as.character(ell_bias_single),        ell_bias_vec)
  ) %>%
  select(-.kernel, -.ell_true_first, -.is_single,
         -ell_final_mean_single, -ell_final_sd_single, -ell_bias_single,
         -ell_final_mean_vec, -ell_final_sd_vec, -ell_bias_vec)

readr::write_csv(df_summary, out_summary)
message("Wrote summary to: ", out_summary)

message("Done.")
