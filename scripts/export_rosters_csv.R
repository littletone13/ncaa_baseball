#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript scripts/export_rosters_csv.R <year>")
}
year <- as.integer(args[[1]])

ds <- arrow::open_dataset("data/raw/rosters", format = "parquet") %>% filter(.data$year == .env$year)
df <- ds %>% collect()

out_dir <- "data/processed/rosters"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_path <- file.path(out_dir, sprintf("rosters_%d.csv", year))
readr::write_csv(df, out_path)
cat(sprintf("Wrote %s (%d rows)\n", out_path, nrow(df)))

missing <- df %>% filter(is.na(.data$player_id) | is.na(.data$player_url))
missing_path <- file.path(out_dir, sprintf("rosters_%d_missing_player_id.csv", year))
readr::write_csv(missing, missing_path)
cat(sprintf("Wrote %s (%d rows)\n", missing_path, nrow(missing)))
