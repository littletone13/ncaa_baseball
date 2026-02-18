#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_flag <- function(flag) {
  any(args == flag)
}

get_value <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (!is.na(idx) && idx < length(args)) return(args[[idx + 1]])
  default
}

parse_seasons <- function(x) {
  if (is.null(x) || nchar(x) == 0) stop("Missing --seasons (e.g. 2021,2022,2023)")
  x <- gsub("\\s+", "", x)
  if (grepl("^[0-9]{4}-[0-9]{4}$", x)) {
    parts <- strsplit(x, "-", fixed = TRUE)[[1]]
    return(seq.int(as.integer(parts[[1]]), as.integer(parts[[2]])))
  }
  as.integer(strsplit(x, ",", fixed = TRUE)[[1]])
}

seasons <- parse_seasons(get_value("--seasons", NULL))
out_dir <- get_value("--out-dir", "data/raw/pbp")
incomplete_dir <- get_value("--incomplete-dir", file.path(out_dir, "incomplete"))
overwrite <- get_flag("--overwrite")
min_rows <- as.integer(get_value("--min-rows", "100000"))
allow_small <- get_flag("--allow-small")

if (!requireNamespace("baseballr", quietly = TRUE)) {
  stop("Missing R package: baseballr. Run: Rscript scripts/install_R_deps.R")
}

write_parquet_ok <- requireNamespace("arrow", quietly = TRUE)
if (!write_parquet_ok && !requireNamespace("readr", quietly = TRUE)) {
  stop("Need either arrow (parquet) or readr (csv). Run: Rscript scripts/install_R_deps.R")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Missing R package: jsonlite. Run: Rscript scripts/install_R_deps.R")
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(incomplete_dir, recursive = TRUE, showWarnings = FALSE)

for (season in seasons) {
  message("PBP season: ", season)
  pbp <- NULL

  if (!exists("load_ncaa_baseball_pbp", where = asNamespace("baseballr"), inherits = FALSE)) {
    stop("baseballr::load_ncaa_baseball_pbp not found (package API changed?)")
  }

  pbp <- baseballr::load_ncaa_baseball_pbp(seasons = season)

  if (!allow_small && nrow(pbp) < min_rows) {
    message(
      "  WARNING: only ", nrow(pbp), " rows (min_rows=", min_rows, "); quarantining as incomplete."
    )
  }

  manifest <- list(
    fetched_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
    season = season,
    n_rows = nrow(pbp),
    n_cols = ncol(pbp),
    min_rows = min_rows,
    status = if (!allow_small && nrow(pbp) < min_rows) "incomplete" else "ok"
  )

  base_dir <- if (!allow_small && nrow(pbp) < min_rows) incomplete_dir else out_dir

  if (write_parquet_ok) {
    out_path <- file.path(base_dir, sprintf("pbp_%d.parquet", season))
    if (!overwrite && file.exists(out_path)) {
      message("  exists, skipping: ", out_path)
    } else {
      arrow::write_parquet(pbp, out_path)
      message("  wrote: ", out_path)
    }
  } else {
    out_path <- file.path(base_dir, sprintf("pbp_%d.csv.gz", season))
    if (!overwrite && file.exists(out_path)) {
      message("  exists, skipping: ", out_path)
    } else {
      readr::write_csv(pbp, out_path)
      message("  wrote: ", out_path)
    }
  }

  manifest_path <- file.path(base_dir, sprintf("pbp_%d.manifest.json", season))
  jsonlite::write_json(manifest, manifest_path, pretty = TRUE, auto_unbox = TRUE)
}
