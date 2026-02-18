#!/usr/bin/env Rscript

# Build a deterministic (no fuzzy matching) crosswalk:
# canonical schools (data/registries/teams.csv) -> baseballr team_id/team_name (from baseballr repo).

args <- commandArgs(trailingOnly = TRUE)

get_value <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (!is.na(idx) && idx < length(args)) return(args[[idx + 1]])
  default
}

year <- as.integer(get_value("--year", "2024"))
canon_path <- get_value("--canon", "data/registries/teams.csv")
out_path <- get_value("--out", "data/registries/name_crosswalk_baseballr.csv")

if (!requireNamespace("baseballr", quietly = TRUE)) {
  stop("Missing R package: baseballr. Run: Rscript scripts/install_R_deps.R")
}
if (!requireNamespace("readr", quietly = TRUE) || !requireNamespace("dplyr", quietly = TRUE)) {
  stop("Missing R packages: readr/dplyr. Run: Rscript scripts/install_R_deps.R")
}

suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(baseballr))

canon <- read_csv(canon_path, show_col_types = FALSE) %>%
  dplyr::select(id, school, conference)

teams <- load_ncaa_baseball_teams()
teams_year <- teams %>%
  dplyr::filter(.data$year == .env$year, .data$division == 1) %>%
  dplyr::select(team_id, team_name, conference, conference_id, season_id)

if (nrow(teams_year) == 0) stop(paste0("No baseballr teams for year=", year))

# Explicit overrides (manual, deterministic) for known display-name mismatches.
overrides <- data.frame(
  school = c(
    "Arizona State",
    "Florida State",
    "Kansas State",
    "Michigan State",
    "Mississippi State",
    "Miami",
    "Ohio State",
    "Oklahoma State",
    "Oregon State",
    "Penn State",
    "Pitt",
    "USC",
    "Cal",
    "Dallas Baptist"
  ),
  baseballr_team_name = c(
    "Arizona St.",
    "Florida St.",
    "Kansas St.",
    "Michigan St.",
    "Mississippi St.",
    "Miami (FL)",
    "Ohio St.",
    "Oklahoma St.",
    "Oregon St.",
    "Penn St.",
    "Pittsburgh",
    "Southern California",
    "California",
    "DBU"
  ),
  match_type = rep("manual_override", 14),
  stringsAsFactors = FALSE
)

canon2 <- canon %>%
  dplyr::left_join(overrides, by = "school") %>%
  dplyr::mutate(lookup_name = dplyr::if_else(!is.na(.data$baseballr_team_name), .data$baseballr_team_name, .data$school),
                match_type = dplyr::if_else(!is.na(.data$match_type), .data$match_type, "exact_school"))

joined <- canon2 %>%
  dplyr::left_join(teams_year, by = c("lookup_name" = "team_name"), suffix = c("_canon", "_baseballr")) %>%
  dplyr::transmute(
    canonical_id = .data$id,
    canonical_school = .data$school,
    canonical_conference = .data$conference_canon,
    baseballr_team_id = .data$team_id,
    baseballr_team_name = .data$lookup_name,
    baseballr_conference = .data$conference_baseballr,
    baseballr_conference_id = .data$conference_id,
    baseballr_season_id = .data$season_id,
    match_type = .data$match_type,
    status = dplyr::if_else(is.na(.data$team_id), "unmatched", "matched"),
    notes = dplyr::if_else(
      is.na(.data$team_id),
      "Not found in baseballr D1 team registry for this year (possible no varsity program).",
      ""
    )
  ) %>%
  dplyr::arrange(.data$status, .data$canonical_conference, .data$canonical_school)

write_csv(joined, out_path)
message("Wrote crosswalk: ", out_path)

unmatched <- joined %>% dplyr::filter(.data$status == "unmatched")
if (nrow(unmatched) > 0) {
  message("Unmatched canonical schools (likely no NCAA D1 baseball program or baseballr registry gap):")
  message(paste0("  - ", unmatched$canonical_school, collapse = "\n"))
}
