#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_value <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (!is.na(idx) && idx < length(args)) return(args[[idx + 1]])
  default
}

season <- as.integer(get_value("--season", NA))
if (is.na(season)) stop("Missing --season (e.g. 2024)")

stop(paste0(
  "Blocked: team player stats pages are now served under /teams/<team_season_id>/... ",
  "and require an interstitial browser challenge (Akamai). ",
  "baseballr's legacy /team/<team_id>/stats endpoints return 'Invalid path'.\n\n",
  "Workable options:\n",
  "1) Use a different performance data source for 2023+ (game-level results is enough for the first model).\n",
  "2) If you have an export/dump, drop it into data/raw/player_stats/ and we ingest it.\n",
  "3) We can still scrape rosters for seasons covered by baseballr team registry (<=2024):\n",
  "   Rscript scripts/02_scrape_rosters.R --season 2024\n"
))

