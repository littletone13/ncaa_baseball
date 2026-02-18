#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_flag <- function(flag) any(args == flag)

get_value <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (!is.na(idx) && idx < length(args)) return(args[[idx + 1]])
  default
}

season <- as.integer(get_value("--season", NA))
if (is.na(season)) stop("Missing --season (e.g. 2024)")

team_id_filter <- get_value("--team-id", NULL)
sleep_s <- as.numeric(get_value("--sleep", "1.0"))
out_dir <- get_value("--out-dir", "data/raw/rosters")
overwrite <- get_flag("--overwrite")

teams_csv <- get_value("--teams-csv", NA_character_)
crosswalk_csv <- get_value("--crosswalk", "data/registries/name_crosswalk_baseballr.csv")
no_filter <- get_flag("--no-filter")

if (!requireNamespace("baseballr", quietly = TRUE)) {
  stop("Missing R package: baseballr. Run: Rscript scripts/install_R_deps.R")
}
if (!requireNamespace("xml2", quietly = TRUE) || !requireNamespace("rvest", quietly = TRUE)) {
  stop("Missing R packages: xml2/rvest. Re-run: Rscript scripts/install_R_deps.R")
}

write_parquet_ok <- requireNamespace("arrow", quietly = TRUE)
if (!write_parquet_ok && !requireNamespace("readr", quietly = TRUE)) {
  stop("Need either arrow (parquet) or readr (csv). Run: Rscript scripts/install_R_deps.R")
}

curl_html <- function(url) {
  x <- tryCatch({
    system2("curl", c("-sL", url), stdout = TRUE, stderr = TRUE)
  }, error = function(e) {
    character(0)
  })
  paste(x, collapse = "\n")
}

parse_roster <- function(html) {
  if (nchar(html) == 0) stop("empty response")
  if (grepl("^Invalid", html)) stop(html)
  doc <- xml2::read_html(html)
  tables <- rvest::html_elements(doc, "table")
  if (length(tables) < 1) stop("no tables found")

  tbl <- tables[[1]]
  df <- rvest::html_table(tbl, fill = TRUE)
  df <- as.data.frame(df, stringsAsFactors = FALSE)
  if (nrow(df) < 2) stop("unexpected roster table shape")

  # First row is header values.
  colnames(df) <- as.character(df[1, ])
  df <- df[-1, , drop = FALSE]

  # Player links: /players/<id>
  links <- rvest::html_elements(tbl, "a")
  link_names <- rvest::html_text(links, trim = TRUE)
  link_hrefs <- rvest::html_attr(links, "href")
  link_df <- data.frame(
    player_name = link_names,
    player_url = paste0("https://stats.ncaa.org", link_hrefs),
    stringsAsFactors = FALSE
  )
  link_df$player_id <- sub(".*/players/", "", link_df$player_url)

  # Keep first occurrence per name (rare duplicates, but we don't guess).
  link_df <- link_df[!duplicated(link_df$player_name), , drop = FALSE]

  if (!("Player" %in% names(df))) stop("roster table missing 'Player' column")
  names(df)[names(df) == "Player"] <- "player_name"

  df <- merge(df, link_df, by = "player_name", all.x = TRUE, sort = FALSE)
  df
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

teams_all <- baseballr::load_ncaa_baseball_teams()
teams_year <- teams_all[teams_all$year == season & teams_all$division == 1, , drop = FALSE]

if (nrow(teams_year) == 0) {
  stop(paste0(
    "No teams available for season=", season, " via baseballr data repository. ",
    "baseballr's team registry currently only goes through 2024."
  ))
}

if (!is.null(team_id_filter)) {
  teams_year <- teams_year[teams_year$team_id == as.integer(team_id_filter), , drop = FALSE]
}

if (!no_filter) {
  # Prefer deterministic team_id crosswalk (avoids display-name mismatches like "Arizona St.")
  if (!is.na(crosswalk_csv) && file.exists(crosswalk_csv)) {
    xw <- if (requireNamespace("readr", quietly = TRUE)) {
      readr::read_csv(crosswalk_csv, show_col_types = FALSE)
    } else {
      utils::read.csv(crosswalk_csv, stringsAsFactors = FALSE)
    }
    need_cols <- c("canonical_id", "canonical_school", "baseballr_team_id", "status")
    miss_cols <- setdiff(need_cols, names(xw))
    if (length(miss_cols) > 0) stop(paste0("--crosswalk missing cols: ", paste(miss_cols, collapse = ", ")))

    unmatched <- xw[xw$status != "matched" | is.na(xw$baseballr_team_id), , drop = FALSE]
    if (nrow(unmatched) > 0) {
      message("Canonical teams not mapped in crosswalk (skipping):")
      message(paste0("  - ", unmatched$canonical_school, " (", unmatched$canonical_id, ")", collapse = "\n"))
    }

    ids <- unique(xw$baseballr_team_id[xw$status == "matched" & !is.na(xw$baseballr_team_id)])
    teams_year <- teams_year[teams_year$team_id %in% ids, , drop = FALSE]
  } else {
    # Fallback: exact school-name matching (no fuzzy). This will miss abbreviations.
    if (is.na(teams_csv)) {
      if (file.exists("data/registries/teams.csv")) teams_csv <- "data/registries/teams.csv"
    }
    if (!is.na(teams_csv) && file.exists(teams_csv)) {
      canon <- if (requireNamespace("readr", quietly = TRUE)) {
        readr::read_csv(teams_csv, show_col_types = FALSE)
      } else {
        utils::read.csv(teams_csv, stringsAsFactors = FALSE)
      }
      if (!("school" %in% names(canon))) stop("--teams-csv must contain 'school' column")
      canon_schools <- unique(tolower(trimws(canon$school)))
      team_names_norm <- tolower(trimws(teams_year$team_name))
      keep <- team_names_norm %in% canon_schools
      missing <- setdiff(canon_schools, team_names_norm)
      if (length(missing) > 0) {
        message("Canonical schools not found in baseballr team registry for this season (exact match only):")
        message("  ", paste(sort(unique(missing)), collapse = ", "))
      }
      teams_year <- teams_year[keep, , drop = FALSE]
    }
  }
}

if (nrow(teams_year) == 0) stop("No teams matched after filtering.")

for (i in seq_len(nrow(teams_year))) {
  row <- teams_year[i, , drop = FALSE]
  tid <- row$team_id[[1]]
  season_id <- row$season_id[[1]]
  tname <- row$team_name[[1]]
  message(sprintf("Roster %d/%d: %s (team_id=%s, season_id=%s)", i, nrow(teams_year), tname, tid, season_id))

  if (write_parquet_ok) {
    out_path <- file.path(out_dir, sprintf("roster_%d_%s.parquet", season, tid))
  } else {
    out_path <- file.path(out_dir, sprintf("roster_%d_%s.csv", season, tid))
  }

  if (!overwrite && file.exists(out_path)) {
    message("  exists, skipping fetch: ", out_path)
    next
  }

  url <- paste0("https://stats.ncaa.org/team/", tid, "/roster/", season_id)
  html <- curl_html(url)
  roster <- parse_roster(html)

  roster$year <- season
  roster$team_id <- tid
  roster$team_name <- tname
  roster$conference <- row$conference[[1]]
  roster$conference_id <- row$conference_id[[1]]
  roster$division <- row$division[[1]]
  roster$season_id <- season_id

  if (write_parquet_ok) {
    arrow::write_parquet(roster, out_path)
    message("  wrote: ", out_path)
  } else {
    readr::write_csv(roster, out_path)
    message("  wrote: ", out_path)
  }

  Sys.sleep(sleep_s)
}
