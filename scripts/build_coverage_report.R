#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(readr)
})

csv_safe_read <- function(path) {
  if (!file.exists(path)) {
    return(NULL)
  }
  readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
}

count_rosters <- function(year) {
  files <- list.files("data/raw/rosters", pattern = sprintf("^roster_%d_\\d+\\.parquet$", year), full.names = TRUE)
  if (length(files) == 0) {
    return(list(year = year, n_files = 0))
  }

  ds <- arrow::open_dataset("data/raw/rosters", format = "parquet") %>% filter(.data$year == .env$year)

  teams <- ds %>%
    summarise(
      n_rows = dplyr::n(),
      n_teams = dplyr::n_distinct(team_id),
      n_players = dplyr::n_distinct(player_id, na.rm = TRUE),
      n_missing_player_id = sum(is.na(player_id))
    ) %>%
    collect()

  dup_players <- ds %>%
    filter(!is.na(player_id)) %>%
    group_by(team_id, player_id) %>%
    summarise(n = dplyr::n(), .groups = "drop") %>%
    filter(.data$n > 1) %>%
    summarise(n_dup_player_team_pairs = dplyr::n()) %>%
    collect()

  list(
    year = year,
    n_files = length(files),
    n_rows = teams$n_rows[[1]],
    n_teams = teams$n_teams[[1]],
    n_players = teams$n_players[[1]],
    n_missing_player_id = teams$n_missing_player_id[[1]],
    n_dup_player_team_pairs = dup_players$n_dup_player_team_pairs[[1]]
  )
}

count_pbp <- function(year) {
  path <- sprintf("data/raw/pbp/pbp_%d.parquet", year)
  path_incomplete <- sprintf("data/raw/pbp/incomplete/pbp_%d.parquet", year)
  manifest <- sprintf("data/raw/pbp/pbp_%d.manifest.json", year)
  manifest_incomplete <- sprintf("data/raw/pbp/incomplete/pbp_%d.manifest.json", year)

  resolved <- NULL
  status <- "missing"
  if (file.exists(path)) {
    resolved <- path
    status <- "present"
  } else if (file.exists(path_incomplete)) {
    resolved <- path_incomplete
    status <- "incomplete"
  }

  n_rows <- NA_integer_
  n_games <- NA_integer_
  date_min <- NA_character_
  date_max <- NA_character_

  if (!is.null(resolved)) {
    ds <- arrow::open_dataset(resolved, format = "parquet")
    if (length(ds$schema$names) == 0) {
      n_rows <- 0L
      n_games <- 0L
      date_min <- NA_character_
      date_max <- NA_character_
    } else {
      n_rows <- ds %>% summarise(n_rows = dplyr::n()) %>% collect() %>% pull(n_rows)
      n_games <- ds %>%
        select(game_pbp_id) %>%
        distinct() %>%
        summarise(n_games = dplyr::n()) %>%
        collect() %>%
        pull(n_games)
      date_min <- ds %>%
        summarise(date_min = min(game_date)) %>%
        collect() %>%
        pull(date_min) %>%
        as.character()
      date_max <- ds %>%
        summarise(date_max = max(game_date)) %>%
        collect() %>%
        pull(date_max) %>%
        as.character()
    }
  }

  man_path <- if (file.exists(manifest)) manifest else if (file.exists(manifest_incomplete)) manifest_incomplete else NA_character_

  list(
    year = year,
    status = status,
    path = resolved %||% NA_character_,
    manifest = man_path,
    n_rows = n_rows,
    n_games = n_games,
    date_min = date_min,
    date_max = date_max
  )
}

`%||%` <- function(a, b) if (!is.null(a)) a else b

count_scoreboard <- function(academic_year) {
  path <- sprintf("data/processed/scoreboard/games_%d.csv", academic_year)
  df <- csv_safe_read(path)
  if (is.null(df)) {
    return(list(academic_year = academic_year, status = "missing", path = path))
  }
  away <- df$away_team_ncaa_id
  home <- df$home_team_ncaa_id
  list(
    academic_year = academic_year,
    status = "present",
    path = path,
    n_rows = nrow(df),
    n_unique_contests = dplyr::n_distinct(df$contest_id),
    n_unique_teams = dplyr::n_distinct(c(away, home)),
    n_missing_scores = sum(is.na(df$away_runs) | is.na(df$home_runs))
  )
}

write_report <- function(lines, out_path) {
  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
  writeLines(lines, con = out_path, useBytes = TRUE)
}

main <- function() {
  teams_2026 <- csv_safe_read("data/registries/ncaa_d1_teams_2026.csv")
  n_teams_2026 <- if (is.null(teams_2026)) NA_integer_ else nrow(teams_2026)

  roster_2023 <- count_rosters(2023)
  roster_2024 <- count_rosters(2024)

  pbp_years <- lapply(c(2021, 2022, 2023, 2024, 2025), count_pbp)

  sb_2023 <- count_scoreboard(2023)
  sb_2024 <- count_scoreboard(2024)
  sb_2025 <- count_scoreboard(2025)
  sb_2026 <- count_scoreboard(2026)

  lines <- c(
    "# NCAA_BASEBALL coverage report",
    "",
    sprintf("Generated: %s", format(Sys.time(), tz = "UTC", usetz = TRUE)),
    "",
    "## Team registry",
    sprintf("- NCAA D1 teams (2026): %s", ifelse(is.na(n_teams_2026), "MISSING", as.character(n_teams_2026))),
    "",
    "## Scoreboard game results (schedule_list)",
    sprintf("- 2023: %s", ifelse(sb_2023$status == "present", sprintf("%d contests, %d teams, %d missing scores", sb_2023$n_unique_contests, sb_2023$n_unique_teams, sb_2023$n_missing_scores), "MISSING")),
    sprintf("- 2024: %s", ifelse(sb_2024$status == "present", sprintf("%d contests, %d teams, %d missing scores", sb_2024$n_unique_contests, sb_2024$n_unique_teams, sb_2024$n_missing_scores), "MISSING")),
    sprintf("- 2025: %s", ifelse(sb_2025$status == "present", sprintf("%d contests, %d teams, %d missing scores", sb_2025$n_unique_contests, sb_2025$n_unique_teams, sb_2025$n_missing_scores), "MISSING")),
    sprintf("- 2026: %s", ifelse(sb_2026$status == "present", sprintf("%d contests, %d teams, %d missing scores", sb_2026$n_unique_contests, sb_2026$n_unique_teams, sb_2026$n_missing_scores), "MISSING")),
    "",
    "## Rosters (NCAA roster pages via season_id)",
    sprintf(
      "- 2023: %s",
      ifelse(
        is.null(roster_2023$n_files) || roster_2023$n_files == 0,
        "MISSING",
        sprintf(
          "%d teams, %d players, %d rows, missing player_id=%d",
          roster_2023$n_teams,
          roster_2023$n_players,
          roster_2023$n_rows,
          roster_2023$n_missing_player_id
        )
      )
    ),
    sprintf(
      "- 2024: %s",
      ifelse(
        is.null(roster_2024$n_files) || roster_2024$n_files == 0,
        "MISSING",
        sprintf(
          "%d teams, %d players, %d rows, missing player_id=%d",
          roster_2024$n_teams,
          roster_2024$n_players,
          roster_2024$n_rows,
          roster_2024$n_missing_player_id
        )
      )
    ),
    "",
    "## Play-by-play (baseballr / SportsDataverse assets)",
    "- NOTE: 2023+ coverage is currently incomplete/missing in SportsDataverse assets.",
    ""
  )

  for (x in pbp_years) {
    if (x$status == "missing") {
      lines <- c(lines, sprintf("- %d: MISSING", x$year))
    } else {
      lines <- c(
        lines,
        sprintf(
          "- %d: %s (%s) - rows=%s, games=%s, date_min=%s, date_max=%s",
          x$year,
          toupper(x$status),
          x$path,
          as.character(x$n_rows),
          as.character(x$n_games),
          as.character(x$date_min),
          as.character(x$date_max)
        )
      )
    }
  }

  lines <- c(
    lines,
    "",
    "## Blockers / gaps",
    "- stats.ncaa.org boxscore/PBP/team stats/2025+ rosters are currently behind an Akamai interstitial challenge (non-browser scraping fails).",
    "- To get 2025 season and 2026 current rosters + full PBP/boxscore stats, we need either:",
    "  - a compliant browser-based workflow (cookie/session + rate limiting), or",
    "  - a third-party data provider/API, or",
    "  - manual export/import from NCAA pages.",
    ""
  )

  out_path <- "data/processed/reports/coverage_report.md"
  write_report(lines, out_path)
  cat(sprintf("Wrote %s\n", out_path))
}

main()
