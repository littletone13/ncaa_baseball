options(repos = c(CRAN = "https://cloud.r-project.org"))

pkgs <- c(
  "baseballr",
  "dplyr",
  "purrr",
  "readr",
  "stringr",
  "arrow",
  "httr",
  "jsonlite"
)

installed <- rownames(installed.packages())
to_install <- setdiff(pkgs, installed)

if (length(to_install) == 0) {
  message("All R deps already installed.")
  quit(status = 0)
}

message("Installing (best-effort): ", paste(to_install, collapse = ", "))

failed <- c()
for (p in to_install) {
  message("-> ", p)
  ok <- tryCatch({
    install.packages(p)
    TRUE
  }, error = function(e) {
    message("!! failed: ", p, " :: ", conditionMessage(e))
    FALSE
  })
  if (!ok) failed <- c(failed, p)
}

if (length(failed) > 0) {
  message("Some packages failed to install: ", paste(failed, collapse = ", "))
  message("If 'arrow' failed, scripts will fall back to CSV (no parquet).")
  quit(status = 1)
}

message("Done.")
