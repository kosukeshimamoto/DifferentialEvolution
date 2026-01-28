#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    stop(paste("missing argument:", flag))
  }
  args[[idx + 1]]
}

get_optional_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    return(default)
  }
  args[[idx + 1]]
}

csv_path <- get_arg("--csv")
outdir <- get_arg("--outdir")

if (!dir.exists(outdir)) {
  dir.create(outdir, recursive = TRUE)
}

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("ggplot2 is required. Install with install.packages('ggplot2').")
}

library(ggplot2)

df <- read.csv(csv_path, stringsAsFactors = FALSE)
if (nrow(df) == 0) {
  stop("CSV is empty.")
}

alg_arg <- get_optional_arg("--algorithms")
if (is.null(alg_arg)) {
  algorithms <- c("BFGS", "Nelder-Mead", "DE", "SHADE", "L-SHADE", "JSO")
} else {
  algorithms <- trimws(strsplit(alg_arg, ",")[[1]])
  algorithms <- algorithms[algorithms != ""]
  if (length(algorithms) == 0) {
    stop("No algorithms provided to --algorithms.")
  }
}

all_colors <- c(
  "BFGS" = "#F8766D",
  "Nelder-Mead" = "#7CAE00",
  "DE" = "#00BFC4",
  "SHADE" = "#C77CFF",
  "L-SHADE" = "#FF61C3",
  "JSO" = "#00A9FF"
)
colors <- all_colors[algorithms]
if (any(is.na(colors))) {
  missing <- algorithms[is.na(colors)]
  fallback <- grDevices::rainbow(length(missing))
  colors[is.na(colors)] <- fallback
}

algorithm_plot_levels <- rev(algorithms)

dim_levels <- sort(unique(df$dim))
dim_labels <- paste0(dim_levels, "D")
df$dim_label <- factor(paste0(df$dim, "D"), levels = rev(dim_labels))
df$algorithm <- factor(df$algorithm, levels = algorithms, ordered = TRUE)
df$algorithm_plot <- factor(df$algorithm, levels = algorithm_plot_levels, ordered = TRUE)

slugify <- function(name) {
  gsub("[^a-z0-9]+", "_", tolower(name))
}

plot_metric <- function(task, value_col, title_suffix, x_label, outpath) {
  dft <- df[df$task == task, ]
  dft$value <- as.numeric(dft[[value_col]])

  dft$algorithm <- factor(dft$algorithm, levels = algorithms, ordered = TRUE)
  dft$algorithm_plot <- factor(dft$algorithm, levels = algorithm_plot_levels, ordered = TRUE)

  p <- ggplot(dft, aes(x = dim_label, y = value, fill = algorithm_plot)) +
    geom_col(position = position_dodge2(width = 0.8, reverse = FALSE), width = 0.7, na.rm = TRUE) +
    coord_flip() +
    scale_fill_manual(values = colors, breaks = algorithms, limits = algorithm_plot_levels, drop = FALSE) +
    labs(
      title = paste(task, title_suffix),
      x = "dimension",
      y = x_label,
      fill = "algorithm"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA),
      legend.key = element_rect(fill = "white", color = NA),
      axis.text = element_text(color = "#222222"),
      axis.title = element_text(color = "#222222"),
      plot.title = element_text(color = "#222222"),
      legend.text = element_text(color = "#222222"),
      legend.title = element_text(color = "#222222"),
      legend.position = "right"
    )

  ggsave(outpath, plot = p, device = "png", width = 6.8, height = 3.2, dpi = 300)
}

tasks <- unique(df$task)
for (task in tasks) {
  plot_metric(
    task,
    "error_mean",
    "- Mean error / dim",
    "mean error / dim",
    file.path(outdir, paste0(slugify(task), "_error.png"))
  )
  plot_metric(
    task,
    "time_mean_s",
    "- Mean time",
    "time (s)",
    file.path(outdir, paste0(slugify(task), "_time.png"))
  )
}
