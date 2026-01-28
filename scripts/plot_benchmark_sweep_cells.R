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
  algorithms <- unique(df$algorithm)
} else {
  algorithms <- trimws(strsplit(alg_arg, ",")[[1]])
  algorithms <- algorithms[algorithms != ""]
  if (length(algorithms) == 0) {
    stop("No algorithms provided to --algorithms.")
  }
}

dims <- sort(unique(df$dim))
maxevals_levels <- sort(unique(df$maxevals))
popsize_levels <- sort(unique(df$popsize))

has_pmax <- "pmax" %in% names(df)
if (has_pmax) {
  df$pmax <- as.numeric(df$pmax)
  pmax_levels <- sort(unique(df$pmax))
  pmax_labels <- sapply(pmax_levels, function(v) {
    paste0("pmax=", formatC(v, format = "f", digits = 2))
  })
} else {
  pmax_levels <- numeric(0)
  pmax_labels <- character(0)
}

maxevals_labels <- sapply(maxevals_levels, function(v) {
  if (v >= 1000000) {
    return(paste0(v / 1000000, "M"))
  }
  if (v >= 1000) {
    return(paste0(v / 1000, "k"))
  }
  as.character(v)
})

pop_colors <- grDevices::hcl.colors(length(popsize_levels), "Dark 3")
names(pop_colors) <- as.character(popsize_levels)

df$algorithm <- factor(df$algorithm, levels = algorithms, ordered = TRUE)
df$popsize <- factor(df$popsize, levels = popsize_levels, ordered = TRUE)
df$maxevals <- as.integer(df$maxevals)

slugify <- function(name) {
  gsub("[^a-z0-9]+", "_", tolower(name))
}

plot_cell <- function(dft, value_col, y_label, outpath) {
  dft$value <- as.numeric(dft[[value_col]])
  dft$maxevals_label <- factor(
    dft$maxevals,
    levels = maxevals_levels,
    labels = maxevals_labels,
    ordered = TRUE
  )
  if (has_pmax) {
    dft$pmax_label <- factor(
      dft$pmax,
      levels = pmax_levels,
      labels = pmax_labels,
      ordered = TRUE
    )
  }

  p <- ggplot(dft, aes(x = maxevals_label, y = value, fill = popsize)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7, na.rm = TRUE) +
    scale_fill_manual(values = pop_colors, drop = FALSE) +
    labs(x = "maxevals", y = y_label, fill = "popsize") +
    theme_minimal(base_size = 9) +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
      axis.text = element_text(color = "#222222", size = 8),
      axis.title = element_text(color = "#222222", size = 9),
      legend.position = "bottom",
      legend.title = element_text(size = 8),
      legend.text = element_text(size = 8),
      strip.text = element_text(size = 8),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA)
    )

  if (has_pmax && length(pmax_levels) > 1) {
    p <- p + facet_wrap(~pmax_label, ncol = 1)
  }

  height <- 2.3
  if (has_pmax && length(pmax_levels) > 1) {
    height <- 2.3 + 0.6 * (length(pmax_levels) - 1)
  }
  ggsave(outpath, plot = p, device = "png", width = 2.6, height = height, dpi = 300)
}

tasks <- unique(df$task)
metrics <- list(
  list(name = "error", col = "error_mean", label = "mean error / dim"),
  list(name = "time", col = "time_mean_s", label = "time (s)")
)

for (task in tasks) {
  task_slug <- slugify(task)
  for (metric in metrics) {
    metric_dir <- file.path(outdir, task_slug, metric$name)
    if (!dir.exists(metric_dir)) {
      dir.create(metric_dir, recursive = TRUE)
    }
    for (dim in dims) {
      for (alg in algorithms) {
        dft <- df[df$task == task & df$dim == dim & df$algorithm == alg, ]
        if (nrow(dft) == 0) {
          next
        }
        filename <- paste0("dim_", dim, "_", slugify(alg), ".png")
        outpath <- file.path(metric_dir, filename)
        plot_cell(dft, metric$col, metric$label, outpath)
      }
    }
  }
}
