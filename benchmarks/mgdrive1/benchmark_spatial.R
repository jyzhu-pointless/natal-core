# Benchmark MGDrivE1 on the same finite local-kernel hex scenario as NATAL.
#
# Usage:
#   MGDRIVE_R_LIB=/path/to/R/library Rscript \
#     benchmarks/mgdrive1/benchmark_spatial.R \
#     deterministic 15 15 30 3 /tmp/mgdrive-spatial.csv

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7L) {
  stop("expected: mode rows cols n_days repeats seed output.csv")
}
mode <- args[[1]]
rows <- as.integer(args[[2]])
cols <- as.integer(args[[3]])
n_days <- as.integer(args[[4]])
repeats <- as.integer(args[[5]])
base_seed <- as.integer(args[[6]])
output_path <- args[[7]]
if (!(mode %in% c("deterministic", "stochastic"))) {
  stop("mode must be deterministic or stochastic")
}

extra_library <- Sys.getenv("MGDRIVE_R_LIB")
if (nzchar(extra_library)) {
  .libPaths(c(extra_library, .libPaths()))
}
if (!requireNamespace("MGDrivE", quietly = TRUE)) {
  stop("MGDrivE 1.6.2 must be installed from the pinned source commit")
}
if (as.character(utils::packageVersion("MGDrivE")) != "1.6.2") {
  stop("benchmark requires MGDrivE 1.6.2")
}

kernel_size <- 5L
sigma <- 1.0
migration_rate <- 0.05
radius <- (kernel_size - 1L) %/% 2L
n_patches <- rows * cols

movement <- matrix(0.0, nrow = n_patches, ncol = n_patches)
for (source in seq_len(n_patches)) {
  source_row <- (source - 1L) %/% cols
  source_col <- (source - 1L) %% cols
  destinations <- integer(0)
  weights <- numeric(0)
  for (dr in seq.int(-radius, radius)) {
    for (dc in seq.int(-radius, radius)) {
      if (dr == 0L && dc == 0L) {
        next
      }
      destination_row <- source_row + dr
      destination_col <- source_col + dc
      if (
        destination_row < 0L || destination_row >= rows ||
        destination_col < 0L || destination_col >= cols
      ) {
        next
      }
      distance_squared <- dr * dr + dc * dc + dr * dc
      destinations <- c(
        destinations,
        destination_row * cols + destination_col + 1L
      )
      weights <- c(
        weights,
        exp(-distance_squared / (2.0 * sigma * sigma))
      )
    }
  }
  if (length(destinations) == 0L) {
    movement[source, source] <- 1.0
    next
  }
  movement[source, source] <- 1.0 - migration_rate
  movement[source, destinations] <- migration_rate * weights / sum(weights)
}
stopifnot(max(abs(rowSums(movement) - 1.0)) < 1e-12)

cube <- MGDrivE::cubeMendelian()
release_parameters <- list(
  releasesStart = 25,
  releasesNumber = 1,
  releasesInterval = 0,
  releaseProportion = 10
)
release_vector <- MGDrivE::generateReleaseVector(
  driveCube = cube,
  releasesParameters = release_parameters
)
patch_releases <- replicate(
  n = n_patches,
  expr = list(
    maleReleases = NULL,
    femaleReleases = NULL,
    eggReleases = NULL,
    matedFemaleReleases = NULL
  ),
  simplify = FALSE
)
center_patch <- ((rows - 1L) %/% 2L) * cols + (cols - 1L) %/% 2L + 1L
patch_releases[[center_patch]]$maleReleases <- release_vector
patch_releases[[center_patch]]$femaleReleases <- release_vector

MGDrivE::setupMGDrivE(
  stochasticityON = identical(mode, "stochastic"),
  verbose = FALSE
)

# MGDrivE 1.6.2 reuses `i` in nested loops in deterministic female
# migration. This replacement preserves its intended dense-matrix semantics.
if (identical(mode, "deterministic")) {
  corrected_deterministic_migration <- function() {
    private$mMoveMat[] <- 0
    private$fMoveArray[] <- 0
    for (source in seq_len(private$nPatch)) {
      private$mMoveMat[] <- private$mMoveMat +
        private$patches[[source]]$get_malePopulation() %*%
        private$migrationMale[source, , drop = FALSE]
      female_population <- private$patches[[source]]$get_femalePopulation()
      for (destination in seq_len(private$nPatch)) {
        private$fMoveArray[, , destination] <-
          private$fMoveArray[, , destination] +
          female_population * private$migrationFemale[source, destination]
      }
    }
    for (destination in seq_len(private$nPatch)) {
      private$patches[[destination]]$oneDay_migrationIn(
        maleIn = private$mMoveMat[, destination],
        femaleIn = private$fMoveArray[, , destination]
      )
    }
  }
  MGDrivE::Network$set(
    which = "public",
    name = "oneDay_Migration",
    value = corrected_deterministic_migration,
    overwrite = TRUE
  )
}

benchmark_run <- function() {
  while (private$simTime >= private$tNow) {
    self$oneDay()
    private$tNow <- private$tNow + 1L
  }
}
adult_totals <- function() {
  male <- numeric(private$driveCube$genotypesN)
  female <- matrix(
    0.0,
    nrow = private$driveCube$genotypesN,
    ncol = private$driveCube$genotypesN
  )
  for (patch in private$patches) {
    male <- male + patch$get_malePopulation()
    female <- female + patch$get_femalePopulation()
  }
  c(male, c(t(female)))
}
recessive_spatial_moments <- function() {
  male_radius2 <- 0.0
  female_radius2 <- 0.0
  center_row <- (rows - 1L) %/% 2L
  center_col <- (cols - 1L) %/% 2L
  for (patch_index in seq_len(private$nPatch)) {
    patch_row <- (patch_index - 1L) %/% cols
    patch_col <- (patch_index - 1L) %% cols
    dr <- patch_row - center_row
    dc <- patch_col - center_col
    distance_squared <- dr * dr + dc * dc + dr * dc
    male_radius2 <- male_radius2 +
      private$patches[[patch_index]]$get_malePopulation()[3] *
      distance_squared
    female_radius2 <- female_radius2 +
      sum(private$patches[[patch_index]]$get_femalePopulation()[3, ]) *
      distance_squared
  }
  c(male_radius2, female_radius2)
}
MGDrivE::Network$set(
  which = "public",
  name = "benchmarkRun",
  value = benchmark_run,
  overwrite = TRUE
)
MGDrivE::Network$set(
  which = "public",
  name = "recessiveSpatialMoments",
  value = recessive_spatial_moments,
  overwrite = TRUE
)
MGDrivE::Network$set(
  which = "public",
  name = "adultTotals",
  value = adult_totals,
  overwrite = TRUE
)

parameters <- MGDrivE::parameterizeMGDrivE(
  runID = 1,
  simTime = n_days + 1L,
  sampTime = n_days + 2L,
  nPatch = n_patches,
  beta = 20,
  muAd = 0.09,
  popGrowth = 1.175,
  tEgg = 5,
  tLarva = 6,
  tPupa = 4,
  AdPopEQ = 500,
  inheritanceCube = cube
)
output_directory <- tempfile("mgdrive1-spatial-benchmark-")
dir.create(output_directory)
on.exit(unlink(output_directory, recursive = TRUE), add = TRUE)
network <- MGDrivE::Network$new(
  params = parameters,
  driveCube = cube,
  patchReleases = patch_releases,
  migrationMale = movement,
  migrationFemale = movement,
  migrationBatch = MGDrivE::basicBatchMigration(
    batchProbs = 0,
    sexProbs = c(0.5, 0.5),
    numPatches = n_patches
  ),
  directory = output_directory,
  verbose = FALSE
)

result <- matrix(0.0, nrow = repeats, ncol = 16)
for (repeat_index in seq_len(repeats)) {
  set.seed(base_seed + repeat_index - 1L)
  elapsed <- system.time(network$benchmarkRun())["elapsed"]
  result[repeat_index, ] <- c(
    repeat_index,
    elapsed,
    network$adultTotals(),
    network$recessiveSpatialMoments()
  )
  if (repeat_index < repeats) {
    invisible(utils::capture.output(network$reset(verbose = FALSE)))
  }
}
colnames(result) <- c(
  "repeat",
  "elapsed_seconds",
  "male_AA",
  "male_Aa",
  "male_aa",
  "female_AA_mate_AA",
  "female_AA_mate_Aa",
  "female_AA_mate_aa",
  "female_Aa_mate_AA",
  "female_Aa_mate_Aa",
  "female_Aa_mate_aa",
  "female_aa_mate_AA",
  "female_aa_mate_Aa",
  "female_aa_mate_aa",
  "spatial_male_aa_radius2",
  "spatial_female_aa_radius2"
)
output <- data.frame(
  engine = "MGDrivE1-1.6.2",
  mode = mode,
  rows = rows,
  cols = cols,
  n_days = n_days,
  source_commit = "f7ec820e8a6b0f4fa5697b190f6cb0b1d2d02311",
  result,
  check.names = FALSE
)
utils::write.csv(output, output_path, row.names = FALSE, quote = FALSE)
