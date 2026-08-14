# Generate daily MGDrivE1 trajectories for the matched spatial validation.

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
  stop("trajectory validation requires MGDrivE 1.6.2")
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
  movement[source, source] <- 1.0 - migration_rate
  movement[source, destinations] <- migration_rate * weights / sum(weights)
}
stopifnot(max(abs(rowSums(movement) - 1.0)) < 1e-12)

cube <- MGDrivE::cubeMendelian()
release_vector <- MGDrivE::generateReleaseVector(
  driveCube = cube,
  releasesParameters = list(
    releasesStart = 25,
    releasesNumber = 1,
    releasesInterval = 0,
    releaseProportion = 10
  )
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

parameters <- MGDrivE::parameterizeMGDrivE(
  runID = 1,
  simTime = n_days + 1L,
  sampTime = 1,
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
output_directory <- tempfile("mgdrive1-spatial-trajectory-")
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

result_rows <- vector("list", repeats)
for (repeat_index in seq_len(repeats)) {
  set.seed(base_seed + repeat_index - 1L)
  network$oneRun(verbose = FALSE)
  run_label <- formatC(
    repeat_index,
    width = 3,
    format = "d",
    flag = "0"
  )
  male <- utils::read.csv(
    file.path(output_directory, paste0("M_Run", run_label, ".csv"))
  )
  female <- utils::read.csv(
    file.path(output_directory, paste0("F_Run", run_label, ".csv"))
  )
  male_summary <- stats::aggregate(
    male[c("AA", "Aa", "aa")],
    list(Time = male$Time),
    sum
  )
  female_names <- setdiff(names(female), c("Time", "Patch"))
  female$adult_total <- rowSums(female[female_names])
  female$aa_adult <- rowSums(female[female_names[7:9]])
  female_summary <- stats::aggregate(
    female[c("adult_total", "aa_adult")],
    list(Time = female$Time),
    sum
  )
  trajectory <- merge(male_summary, female_summary, by = "Time", sort = TRUE)
  stopifnot(nrow(trajectory) == n_days + 1L)
  result_rows[[repeat_index]] <- data.frame(
    engine = "MGDrivE1",
    scenario = "spatial",
    mode = mode,
    "repeat" = repeat_index,
    transition = seq.int(0L, n_days),
    adult_total = rowSums(trajectory[c("AA", "Aa", "aa")]) +
      trajectory$adult_total,
    aa_adult = trajectory$aa + trajectory$aa_adult,
    check.names = FALSE
  )
  if (repeat_index < repeats) {
    invisible(utils::capture.output(network$reset(verbose = FALSE)))
  }
}
utils::write.csv(
  do.call(rbind, result_rows),
  output_path,
  row.names = FALSE,
  quote = FALSE
)
