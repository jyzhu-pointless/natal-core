# Generate the deterministic single-patch MGDrivE1 reference trajectory.
#
# Install MGDrivE from commit f7ec820e8a6b0f4fa5697b190f6cb0b1d2d02311,
# then run:
#   MGDRIVE_R_LIB=/path/to/R/library Rscript \
#     benchmarks/mgdrive1/generate_reference.R \
#     benchmarks/mgdrive1/reference/mendelian_single_patch.csv.fixture

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("expected one output CSV path")
}

extra_library <- Sys.getenv("MGDRIVE_R_LIB")
if (nzchar(extra_library)) {
  .libPaths(c(extra_library, .libPaths()))
}
if (!requireNamespace("MGDrivE", quietly = TRUE)) {
  stop("MGDrivE 1.6.2 must be installed from the pinned source commit")
}
if (as.character(utils::packageVersion("MGDrivE")) != "1.6.2") {
  stop("reference requires MGDrivE 1.6.2")
}

output_directory <- tempfile("mgdrive1-reference-")
dir.create(output_directory)
on.exit(unlink(output_directory, recursive = TRUE), add = TRUE)

t_max <- 365L
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
patch_releases <- list(list(
  maleReleases = release_vector,
  femaleReleases = release_vector,
  eggReleases = NULL,
  matedFemaleReleases = NULL
))

MGDrivE::setupMGDrivE(stochasticityON = FALSE, verbose = FALSE)
parameters <- MGDrivE::parameterizeMGDrivE(
  runID = 1,
  simTime = t_max,
  sampTime = 1,
  nPatch = 1,
  beta = 20,
  muAd = 0.09,
  popGrowth = 1.175,
  tEgg = 5,
  tLarva = 6,
  tPupa = 4,
  AdPopEQ = 500,
  inheritanceCube = cube
)
network <- MGDrivE::Network$new(
  params = parameters,
  driveCube = cube,
  patchReleases = patch_releases,
  migrationMale = matrix(1, nrow = 1, ncol = 1),
  migrationFemale = matrix(1, nrow = 1, ncol = 1),
  migrationBatch = MGDrivE::basicBatchMigration(
    batchProbs = 0,
    sexProbs = c(0.5, 0.5),
    numPatches = 1
  ),
  directory = output_directory,
  verbose = FALSE
)
network$oneRun(verbose = FALSE)

male <- utils::read.csv(file.path(output_directory, "M_Run001.csv"))
female <- utils::read.csv(file.path(output_directory, "F_Run001.csv"))
stopifnot(identical(male$Time, female$Time), nrow(male) == t_max)

reference <- cbind(
  data.frame(day = male$Time),
  setNames(male[c("AA", "Aa", "aa")], paste0("male_", c("AA", "Aa", "aa"))),
  setNames(
    female[setdiff(names(female), c("Time", "Patch"))],
    paste0("female_", rep(c("AA", "Aa", "aa"), each = 3), "_mate_",
           rep(c("AA", "Aa", "aa"), times = 3))
  )
)
reference$source_commit <- "f7ec820e8a6b0f4fa5697b190f6cb0b1d2d02311"
utils::write.csv(reference, args[[1]], row.names = FALSE, quote = FALSE)
