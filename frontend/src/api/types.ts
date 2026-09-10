/**
 * Wire contract between the Vue frontend and the Python webui server.
 *
 * These types mirror the TypedDicts in
 * `src/natal/frontend/webui/serialization.py` and the Pydantic models in
 * `protocol.py` / `rest.py`.  The Python contract tests
 * (tests/test_webui_api.py) are the source of truth — keep both sides in
 * sync when changing a shape.
 */

export type DashboardType = "population" | "spatial";

export type BackendKind = "rust" | "python";

export type SimulationStatus = "ready" | "running" | "finished" | "error";

export type LogLevel = "debug" | "info" | "warning" | "error";

/** Response of `GET /api/meta`. */
export interface MetaInfo {
  app: string;
  title: string;
  dashboard_type: DashboardType;
  backend: BackendKind;
  tick: number;
  status: SimulationStatus;
  interval_ms: number;
  population_name: string;
}

// ---------------------------------------------------------------------------
// State snapshots
// ---------------------------------------------------------------------------

export interface SpermEntry {
  age: number;
  female_index: number;
  male_index: number;
  female_label: string;
  male_label: string;
  value: number;
}

export interface GenotypeStateRow {
  index: number;
  label: string;
  ztype_indices: number[];
  female: number;
  male: number;
  total: number;
  female_per_age: number[];
  male_per_age: number[];
  viability: [number, number];
  fecundity: [number, number];
}

/** Response of `GET /api/state`. */
export interface StateSnapshot {
  tick: number;
  mode: "live" | "history";
  found: boolean;
  is_age_structured: boolean;
  total: number;
  female: number;
  male: number;
  female_per_age: number[];
  male_per_age: number[];
  genotypes: GenotypeStateRow[];
  sperm_storage: SpermEntry[] | null;
  history_len: number;
}

// ---------------------------------------------------------------------------
// History series
// ---------------------------------------------------------------------------

/** Response of `GET /api/history/series`. */
export interface HistorySeries {
  ticks: number[];
  total: number[];
  female: number[];
  male: number[];
  known_alleles: string[];
  allele_frequencies: Record<string, number[]>;
  truncated: boolean;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

export interface GrowthModeInfo {
  code: number;
  name: string;
}

export interface ConfigScalars {
  population_name: string;
  stochastic: boolean;
  continuous_sampling: boolean;
  discrete_generation: boolean;
  extreme_speed_mode: number;
  n_sexes: number;
  n_ages: number;
  n_genotypes: number;
  n_gtypes: number;
  n_glabs: number;
  n_slabs: number;
  new_adult_age: number;
  carrying_capacity: number;
  eggs_per_female: number;
  sex_ratio: number;
  sperm_displacement_rate: number;
  low_density_growth_rate: number;
  expected_competition_strength: number;
  expected_survival_rate: number;
  generation_time: number;
  fixed_egg_count: boolean;
  juvenile_growth_mode: GrowthModeInfo;
}

export interface FitnessRow {
  genotype: string;
  age: number | null;
  female: number;
  male: number;
}

export interface SexualSelectionRow {
  female_genotype: string;
  male_genotype: string;
  preference: number;
}

export interface PresetModifierItem {
  id: number;
  name: string;
  kind: string;
}

export interface PresetInfo {
  preset_name: string;
  gamete_modifiers: PresetModifierItem[];
  zygote_modifiers: PresetModifierItem[];
}

export interface PresetsSummary {
  preset_count: number;
  presets: PresetInfo[];
}

/** Response of `GET /api/config`. */
export interface ConfigPayload {
  scalars: ConfigScalars;
  full: Record<string, unknown>;
  presets: PresetsSummary;
  viability: FitnessRow[];
  fecundity: FitnessRow[];
  sexual_selection: SexualSelectionRow[];
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

export interface HookOpInfo {
  type: string;
  genotypes: string | string[];
  ages: number | number[] | string;
  sex: string;
  param: number;
  condition: string | null;
}

export interface HookInfo {
  event: string;
  name: string;
  priority: number;
  kind: "declarative" | "callback" | "compiled";
  operations: HookOpInfo[] | null;
  signature: string | null;
  source: string | null;
}

// ---------------------------------------------------------------------------
// Genetics matrices
// ---------------------------------------------------------------------------

export interface Matrix2D {
  row_labels: string[];
  col_labels: string[];
  data: number[][];
}

export interface FertilizationMatrix {
  row_labels: string[];
  col_labels: string[];
  zygote_labels: string[];
  primary_index: number[][];
  /** Probability of the primary (most probable) offspring; NaN when empty. */
  primary_probability: number[][];
  cell_text: string[][];
  too_large: boolean;
}

/** Response of `GET /api/genetics/matrices`. */
export interface GeneticsPayload {
  meiosis: Matrix2D[];
  fertilization: FertilizationMatrix;
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

export interface GenotypeEntry {
  index: number;
  label: string;
  ztype_indices: number[];
  svg: string;
}

export interface ZTypeEntry {
  index: number;
  genotype_index: number;
  genotype_label: string;
  slab: string;
}

export interface GTypeEntry {
  index: number;
  label: string;
  gamete_label: string;
}

export interface AlleleEntry {
  name: string;
  locus: string;
  color: string;
}

/** Response of `GET /api/registry`. */
export interface RegistryPayload {
  genotypes: GenotypeEntry[];
  ztypes: ZTypeEntry[];
  gtypes: GTypeEntry[];
  alleles: AlleleEntry[];
  unordered_genotype_labels: string[];
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

export interface ObservationGroupBody {
  genotype?: string[];
  sex?: "female" | "male";
  age_start?: number;
  age_end?: number;
}

/** Request body of `POST /api/observation`. */
export interface ObservationBody {
  groups: ObservationGroupBody[];
  collapse_age: boolean;
}

export interface ObservationRow {
  group: string;
  age: number | null;
  female: number;
  male: number;
  total: number;
}

/** Response of `POST /api/observation`. */
export interface ObservationResultPayload {
  labels: string[];
  collapse_age: boolean;
  rows: ObservationRow[];
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

export interface ParamChangeRow {
  tick: number;
  name: string;
  old: number;
  new: number;
}

export interface DiffGenotypeRow {
  label: string;
  female_a: number;
  female_b: number;
  male_a: number;
  male_b: number;
  delta_female: number;
  delta_male: number;
  delta_total: number;
}

/** Response of `GET /api/debug/diff`. */
export interface DiffPayload {
  tick_a: number;
  tick_b: number;
  found_a: boolean;
  found_b: boolean;
  total_a: number;
  total_b: number;
  delta_total: number;
  genotypes: DiffGenotypeRow[];
}

/** Response of `GET /api/debug/state_raw`. */
export interface RawStateDump {
  tick: number;
  mode: "live" | "history";
  found: boolean;
  individual_count: number[][][];
  sperm_storage: number[][][] | null;
}

// ---------------------------------------------------------------------------
// WebSocket frames
// ---------------------------------------------------------------------------

export interface PingRequest {
  type: "ping";
  nonce: string;
}

export type ClientMessage =
  | PingRequest
  | { type: "play" }
  | { type: "pause" }
  | { type: "step"; n: number }
  | { type: "run_to_tick"; tick: number }
  | { type: "set_interval_ms"; value: number }
  | { type: "set_record_every"; value: number }
  | { type: "set_max_history"; value: number }
  | { type: "reset" }
  | { type: "restore"; tick: number };

/** Server -> client: sent once per connection after accept. */
export interface HelloMessage {
  type: "hello";
  tick: number;
  status: SimulationStatus;
  interval_ms: number;
  backend: BackendKind;
  dashboard_type: DashboardType;
}

export interface PongMessage {
  type: "pong";
  nonce: string;
}

export interface StatusMessage {
  type: "status";
  status: SimulationStatus;
  error: string | null;
}

export interface TickUpdateMessage {
  type: "tick_update";
  tick: number;
  total: number;
  female: number;
  male: number;
  is_finished: boolean;
  history_len: number;
}

export interface LogMessage {
  type: "log";
  ts: number;
  level: LogLevel;
  source: string;
  message: string;
  data: string | null;
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export interface ResetDoneMessage {
  type: "reset_done";
}

export interface RestoredMessage {
  type: "restored";
  tick: number;
}

export type ServerMessage =
  | HelloMessage
  | PongMessage
  | StatusMessage
  | TickUpdateMessage
  | LogMessage
  | ErrorMessage
  | ResetDoneMessage
  | RestoredMessage;

export function parseServerMessage(raw: unknown): ServerMessage | null {
  if (typeof raw !== "object" || raw === null) {
    return null;
  }
  const message = raw as Record<string, unknown>;
  if (typeof message["type"] !== "string") {
    return null;
  }
  return message as unknown as ServerMessage;
}

// ---------------------------------------------------------------------------
// Spatial dashboard
// ---------------------------------------------------------------------------

export interface SpatialTopologyInfo {
  kind: "hex" | "square" | "none";
  rows: number;
  cols: number;
  wrap: boolean;
  grid_ij: number[][] | null;
  xy: number[][] | null;
}

/** Response of `GET /api/spatial/landscape`. */
export interface SpatialLandscapePayload {
  n_demes: number;
  topology: SpatialTopologyInfo;
  deme_names: string[];
  totals: number[];
  females: number[];
  males: number[];
  genotype_labels: string[];
  genotype_counts: number[][];
  allele_names: string[];
  allele_frequencies: number[][];
}

/** Response of `GET /api/spatial/deme/{idx}`. */
export interface SpatialDemeDetail {
  index: number;
  name: string;
  grid_ij: number[] | null;
  total: number;
  female: number;
  male: number;
  is_age_structured: boolean;
  female_per_age: number[];
  male_per_age: number[];
  genotypes: GenotypeStateRow[];
}

export interface SpatialMigrationEntry {
  dest: number;
  dest_name: string;
  weight: number;
  share: number;
  dest_total: number;
}

/** Response of `GET /api/spatial/migration/{idx}`. */
export interface SpatialMigrationDetail {
  source: number;
  source_name: string;
  rate_mean: number;
  entries: SpatialMigrationEntry[];
}

export type LandscapeMetric =
  | { kind: "total" }
  | { kind: "female" }
  | { kind: "male" }
  | { kind: "genotype"; label: string }
  | { kind: "allele"; name: string };

export interface SpatialParamChangeRow {
  tick: number;
  name: string; // "deme{i}:{param}"
  old: number;
  new: number;
}

export interface SpatialDiffEntry {
  deme: number;
  name: string;
  total_a: number;
  total_b: number;
  delta: number;
}

/** Response of `GET /api/debug/diff` for spatial dashboards. */
export interface SpatialDiffPayload {
  tick_a: number;
  tick_b: number;
  found_a: boolean;
  found_b: boolean;
  delta_total: number;
  demes: SpatialDiffEntry[];
}

/** Response of `GET /api/debug/state_raw` for spatial dashboards. */
export interface SpatialRawDump {
  tick: number;
  deme: number;
  mode: "live" | "history";
  found: boolean;
  individual_count: number[][][];
}

export type DebugDiffPayload = DiffPayload | SpatialDiffPayload;
export type DebugRawDump = RawStateDump | SpatialRawDump;
export type DebugParamRow = ParamChangeRow | SpatialParamChangeRow;
