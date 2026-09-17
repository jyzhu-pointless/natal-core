<script setup lang="ts">
/**
 * Debug panel: four debugging tools in one tab.
 *
 * - Event log: live WebSocket log stream (level-filtered).
 * - Param audit: every committed parameter write, tick-ordered (hooks'
 *   set_param writes included; spatial rows carry a "deme{i}:" prefix).
 * - Tick diff: per-genotype (panmictic) or per-deme (spatial) deltas
 *   between two ticks, with the additive-delta invariant on display.
 * - Raw dump: the unrounded state tensor for one tick (one deme when
 *   spatial), for eyeballing what the engine actually holds.
 */
import { computed, ref } from "vue";
import {
  NButton,
  NCard,
  NCheckbox,
  NInputNumber,
  NSelect,
  NSpace,
  NTable,
  NTag,
  NText,
} from "naive-ui";

import type { SpatialDiffPayload } from "../../api/types";
import { useDebugStore } from "../../stores/debug";
import { useSpatialStore } from "../../stores/spatial";

const props = defineProps<{
  dashboardType: "population" | "spatial";
}>();

const debug = useDebugStore();
const spatial = useSpatialStore();

// -- event log ------------------------------------------------------------

const levelFilter = ref<string[]>([]);

const levelOptions = [
  { label: "debug", value: "debug" },
  { label: "info", value: "info" },
  { label: "warning", value: "warning" },
  { label: "error", value: "error" },
];

const visibleLogs = computed(() => {
  const logs = [...debug.logs].reverse().slice(0, 300);
  if (!levelFilter.value.length) {
    return logs;
  }
  return logs.filter((entry) => levelFilter.value.includes(entry.level));
});

function levelColor(level: string): string {
  switch (level) {
    case "error":
      return "#d03050";
    case "warning":
      return "#f0a020";
    case "info":
      return "#2080f0";
    default:
      return "#909090";
  }
}

function fmtTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString();
}

// -- param audit ----------------------------------------------------------

const paramsLoading = ref(false);

async function refreshParams(): Promise<void> {
  paramsLoading.value = true;
  try {
    await debug.refreshParamsLog();
  } finally {
    paramsLoading.value = false;
  }
}

// -- tick diff ------------------------------------------------------------

const diffA = ref<number>(0);
const diffB = ref<number>(0);
const diffLoading = ref(false);

async function runDiff(): Promise<void> {
  diffLoading.value = true;
  try {
    await debug.fetchDiff(diffA.value, diffB.value);
  } finally {
    diffLoading.value = false;
  }
}

const spatialDiff = computed(
  () =>
    props.dashboardType === "spatial"
      ? (debug.diff as SpatialDiffPayload | null)
      : null,
);

function fmt(value: number): string {
  return Math.round(value).toLocaleString();
}

function fmtDelta(value: number): string {
  const rounded = Math.round(value);
  return (rounded > 0 ? "+" : "") + rounded.toLocaleString();
}

// -- raw dump -------------------------------------------------------------

const rawTick = ref<number | null>(null);
const rawDeme = ref<number>(0);
const rawLoading = ref(false);
const rawShowAll = ref(false);

const demeOptions = computed(() =>
  Array.from({ length: spatial.landscape?.n_demes ?? 0 }, (_, i) => ({
    label: spatial.landscape?.deme_names[i] ?? `deme ${i}`,
    value: i,
  })),
);

async function fetchRaw(): Promise<void> {
  rawLoading.value = true;
  try {
    await debug.fetchRawDump(
      rawTick.value,
      props.dashboardType === "spatial" ? rawDeme.value : undefined,
    );
  } finally {
    rawLoading.value = false;
  }
}

const rawRows = computed(() => {
  const dump = debug.rawDump;
  if (!dump) {
    return [];
  }
  // Render as a flat (sex, age) -> per-ztype row list for readability.
  const rows: Array<{ sex: number; age: number; values: number[] }> = [];
  dump.individual_count.forEach((sex, sexIdx) => {
    sex.forEach((ages, ageIdx) => {
      rows.push({ sex: sexIdx, age: ageIdx, values: ages });
    });
  });
  return rawShowAll.value ? rows : rows.slice(0, 12);
});
</script>

<template>
  <NSpace
    vertical
    size="large"
  >
    <NCard
      title="Event log (live stream)"
      size="small"
    >
      <NSpace align="center">
        <NSelect
          v-model:value="levelFilter"
          multiple
          size="small"
          placeholder="all levels"
          :options="levelOptions"
          style="width: 260px"
        />
        <NButton
          size="small"
          @click="debug.clearLogs()"
        >
          Clear
        </NButton>
        <NTag size="small">
          {{ debug.logs.length }} frames
        </NTag>
      </NSpace>
      <div class="log-box">
        <div
          v-for="(entry, index) in visibleLogs"
          :key="`${entry.ts}-${index}`"
          class="log-row"
        >
          <span class="log-ts">{{ fmtTime(entry.ts) }}</span>
          <span
            class="log-level"
            :style="{ color: levelColor(entry.level) }"
          >{{ entry.level }}</span>
          <span class="log-src">{{ entry.source }}</span>
          <span class="log-msg">{{ entry.message }}</span>
        </div>
        <NText
          v-if="!visibleLogs.length"
          depth="3"
        >
          No log frames.
        </NText>
      </div>
    </NCard>

    <NCard
      title="Parameter audit (set_param writes)"
      size="small"
    >
      <NSpace align="center">
        <NButton
          size="small"
          :loading="paramsLoading"
          @click="refreshParams"
        >
          Refresh
        </NButton>
        <NText
          v-if="debug.paramsLog"
          depth="3"
        >
          {{ debug.paramsLog.length }} writes
        </NText>
      </NSpace>
      <NTable
        v-if="debug.paramsLog && debug.paramsLog.length"
        size="small"
        :single-line="false"
        style="margin-top: 8px"
      >
        <thead>
          <tr>
            <th>Tick</th>
            <th>Parameter</th>
            <th>Old</th>
            <th>New</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(row, index) in debug.paramsLog"
            :key="index"
          >
            <td>{{ row.tick }}</td>
            <td class="mono">
              {{ row.name }}
            </td>
            <td>{{ row.old }}</td>
            <td>{{ row.new }}</td>
          </tr>
        </tbody>
      </NTable>
      <NText
        v-else
        depth="3"
      >
        No parameter writes recorded (hooks can change ecology parameters via
        set_param; those writes appear here).
      </NText>
    </NCard>

    <NCard
      title="Tick diff"
      size="small"
    >
      <NSpace align="center">
        <NText depth="3">
          from
        </NText>
        <NInputNumber
          v-model:value="diffA"
          size="small"
          :min="0"
          :show-button="false"
          style="width: 90px"
        />
        <NText depth="3">
          to
        </NText>
        <NInputNumber
          v-model:value="diffB"
          size="small"
          :min="0"
          :show-button="false"
          style="width: 90px"
        />
        <NButton
          size="small"
          type="primary"
          :loading="diffLoading"
          @click="runDiff"
        >
          Diff
        </NButton>
        <template v-if="debug.diff">
          <NTag size="small">
            Δ total {{ fmtDelta(debug.diff.delta_total) }}
          </NTag>
          <NTag
            v-if="!debug.diff.found_a || !debug.diff.found_b"
            size="small"
            type="warning"
          >
            some ticks not in history — live state substituted
          </NTag>
        </template>
      </NSpace>

      <template v-if="debug.diff">
        <!-- Panmictic: per-genotype deltas -->
        <NTable
          v-if="dashboardType === 'population' && 'genotypes' in debug.diff"
          size="small"
          :single-line="false"
          style="margin-top: 8px"
        >
          <thead>
            <tr>
              <th>Genotype</th>
              <th>Female A→B</th>
              <th>Male A→B</th>
              <th>Δ total</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="row in debug.diff.genotypes"
              :key="row.label"
            >
              <td>{{ row.label }}</td>
              <td>{{ fmt(row.female_a) }} → {{ fmt(row.female_b) }}</td>
              <td>{{ fmt(row.male_a) }} → {{ fmt(row.male_b) }}</td>
              <td :style="{ color: row.delta_total < 0 ? '#d03050' : '#18a058' }">
                {{ fmtDelta(row.delta_total) }}
              </td>
            </tr>
          </tbody>
        </NTable>

        <!-- Spatial: per-deme totals -->
        <NTable
          v-else-if="spatialDiff"
          size="small"
          :single-line="false"
          style="margin-top: 8px"
        >
          <thead>
            <tr>
              <th>Deme</th>
              <th>A</th>
              <th>B</th>
              <th>Δ</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="entry in spatialDiff.demes"
              :key="entry.deme"
            >
              <td>{{ entry.name }}</td>
              <td>{{ fmt(entry.total_a) }}</td>
              <td>{{ fmt(entry.total_b) }}</td>
              <td :style="{ color: entry.delta < 0 ? '#d03050' : '#18a058' }">
                {{ fmtDelta(entry.delta) }}
              </td>
            </tr>
          </tbody>
        </NTable>
      </template>
      <NText
        v-else
        depth="3"
      >
        Pick two ticks and press Diff.
      </NText>
    </NCard>

    <NCard
      title="Raw state dump"
      size="small"
    >
      <NSpace align="center">
        <NText depth="3">
          tick (empty = live)
        </NText>
        <NInputNumber
          v-model:value="rawTick"
          size="small"
          :min="0"
          :show-button="false"
          placeholder="live"
          style="width: 100px"
        />
        <template v-if="dashboardType === 'spatial'">
          <NText depth="3">
            deme
          </NText>
          <NSelect
            v-model:value="rawDeme"
            size="small"
            :options="demeOptions"
            style="width: 160px"
          />
        </template>
        <NButton
          size="small"
          :loading="rawLoading"
          @click="fetchRaw"
        >
          Dump
        </NButton>
        <NCheckbox v-model:checked="rawShowAll">
          show all rows
        </NCheckbox>
        <NTag
          v-if="debug.rawDump"
          size="small"
          :type="debug.rawDump.found ? 'default' : 'warning'"
        >
          {{ debug.rawDump.mode }} @ tick {{ debug.rawDump.tick }}
        </NTag>
      </NSpace>

      <NTable
        v-if="debug.rawDump"
        size="small"
        :single-line="false"
        style="margin-top: 8px"
      >
        <thead>
          <tr>
            <th>Sex</th>
            <th>Age</th>
            <th>Per-ztype counts</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in rawRows"
            :key="`${row.sex}-${row.age}`"
          >
            <td>{{ row.sex === 0 ? "F" : "M" }}</td>
            <td>{{ row.age }}</td>
            <td class="mono">
              {{ row.values.map((v) => Math.round(v * 1000) / 1000).join("  ") }}
            </td>
          </tr>
        </tbody>
      </NTable>
      <NText
        v-else
        depth="3"
      >
        Press Dump to fetch the unrounded state tensor.
      </NText>
    </NCard>
  </NSpace>
</template>

<style scoped>
.log-box {
  margin-top: 8px;
  max-height: 260px;
  overflow-y: auto;
  border: 1px solid #eee;
  border-radius: 4px;
  padding: 4px 8px;
  font-family: monospace;
  font-size: 12px;
}

.log-row {
  display: flex;
  gap: 8px;
  padding: 1px 0;
}

.log-ts {
  color: #aaa;
  flex-shrink: 0;
}

.log-level {
  flex-shrink: 0;
  width: 56px;
  font-weight: 600;
}

.log-src {
  color: #777;
  flex-shrink: 0;
  width: 64px;
}

.mono {
  font-family: monospace;
  font-size: 11px;
}
</style>
