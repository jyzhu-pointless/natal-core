<script setup lang="ts">
/**
 * Panmictic dashboard composition root: wires the stores to the shell and
 * panels, and owns the inspection / zoom / restore interactions.
 */
import { computed, onMounted, ref, watch } from "vue";
import { NCard, NSpace, NText, useDialog } from "naive-ui";

import AlleleFreqChart from "../charts/AlleleFreqChart.vue";
import PopulationChart from "../charts/PopulationChart.vue";
import ConfigPanel from "../panels/ConfigPanel.vue";
import DebugPanel from "../panels/DebugPanel.vue";
import GeneticsPanel from "../panels/GeneticsPanel.vue";
import HooksPanel from "../panels/HooksPanel.vue";
import InspectionPanel from "../inspection/InspectionPanel.vue";
import ObservationPanel from "../panels/ObservationPanel.vue";
import DashboardShell from "../layout/DashboardShell.vue";
import { useDebugStore } from "../../stores/debug";
import { useDomainStore } from "../../stores/domain";
import { useHistoryStore } from "../../stores/history";
import { useRegistryStore } from "../../stores/registry";
import { useSimulationStore } from "../../stores/simulation";

const TABS = [
  { name: "inspection", label: "Inspection" },
  { name: "config", label: "Configuration" },
  { name: "hooks", label: "Hooks" },
  { name: "observation", label: "Observation" },
  { name: "genetics", label: "Genetics" },
  { name: "debug", label: "Debug" },
];

const sim = useSimulationStore();
const registry = useRegistryStore();
const domain = useDomainStore();
const history = useHistoryStore();
const debug = useDebugStore();
const dialog = useDialog();

const tabValue = ref<string>("inspection");
const loaded = ref(false);

onMounted(() => {
  void boot();
});

async function boot(): Promise<void> {
  await sim.initialize();
  await Promise.all([
    registry.initialize(),
    domain.initialize(),
    history.refreshSeries(),
    history.refreshLive(),
  ]);
  loaded.value = true;
  // Throttled live-snapshot refresh while running (genotype cards update).
  let lastRefresh = 0;
  watch(
    () => sim.liveTick,
    () => {
      if (history.inspected !== null) {
        return;
      }
      const now = Date.now();
      if (now - lastRefresh >= 500) {
        lastRefresh = now;
        void history.refreshLive();
      }
    },
  );
}

const displayedSnapshot = computed(
  () => history.inspected ?? history.liveSnapshot,
);

const inspectedTick = computed(() => history.inspected?.tick ?? null);

const alleleColorMap = computed<Record<string, string>>(() => {
  const colors: Record<string, string> = {};
  for (const allele of registry.payload?.alleles ?? []) {
    colors[allele.name] = allele.color;
  }
  return colors;
});

async function goToTick(tick: number): Promise<void> {
  await history.inspectTick(tick);
}

function backToLive(): void {
  history.clearInspection();
}

function confirmRestore(tick: number): void {
  dialog.warning({
    title: "Time travel",
    content:
      `Restore the population to tick ${tick}? ` +
      "All later history is truncated and the simulation continues from there.",
    positiveText: "Restore",
    negativeText: "Cancel",
    onPositiveClick: () => {
      debug.pushLocal("warning", "ui", `time travel requested to tick ${tick}`);
      sim.restore(tick);
      history.clearInspection();
    },
  });
}

interface ZoomBatch {
  start?: number;
  end?: number;
}

/** ECharts zoom relay: re-fetch higher-resolution data for the window. */
function onZoom(raw: unknown): void {
  const params = raw as { batch?: ZoomBatch[] };
  const batch = params.batch?.[0];
  if (!batch) {
    return;
  }
  const currentSeries = history.series;
  const ticks = currentSeries?.ticks ?? [];
  if (!ticks.length) {
    return;
  }
  const start = Math.round(((batch.start ?? 0) / 100) * (ticks.length - 1));
  const end = Math.round(((batch.end ?? 100) / 100) * (ticks.length - 1));
  history.viewFrom = ticks[Math.min(start, ticks.length - 1)] ?? null;
  history.viewTo = ticks[Math.min(end, ticks.length - 1)] ?? null;
  void history.refreshSeries(2000);
}
</script>

<template>
  <DashboardShell
    v-model:tab-value="tabValue"
    :tabs="TABS"
  >
    <NSpace
      v-if="loaded"
      vertical
      size="large"
    >
      <template v-if="tabValue === 'inspection'">
        <div class="chart-grid">
          <NCard
            title="Population over time (click a point to inspect that tick)"
            size="small"
          >
            <PopulationChart
              v-if="history.series"
              :series="history.series"
              :inspected-tick="inspectedTick"
              @tick-select="goToTick"
              @data-zoom="onZoom"
            />
          </NCard>
          <NCard
            title="Allele frequencies"
            size="small"
          >
            <AlleleFreqChart
              v-if="history.series"
              :series="history.series"
              :allele-colors="alleleColorMap"
              @tick-select="goToTick"
              @data-zoom="onZoom"
            />
          </NCard>
        </div>
        <NCard
          v-if="displayedSnapshot"
          size="small"
        >
          <InspectionPanel
            :snapshot="displayedSnapshot"
            :inspected-tick="inspectedTick"
            :registry="registry.payload ?? { genotypes: [], ztypes: [], gtypes: [], alleles: [], unordered_genotype_labels: [] }"
            @go-to-tick="goToTick"
            @back-to-live="backToLive"
            @restore-here="confirmRestore(displayedSnapshot.tick)"
          />
        </NCard>
      </template>

      <!-- KeepAlive preserves panel-local state (e.g. observation group
           drafts) across tab switches. -->
      <KeepAlive>
        <ConfigPanel v-if="tabValue === 'config'" />
        <HooksPanel v-else-if="tabValue === 'hooks'" />
        <ObservationPanel v-else-if="tabValue === 'observation'" />
        <GeneticsPanel v-else-if="tabValue === 'genetics'" />
        <DebugPanel
          v-else-if="tabValue === 'debug'"
          dashboard-type="population"
        />
      </KeepAlive>
    </NSpace>
    <NText
      v-else
      depth="3"
    >
      Loading dashboard…
    </NText>
  </DashboardShell>
</template>

<style scoped>
.chart-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
  gap: 16px;
}
</style>
