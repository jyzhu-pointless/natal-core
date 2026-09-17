<script setup lang="ts">
/**
 * Spatial dashboard composition root: landscape with metric selector and
 * deme picking, selected-deme / migration views, global aggregate charts,
 * and the deme-0 genetics panels (config / hooks / genetics).
 */
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { NCard, NSelect, NSpace, NText } from "naive-ui";

import AlleleFreqChart from "../charts/AlleleFreqChart.vue";
import LandscapeMap from "../charts/LandscapeMap.vue";
import PopulationChart from "../charts/PopulationChart.vue";
import ConfigPanel from "../panels/ConfigPanel.vue";
import DebugPanel from "../panels/DebugPanel.vue";
import GeneticsPanel from "../panels/GeneticsPanel.vue";
import HooksPanel from "../panels/HooksPanel.vue";
import DashboardShell from "../layout/DashboardShell.vue";
import MigrationPanel from "../spatial/MigrationPanel.vue";
import SelectedDemePanel from "../spatial/SelectedDemePanel.vue";
import { useDomainStore } from "../../stores/domain";
import { useRegistryStore } from "../../stores/registry";
import { useSimulationStore } from "../../stores/simulation";
import { useSpatialStore } from "../../stores/spatial";
import type { LandscapeMetric } from "../../api/types";

const TABS = [
  { name: "landscape", label: "Landscape" },
  { name: "deme", label: "Selected deme" },
  { name: "config", label: "Configuration" },
  { name: "hooks", label: "Hooks" },
  { name: "genetics", label: "Genetics" },
  { name: "debug", label: "Debug" },
];

const sim = useSimulationStore();
const registry = useRegistryStore();
const domain = useDomainStore();
const spatial = useSpatialStore();

const tabValue = ref<string>("landscape");
const loaded = ref(false);
const bootError = ref<string | null>(null);

const metric = ref<LandscapeMetric>({ kind: "total" });

const metricOptions = computed(() => {
  const land = spatial.landscape ?? null;
  const options: Array<{ label: string; value: string }> = [
    { label: "Total population", value: "total" },
    { label: "Female", value: "female" },
    { label: "Male", value: "male" },
  ];
  for (const allele of land?.allele_names ?? []) {
    options.push({ label: `${allele} frequency`, value: `allele:${allele}` });
  }
  for (const genotype of land?.genotype_labels ?? []) {
    options.push({ label: `${genotype} frequency`, value: `genotype:${genotype}` });
  }
  return options;
});

const metricValue = computed({
  get: () => {
    const m = metric.value;
    if (m.kind === "genotype") return `genotype:${m.label}`;
    if (m.kind === "allele") return `allele:${m.name}`;
    return m.kind;
  },
  set: (raw: string | null) => {
    if (raw === null) {
      metric.value = { kind: "total" };
      return;
    }
    if (raw.startsWith("genotype:")) {
      metric.value = { kind: "genotype", label: raw.slice("genotype:".length) };
    } else if (raw.startsWith("allele:")) {
      metric.value = { kind: "allele", name: raw.slice("allele:".length) };
    } else {
      metric.value = { kind: raw as "total" | "female" | "male" };
    }
  },
});

const alleleColorMap = computed<Record<string, string>>(() => {
  const colors: Record<string, string> = {};
  for (const allele of registry.payload?.alleles ?? []) {
    colors[allele.name] = allele.color;
  }
  return colors;
});

onMounted(() => {
  void boot();
});

async function boot(): Promise<void> {
  try {
    await sim.initialize();
    await Promise.all([registry.initialize(), domain.initialize()]);
    await spatial.initialize();
    loaded.value = true;
  } catch (caught) {
    bootError.value = String(caught);
  }
}

onUnmounted(() => {
  spatial.dispose();
});

function onSelectDeme(index: number): void {
  void spatial.selectDeme(index);
  tabValue.value = "deme";
}

watch(
  () => spatial.selectedId,
  (selected) => {
    // Keep the deme tab in sync when selection clears (future feature).
    if (selected === null && tabValue.value === "deme") {
      tabValue.value = "landscape";
    }
  },
);
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
      <template v-if="tabValue === 'landscape'">
        <NCard size="small">
          <NSpace align="center">
            <NSelect
              v-model:value="metricValue"
              :options="metricOptions"
              size="small"
              style="width: 260px"
            />
            <NText depth="3">
              click a deme to inspect it
            </NText>
          </NSpace>
          <LandscapeMap
            v-if="spatial.landscape"
            :landscape="spatial.landscape"
            :metric="metric"
            :selected="spatial.selectedId"
            @deme-select="onSelectDeme"
          />
        </NCard>
        <div class="chart-grid">
          <NCard
            title="Global population (all demes)"
            size="small"
          >
            <PopulationChart
              v-if="spatial.series"
              :series="spatial.series"
              :inspected-tick="null"
            />
          </NCard>
          <NCard
            title="Global allele frequencies"
            size="small"
          >
            <AlleleFreqChart
              v-if="spatial.series"
              :series="spatial.series"
              :allele-colors="alleleColorMap"
            />
          </NCard>
        </div>
      </template>

      <template v-else-if="tabValue === 'deme'">
        <template v-if="spatial.selectedId !== null && spatial.demeDetail">
          <div class="deme-grid">
            <SelectedDemePanel
              :detail="spatial.demeDetail"
              :registry="registry.payload ?? { genotypes: [], ztypes: [], gtypes: [], alleles: [], unordered_genotype_labels: [] }"
            />
            <MigrationPanel :detail="spatial.migrationDetail" />
          </div>
        </template>
        <NText
          v-else
          depth="3"
        >
          Select a deme on the Landscape tab first.
        </NText>
      </template>

      <!-- KeepAlive preserves panel-local state across tab switches,
           matching the panmictic dashboard. -->
      <KeepAlive>
        <ConfigPanel v-if="tabValue === 'config'" />
        <HooksPanel v-else-if="tabValue === 'hooks'" />
        <GeneticsPanel v-else-if="tabValue === 'genetics'" />
        <DebugPanel
          v-else-if="tabValue === 'debug'"
          dashboard-type="spatial"
        />
      </KeepAlive>
    </NSpace>
    <NText
      v-else-if="bootError"
      type="error"
    >
      Dashboard failed to load: {{ bootError }}
    </NText>
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

.deme-grid {
  display: grid;
  grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
  gap: 16px;
}

@media (max-width: 1000px) {
  .deme-grid {
    grid-template-columns: 1fr;
  }
}
</style>
