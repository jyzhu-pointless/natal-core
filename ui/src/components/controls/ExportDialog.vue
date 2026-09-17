<script setup lang="ts">
/**
 * Export dialog: builds the legacy-compatible JSON export server-side and
 * downloads it through a Blob link.
 */
import { computed, ref } from "vue";
import {
  NButton,
  NCheckbox,
  NModal,
  NSpace,
  useMessage,
} from "naive-ui";

import { useSimulationStore } from "../../stores/simulation";

const sim = useSimulationStore();
const message = useMessage();

const visible = ref(false);
const includeConfig = ref(true);
const includeHistory = ref(true);
const includeHooks = ref(true);
const busy = ref(false);

const filename = computed(
  () =>
    `natal_export_${sim.meta?.population_name ?? "population"}_${sim.liveTick}.json`,
);

async function doExport(): Promise<void> {
  busy.value = true;
  try {
    const params = new URLSearchParams({
      config: includeConfig.value ? "1" : "0",
      history: includeHistory.value ? "1" : "0",
      hooks: includeHooks.value ? "1" : "0",
    });
    const response = await fetch(`/api/export?${params.toString()}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename.value;
    anchor.click();
    URL.revokeObjectURL(url);
    visible.value = false;
  } catch (error) {
    message.error(`Export failed: ${String(error)}`);
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <NButton
    size="small"
    block
    @click="visible = true"
  >
    Export JSON
  </NButton>

  <NModal
    v-model:show="visible"
    preset="card"
    title="Select items to export"
    style="width: 360px"
  >
    <NSpace
      vertical
      size="medium"
    >
      <NCheckbox v-model:checked="includeConfig">
        Configuration & Fitness
      </NCheckbox>
      <NCheckbox v-model:checked="includeHistory">
        Population History
      </NCheckbox>
      <NCheckbox v-model:checked="includeHooks">
        Hooks
      </NCheckbox>
      <NSpace justify="end">
        <NButton
          size="small"
          @click="visible = false"
        >
          Cancel
        </NButton>
        <NButton
          size="small"
          type="primary"
          :loading="busy"
          @click="doExport"
        >
          Export
        </NButton>
      </NSpace>
    </NSpace>
  </NModal>
</template>
