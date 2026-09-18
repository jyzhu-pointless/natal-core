<script setup lang="ts">
import { computed, onMounted } from "vue";
import {
  NConfigProvider,
  NDialogProvider,
  NMessageProvider,
  NSpin,
} from "naive-ui";

import PanmicticDashboard from "./components/views/PanmicticDashboard.vue";
import SpatialDashboard from "./components/views/SpatialDashboard.vue";
import { useSimulationStore } from "./stores/simulation";

const sim = useSimulationStore();

onMounted(() => {
  void sim.initialize();
});

const dashboard = computed(() => sim.meta?.dashboard_type ?? null);
</script>

<template>
  <NConfigProvider>
    <NMessageProvider>
      <NDialogProvider>
        <NSpin
          v-if="dashboard === null"
          style="width: 100%; margin-top: 40vh"
        />
        <SpatialDashboard v-else-if="dashboard === 'spatial'" />
        <PanmicticDashboard v-else />
      </NDialogProvider>
    </NMessageProvider>
  </NConfigProvider>
</template>
