<script setup lang="ts">
/**
 * Dashboard shell: header bar (identity + live status), control drawer
 * (execution + history settings + export), and the tabbed main area.
 * Panels are provided by the parent via slots.
 */
import { computed } from "vue";
import {
  NButton,
  NCard,
  NLayout,
  NLayoutContent,
  NLayoutSider,
  NSpace,
  NTag,
  NText,
} from "naive-ui";

import ControlBar from "../controls/ControlBar.vue";
import ExportDialog from "../controls/ExportDialog.vue";
import HistorySettings from "../controls/HistorySettings.vue";
import SpeedSlider from "../controls/SpeedSlider.vue";
import { useSimulationStore } from "../../stores/simulation";

const sim = useSimulationStore();

defineProps<{
  tabValue: string;
  tabs: Array<{ name: string; label: string }>;
}>();

const emit = defineEmits<{
  "update:tabValue": [value: string];
}>();

const statusType = computed(() => {
  if (sim.status === "running") return "success";
  if (sim.status === "error") return "error";
  if (sim.status === "finished") return "info";
  return "default";
});
</script>

<template>
  <NLayout>
    <NLayoutHeader
      bordered
      style="padding: 10px 20px"
    >
      <NSpace
        align="center"
        justify="space-between"
      >
        <NSpace align="center">
          <NText strong>
            🧬 {{ sim.meta?.title ?? "NATAL Dashboard" }}
          </NText>
          <NText
            v-if="sim.meta"
            depth="3"
          >
            {{ sim.meta.population_name }} ·
            {{ sim.meta.dashboard_type }} · backend: {{ sim.meta.backend }}
          </NText>
        </NSpace>
        <NSpace align="center">
          <NTag
            :type="sim.wsStatus === 'open' ? 'success' : 'warning'"
            size="small"
          >
            WS {{ sim.wsStatus }}
          </NTag>
          <NTag
            :type="statusType"
            size="small"
          >
            {{ sim.status }} @ tick {{ sim.liveTick }}
          </NTag>
        </NSpace>
      </NSpace>
    </NLayoutHeader>

    <NLayout has-sider>
      <NLayoutSider
        bordered
        :width="280"
        content-style="padding: 16px"
      >
        <NSpace
          vertical
          size="large"
        >
          <NCard
            title="Execution"
            size="small"
          >
            <ControlBar />
          </NCard>
          <NCard
            title="Speed"
            size="small"
          >
            <SpeedSlider />
          </NCard>
          <NCard
            title="History settings"
            size="small"
          >
            <HistorySettings />
          </NCard>
          <ExportDialog />
        </NSpace>
      </NLayoutSider>

      <NLayoutContent
        content-style="padding: 16px 24px"
        style="min-height: calc(100vh - 56px)"
      >
        <div class="tab-bar">
          <NButton
            v-for="tab in tabs"
            :key="tab.name"
            size="small"
            :type="tabValue === tab.name ? 'primary' : 'default'"
            style="margin-right: 8px"
            @click="emit('update:tabValue', tab.name)"
          >
            {{ tab.label }}
          </NButton>
        </div>
        <slot />
      </NLayoutContent>
    </NLayout>
  </NLayout>
</template>

<style scoped>
.tab-bar {
  margin-bottom: 16px;
}
</style>
