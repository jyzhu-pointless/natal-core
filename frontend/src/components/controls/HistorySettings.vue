<script setup lang="ts">
/**
 * History recording settings (record interval + rolling window size),
 * applied live through WebSocket commands.
 */
import { computed } from "vue";
import { NInputNumber } from "naive-ui";

import { useSimulationStore } from "../../stores/simulation";
import { useHistoryStore } from "../../stores/history";

const sim = useSimulationStore();
const history = useHistoryStore();

const recordEvery = computed<number | null>({
  get: () => sim.recordEvery,
  set: (next: number | null) => {
    if (next !== null && next >= 1) {
      sim.setRecordEvery(next);
    }
  },
});

const maxHistory = computed<number | null>({
  get: () => sim.maxHistory,
  set: (next: number | null) => {
    if (next !== null && next >= 10) {
      sim.setMaxHistory(next);
    }
  },
});
</script>

<template>
  <div class="history-settings">
    <div class="settings-grid">
      <label>Record every</label>
      <NInputNumber
        :value="recordEvery"
        size="small"
        :min="1"
        :show-button="false"
        @update:value="recordEvery = $event"
      />
      <label>Max history</label>
      <NInputNumber
        :value="maxHistory"
        size="small"
        :min="10"
        :show-button="false"
        @update:value="maxHistory = $event"
      />
    </div>
    <div class="snapshot-count">
      {{ history.liveSnapshot?.history_len ?? 0 }} snapshots recorded
    </div>
  </div>
</template>

<style scoped>
.settings-grid {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 6px 8px;
  align-items: center;
  font-size: 12px;
  color: #666;
}

.snapshot-count {
  margin-top: 6px;
  font-size: 11px;
  font-style: italic;
  color: #999;
}
</style>
