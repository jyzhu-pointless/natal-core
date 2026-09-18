<script setup lang="ts">
/**
 * Playback speed: delay between ticks in ms.  0 = turbo (server runs batches
 * of up to 50 ticks per 100ms window).
 */
import { computed } from "vue";
import { NSlider } from "naive-ui";

import { useSimulationStore } from "../../stores/simulation";

const sim = useSimulationStore();

// Discrete slider stops keep the mapping readable (0 = turbo).
const marks: Record<number, string> = {
  0: "Turbo",
  250: "250ms",
};

const value = computed({
  get: () => sim.intervalMs,
  set: (next: number | null) => {
    if (next !== null) {
      sim.setIntervalMs(next);
    }
  },
});
</script>

<template>
  <div class="speed">
    <div class="speed-label">
      Interval
      <span class="speed-value">{{ value === 0 ? "turbo" : `${value} ms` }}</span>
    </div>
    <NSlider
      v-model:value="value"
      :step="10"
      :min="0"
      :max="500"
      :marks="marks"
      :tooltip="false"
    />
  </div>
</template>

<style scoped>
.speed-label {
  font-size: 12px;
  color: #909090;
  text-transform: uppercase;
  font-weight: 700;
  margin-bottom: 4px;
  display: flex;
  justify-content: space-between;
}

.speed-value {
  font-family: monospace;
  text-transform: none;
}
</style>
