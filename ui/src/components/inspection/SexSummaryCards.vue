<script setup lang="ts">
/**
 * Female / Male / Total summary cards for one state snapshot.
 */
import { computed } from "vue";
import { NCard } from "naive-ui";

import type { StateSnapshot } from "../../api/types";

const props = defineProps<{
  snapshot: StateSnapshot;
}>();

function fmt(value: number): string {
  return Math.round(value).toLocaleString();
}

const stats = computed(() => [
  { label: "Female", value: fmt(props.snapshot.female), color: "#d6418f" },
  { label: "Male", value: fmt(props.snapshot.male), color: "#2080f0" },
  { label: "Total", value: fmt(props.snapshot.total), color: "#18a058" },
]);
</script>

<template>
  <div class="summary-row">
    <NCard
      v-for="stat in stats"
      :key="stat.label"
      size="small"
      class="stat-card"
    >
      <div class="stat">
        <span
          class="stat-label"
          :style="{ color: stat.color }"
        >{{ stat.label }}</span>
        <span
          class="stat-value"
          :style="{ color: stat.color }"
        >{{ stat.value }}</span>
      </div>
    </NCard>
  </div>
</template>

<style scoped>
.summary-row {
  display: flex;
  gap: 12px;
}

.stat-card {
  flex: 1;
}

.stat {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
}

.stat-label {
  font-weight: 600;
}

.stat-value {
  font-family: monospace;
  font-size: 1.2rem;
  font-weight: 700;
}
</style>
