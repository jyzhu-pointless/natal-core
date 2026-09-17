<script setup lang="ts">
/**
 * Primary execution controls: play/pause, step, run-to-tick breakpoint,
 * reset.  All commands go through the simulation store's WebSocket.
 */
import { computed, ref } from "vue";
import {
  NButton,
  NInputNumber,
  NPopconfirm,
  NSpace,
} from "naive-ui";

import { useSimulationStore } from "../../stores/simulation";

const sim = useSimulationStore();

const runToTarget = ref<number>(0);

const playLabel = computed(() =>
  sim.status === "running" ? "Pause" : "Play",
);

function togglePlay(): void {
  if (sim.status === "running") {
    sim.pause();
  } else {
    sim.play();
  }
}

function runToTick(): void {
  if (runToTarget.value > sim.liveTick) {
    sim.runToTick(runToTarget.value);
  }
}
</script>

<template>
  <NSpace
    vertical
    size="small"
  >
    <NSpace>
      <NButton
        type="primary"
        size="small"
        :disabled="sim.wsStatus !== 'open'"
        @click="togglePlay"
      >
        {{ playLabel }}
      </NButton>
      <NButton
        size="small"
        :disabled="sim.wsStatus !== 'open' || sim.status === 'finished'"
        @click="sim.step(1)"
      >
        Step
      </NButton>
      <NButton
        size="small"
        :disabled="sim.wsStatus !== 'open' || sim.status === 'finished'"
        @click="sim.step(10)"
      >
        +10
      </NButton>
    </NSpace>

    <div class="run-to-row">
      <NButton
        size="small"
        :disabled="sim.wsStatus !== 'open'"
        @click="runToTick"
      >
        Run to
      </NButton>
      <NInputNumber
        v-model:value="runToTarget"
        size="small"
        :min="0"
        :show-button="false"
        placeholder="tick"
        style="flex: 1"
      />
    </div>

    <NPopconfirm @positive-click="sim.reset()">
      <template #trigger>
        <NButton
          size="small"
          type="warning"
          :disabled="sim.wsStatus !== 'open'"
          style="width: 100%"
        >
          Reset
        </NButton>
      </template>
      Reset to the initial state and clear all recorded history?
    </NPopconfirm>
  </NSpace>
</template>

<style scoped>
.run-to-row {
  display: flex;
  gap: 8px;
}
</style>
