<script setup lang="ts">
/**
 * Genetics tab: meiosis probability heatmaps (female/male, side by side on
 * wide screens) and the fertilization outcomes matrix.
 */
import { NCard, NSpace } from "naive-ui";

import FertilizationHeatmap from "../charts/FertilizationHeatmap.vue";
import MeiosisHeatmap from "../charts/MeiosisHeatmap.vue";
import { useDomainStore } from "../../stores/domain";

const domain = useDomainStore();

const sexTitle = (index: number): string =>
  index === 0 ? "Female meiosis" : "Male meiosis";
</script>

<template>
  <NSpace
    vertical
    size="large"
  >
    <template v-if="domain.genetics">
      <div class="meiosis-grid">
        <NCard
          v-for="(matrix, index) in domain.genetics.meiosis"
          :key="index"
          size="small"
        >
          <MeiosisHeatmap
            :title="sexTitle(index)"
            :matrix="matrix"
          />
        </NCard>
      </div>
      <NCard size="small">
        <FertilizationHeatmap :matrix="domain.genetics.fertilization" />
      </NCard>
    </template>
  </NSpace>
</template>

<style scoped>
.meiosis-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
  gap: 16px;
}
</style>
