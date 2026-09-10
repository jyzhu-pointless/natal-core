<script setup lang="ts">
/**
 * One genotype's inspection card: cell SVG, fitness badges, sex counts, and
 * the per-age breakdown (age-structured populations only).
 */
import { computed } from "vue";

import CellSvg from "./CellSvg.vue";
import type { GenotypeStateRow } from "../../api/types";

const props = defineProps<{
  row: GenotypeStateRow;
  svg: string;
  isAgeStructured: boolean;
}>();

function fmt(value: number): string {
  return Math.round(value).toLocaleString();
}

const viabilityText = computed(() => {
  const [f, m] = props.row.viability;
  return f === 1 && m === 1 ? null : `Via ${f} / ${m}`;
});

const fecundityText = computed(() => {
  const [f, m] = props.row.fecundity;
  return f === 1 && m === 1 ? null : `Fec ${f} / ${m}`;
});

const ageRows = computed(() => {
  if (!props.isAgeStructured) {
    return [];
  }
  return props.row.female_per_age
    .map((female, age) => ({
      age,
      female,
      male: props.row.male_per_age[age] ?? 0,
    }))
    .filter((entry) => entry.female > 0 || entry.male > 0);
});
</script>

<template>
  <div class="genotype-card">
    <CellSvg
      :svg="svg"
      :size="76"
    />
    <div class="label">
      {{ row.label }}
    </div>
    <div
      v-if="viabilityText || fecundityText"
      class="fitness"
    >
      <div v-if="viabilityText">
        {{ viabilityText }}
      </div>
      <div v-if="fecundityText">
        {{ fecundityText }}
      </div>
    </div>
    <div class="counts">
      <span class="female">{{ fmt(row.female) }}</span>
      <span class="male">{{ fmt(row.male) }}</span>
    </div>
    <div
      v-if="ageRows.length"
      class="ages"
    >
      <div
        v-for="entry in ageRows"
        :key="entry.age"
        class="age-row"
      >
        <span>A{{ entry.age }}</span>
        <span>{{ fmt(entry.female) }} / {{ fmt(entry.male) }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.genotype-card {
  border: 1px solid #e0e0e6;
  border-radius: 6px;
  padding: 8px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  background: #fff;
}

.label {
  font-weight: 700;
  text-align: center;
  line-height: 1.1;
}

.fitness {
  font-size: 11px;
  color: #666;
  background: #f7f7fa;
  border-radius: 4px;
  padding: 2px 6px;
  text-align: center;
}

.counts {
  display: flex;
  justify-content: space-between;
  width: 100%;
  font-weight: 700;
}

.female {
  color: #d6418f;
}

.male {
  color: #2080f0;
}

.ages {
  width: 100%;
  font-size: 11px;
  color: #777;
}

.age-row {
  display: flex;
  justify-content: space-between;
}
</style>
