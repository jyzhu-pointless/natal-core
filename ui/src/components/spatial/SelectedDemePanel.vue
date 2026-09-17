<script setup lang="ts">
/**
 * Selected-deme inspection: identity banner, sex counts, per-age table, and
 * the genotype card grid (reuse of the panmictic inspection components).
 */
import { computed } from "vue";
import { NCard, NEmpty, NTag } from "naive-ui";

import GenotypeCard from "../inspection/GenotypeCard.vue";
import type { SpatialDemeDetail } from "../../api/types";
import type { RegistryPayload } from "../../api/types";

const props = defineProps<{
  detail: SpatialDemeDetail;
  registry: RegistryPayload;
}>();

const svgByLabel = computed(() => {
  const map = new Map<string, string>();
  for (const entry of props.registry.genotypes) {
    map.set(entry.label, entry.svg);
  }
  return map;
});

function fmt(value: number): string {
  return Math.round(value).toLocaleString();
}

const ageRows = computed(() => {
  if (!props.detail.is_age_structured) {
    return [];
  }
  return props.detail.female_per_age
    .map((female, age) => ({
      age,
      female,
      male: props.detail.male_per_age[age] ?? 0,
    }))
    .filter((entry) => entry.female > 0 || entry.male > 0);
});
</script>

<template>
  <div v-if="detail">
    <NCard size="small">
      <template #header>
        <NSpace align="center">
          <span>{{ detail.name }}</span>
          <NTag size="small">
            deme {{ detail.index }}
          </NTag>
          <NTag
            v-if="detail.grid_ij"
            size="small"
            :bordered="false"
          >
            grid ({{ detail.grid_ij[0] }}, {{ detail.grid_ij[1] }})
          </NTag>
        </NSpace>
      </template>
      <div class="counts">
        <span class="female">F {{ fmt(detail.female) }}</span>
        <span class="male">M {{ fmt(detail.male) }}</span>
        <span class="total">Total {{ fmt(detail.total) }}</span>
      </div>
    </NCard>

    <NCard
      v-if="ageRows.length"
      title="Age distribution"
      size="small"
      style="margin-top: 12px"
    >
      <table class="age-table">
        <thead>
          <tr>
            <th>Age</th>
            <th>Female</th>
            <th>Male</th>
            <th>Total</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in ageRows"
            :key="row.age"
          >
            <td>{{ row.age }}</td>
            <td>{{ fmt(row.female) }}</td>
            <td>{{ fmt(row.male) }}</td>
            <td>{{ fmt(row.female + row.male) }}</td>
          </tr>
        </tbody>
      </table>
    </NCard>

    <NCard
      title="Genotypes"
      size="small"
      style="margin-top: 12px"
    >
      <NEmpty
        v-if="!detail.genotypes.length"
        description="No genotypes."
      />
      <div
        v-else
        class="genotype-grid"
      >
        <GenotypeCard
          v-for="row in detail.genotypes"
          :key="row.index"
          :row="row"
          :svg="svgByLabel.get(row.label) ?? ''"
          :is-age-structured="detail.is_age_structured"
        />
      </div>
    </NCard>
  </div>
</template>
