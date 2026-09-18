<script setup lang="ts">
/**
 * Configuration tab: scalar parameters, fitness tables (viability,
 * fecundity, sexual selection), and the preset summary.
 */
import { computed } from "vue";
import { NCard, NTable, NTag, NText } from "naive-ui";

import { useDomainStore } from "../../stores/domain";

const domain = useDomainStore();

const scalars = computed(() => domain.config?.scalars ?? null);

const scalarRows = computed(() => {
  if (!scalars.value) {
    return [];
  }
  const s = scalars.value;
  return [
    ["Stochastic", s.stochastic ? "yes" : "no"],
    ["Continuous sampling", s.continuous_sampling ? "yes" : "no"],
    ["Discrete generation", s.discrete_generation ? "yes" : "no"],
    ["Sexes / Ages", `${s.n_sexes} / ${s.n_ages}`],
    ["Genotypes / Gametes", `${s.n_genotypes} / ${s.n_gtypes}`],
    ["New adult age", s.new_adult_age],
    ["Carrying capacity", s.carrying_capacity.toLocaleString()],
    ["Eggs per female", s.eggs_per_female],
    ["Sex ratio", s.sex_ratio],
    ["Sperm displacement", s.sperm_displacement_rate],
    ["Low-density growth rate", s.low_density_growth_rate],
    ["Growth mode", `${s.juvenile_growth_mode.name} (${s.juvenile_growth_mode.code})`],
    ["Fixed egg count", s.fixed_egg_count ? "yes" : "no"],
  ] as Array<[string, string | number]>;
});

function fmtFitness(value: number): string {
  return value === Math.trunc(value) ? String(value) : value.toFixed(4);
}
</script>

<template>
  <NSpace
    vertical
    size="large"
  >
    <NCard
      title="Parameters"
      size="small"
    >
      <NTable
        size="small"
        :bordered="false"
        :single-line="false"
      >
        <tbody>
          <tr
            v-for="[key, value] in scalarRows"
            :key="key"
          >
            <td class="param-key">
              {{ key }}
            </td>
            <td>{{ value }}</td>
          </tr>
        </tbody>
      </NTable>
    </NCard>

    <NCard
      title="Viability fitness"
      size="small"
    >
      <NTable
        size="small"
        :single-line="false"
      >
        <thead>
          <tr>
            <th>Genotype</th>
            <th>Age</th>
            <th>Female</th>
            <th>Male</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in domain.config?.viability"
            :key="row.genotype"
          >
            <td>{{ row.genotype }}</td>
            <td>{{ row.age }}</td>
            <td>{{ fmtFitness(row.female) }}</td>
            <td>{{ fmtFitness(row.male) }}</td>
          </tr>
        </tbody>
      </NTable>
    </NCard>

    <NCard
      title="Fecundity fitness"
      size="small"
    >
      <NTable
        size="small"
        :single-line="false"
      >
        <thead>
          <tr>
            <th>Genotype</th>
            <th>Female</th>
            <th>Male</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in domain.config?.fecundity"
            :key="row.genotype"
          >
            <td>{{ row.genotype }}</td>
            <td>{{ fmtFitness(row.female) }}</td>
            <td>{{ fmtFitness(row.male) }}</td>
          </tr>
        </tbody>
      </NTable>
    </NCard>

    <NCard
      v-if="domain.config?.sexual_selection.length"
      title="Sexual selection"
      size="small"
    >
      <NTable
        size="small"
        :single-line="false"
      >
        <thead>
          <tr>
            <th>Female genotype</th>
            <th>Male genotype</th>
            <th>Preference</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in domain.config?.sexual_selection"
            :key="`${row.female_genotype}|${row.male_genotype}`"
          >
            <td>{{ row.female_genotype }}</td>
            <td>{{ row.male_genotype }}</td>
            <td>{{ fmtFitness(row.preference) }}</td>
          </tr>
        </tbody>
      </NTable>
    </NCard>

    <NCard
      title="Presets"
      size="small"
    >
      <NTag
        v-for="preset in domain.config?.presets.presets"
        :key="preset.preset_name"
        style="margin-right: 8px"
      >
        {{ preset.preset_name }} ({{ preset.gamete_modifiers.length }} gamete /
        {{ preset.zygote_modifiers.length }} zygote modifiers)
      </NTag>
      <NText
        v-if="!domain.config?.presets.preset_count"
        depth="3"
      >
        No presets registered.
      </NText>
    </NCard>
  </NSpace>
</template>

<style scoped>
.param-key {
  color: #777;
  width: 40%;
}
</style>
