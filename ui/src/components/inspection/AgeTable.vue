<script setup lang="ts">
/**
 * Per-age Female/Male/Total table (age-structured populations only).
 */
import { computed } from "vue";
import { NTable } from "naive-ui";

import type { StateSnapshot } from "../../api/types";

const props = defineProps<{
  snapshot: StateSnapshot;
}>();

interface AgeRow {
  age: number;
  female: number;
  male: number;
  total: number;
}

const rows = computed<AgeRow[]>(() =>
  props.snapshot.female_per_age.map((female, age) => {
    const male = props.snapshot.male_per_age[age] ?? 0;
    return { age, female, male, total: female + male };
  }),
);

function fmt(value: number): string {
  return Math.round(value).toLocaleString();
}
</script>

<template>
  <NTable
    size="small"
    :bordered="false"
    :single-line="false"
  >
    <thead>
      <tr>
        <th>Age</th>
        <th style="color: #d6418f">
          Female
        </th>
        <th style="color: #2080f0">
          Male
        </th>
        <th>Total</th>
      </tr>
    </thead>
    <tbody>
      <tr
        v-for="row in rows"
        :key="row.age"
      >
        <td>{{ row.age }}</td>
        <td>{{ fmt(row.female) }}</td>
        <td>{{ fmt(row.male) }}</td>
        <td>{{ fmt(row.total) }}</td>
      </tr>
    </tbody>
  </NTable>
</template>
