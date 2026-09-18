<script setup lang="ts">
/**
 * Migration view: outbound edges of the selected source deme, best-first by
 * weight, with normalized share and destination load.
 */
import { computed } from "vue";
import { NCard, NEmpty, NTable, NTag } from "naive-ui";

import type { SpatialMigrationDetail } from "../../api/types";

const props = defineProps<{
  detail: SpatialMigrationDetail | null;
}>();

const entries = computed(() =>
  [...(props.detail?.entries ?? [])].sort((a, b) => b.weight - a.weight),
);

function fmt(value: number): string {
  return Math.round(value).toLocaleString();
}

function pct(share: number): string {
  return `${(share * 100).toFixed(1)}%`;
}
</script>

<template>
  <NCard
    title="Migration"
    size="small"
  >
    <NEmpty
      v-if="!detail"
      description="Select a deme to inspect its outbound migration."
    />
    <template v-else>
      <NTag
        size="small"
        style="margin-bottom: 8px"
      >
        mean rate {{ detail.rate_mean.toFixed(4) }}
      </NTag>
      <NTable
        size="small"
        :single-line="false"
      >
        <thead>
          <tr>
            <th>To</th>
            <th>Share</th>
            <th>Weight</th>
            <th>Dest total</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="entry in entries"
            :key="entry.dest"
          >
            <td>{{ entry.dest_name }}</td>
            <td>{{ pct(entry.share) }}</td>
            <td>{{ entry.weight.toFixed(4) }}</td>
            <td>{{ fmt(entry.dest_total) }}</td>
          </tr>
        </tbody>
      </NTable>
    </template>
  </NCard>
</template>
