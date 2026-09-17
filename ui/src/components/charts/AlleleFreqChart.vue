<script setup lang="ts">
/**
 * Per-allele frequency curves.  Frequencies are zero-filled by the server so
 * every known allele stays aligned across the whole timeline.  Clicking a
 * point selects that tick for history inspection.
 */
import { computed } from "vue";

import BaseChart from "./BaseChart.vue";
import type { HistorySeries } from "../../api/types";

const props = defineProps<{
  series: HistorySeries;
  alleleColors: Record<string, string>;
}>();

const emit = defineEmits<{
  "tick-select": [tick: number];
  "data-zoom": [params: unknown];
}>();

interface ClickParams {
  value: [number, number];
  seriesName?: string;
}

const option = computed(() => ({
  animation: false,
  tooltip: {
    trigger: "axis",
    valueFormatter: (value: unknown) =>
      typeof value === "number" ? value.toFixed(4) : String(value ?? ""),
  },
  legend: { bottom: 0 },
  grid: { left: 48, right: 16, top: 16, bottom: 64 },
  dataZoom: [{ type: "inside" }],
  xAxis: { type: "value", name: "tick", min: "dataMin", max: "dataMax" },
  yAxis: { type: "value", min: 0, max: 1 },
  series: props.series.known_alleles.map((allele) => ({
    name: allele,
    type: "line",
    showSymbol: false,
    // Locally appended live ticks have no frequency column yet; null keeps
    // the line connected (connectNulls) until the throttled refresh fills it.
    data: props.series.ticks.map(
      (tick, i) =>
        [tick, props.series.allele_frequencies[allele]?.[i] ?? null] as [
          number,
          number | null,
        ],
    ),
    connectNulls: true,
    color: props.alleleColors[allele] ?? "#7f7f7f",
    lineStyle: { width: 1.8 },
  })),
}));

function onClick(raw: unknown): void {
  const params = raw as ClickParams;
  if (Array.isArray(params.value) && typeof params.value[0] === "number") {
    emit("tick-select", params.value[0]);
  }
}
</script>

<template>
  <BaseChart
    :option="option"
    height="320px"
    @item-click="onClick"
    @data-zoom="(params: unknown) => emit('data-zoom', params)"
  />
</template>
