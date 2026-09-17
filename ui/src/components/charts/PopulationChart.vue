<script setup lang="ts">
/**
 * Population totals over ticks.  Clicking a point selects that tick for
 * history inspection; the optional marker highlights the inspected tick.
 */
import { computed } from "vue";

import BaseChart from "./BaseChart.vue";
import type { HistorySeries } from "../../api/types";

const props = defineProps<{
  series: HistorySeries;
  inspectedTick?: number | null;
}>();

const emit = defineEmits<{
  "tick-select": [tick: number];
  "data-zoom": [params: unknown];
}>();

interface ClickParams {
  value: [number, number];
}

const option = computed(() => {
  const series: Array<Record<string, unknown>> = [
    {
      name: "Total",
      type: "line",
      data: props.series.ticks.map((tick, i) => [tick, props.series.total[i] ?? 0]),
      showSymbol: false,
      lineStyle: { width: 2.5, color: "#4098fc" },
      itemStyle: { color: "#4098fc" },
    },
    {
      name: "Female",
      type: "line",
      data: props.series.ticks.map((tick, i) => [tick, props.series.female[i] ?? 0]),
      showSymbol: false,
      lineStyle: { width: 1.2, opacity: 0.65, color: "#f0883a" },
      itemStyle: { color: "#f0883a" },
    },
    {
      name: "Male",
      type: "line",
      data: props.series.ticks.map((tick, i) => [tick, props.series.male[i] ?? 0]),
      showSymbol: false,
      lineStyle: { width: 1.2, opacity: 0.65, color: "#2080f0" },
      itemStyle: { color: "#2080f0" },
    },
  ];
  if (props.inspectedTick != null) {
    const index = props.series.ticks.indexOf(props.inspectedTick);
    if (index >= 0) {
      series.push({
        name: "Inspected",
        type: "line",
        data: [
          [props.inspectedTick, 0],
          [props.inspectedTick, props.series.total[index] ?? 0],
        ],
        showSymbol: false,
        lineStyle: { width: 1.5, type: "dashed", color: "#f0a020" },
        tooltip: { show: false },
        silent: true,
      });
    }
  }
  return {
    animation: false,
    tooltip: { trigger: "axis" },
    legend: { bottom: 0 },
    grid: { left: 70, right: 16, top: 16, bottom: 64 },
    dataZoom: [{ type: "inside" }],
    xAxis: { type: "value", name: "tick", min: "dataMin", max: "dataMax" },
    yAxis: { type: "value", scale: true },
    series,
  };
});

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
