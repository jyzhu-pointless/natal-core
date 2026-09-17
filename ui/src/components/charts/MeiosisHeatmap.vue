<script setup lang="ts">
/**
 * Meiosis probability heatmap: parent genotypes (rows) x gametes (cols).
 */
import { computed } from "vue";

import BaseChart from "./BaseChart.vue";
import type { Matrix2D } from "../../api/types";

const props = defineProps<{
  title: string;
  matrix: Matrix2D;
}>();

const option = computed(() => {
  const data: Array<[number, number, number]> = [];
  props.matrix.data.forEach((row, y) => {
    row.forEach((value, x) => {
      data.push([x, y, value]);
    });
  });
  return {
    animation: false,
    title: { text: props.title, left: "center", textStyle: { fontSize: 14 } },
    tooltip: {
      formatter: (raw: unknown) => {
        const params = raw as { value: [number, number, number] };
        const [x, y, v] = params.value;
        return `${props.matrix.row_labels[y]} → ${props.matrix.col_labels[x]}: ${Number(v).toFixed(4)}`;
      },
    },
    grid: { left: 140, right: 90, top: 40, bottom: 90 },
    xAxis: {
      type: "category",
      data: props.matrix.col_labels,
      axisLabel: { rotate: 45, fontSize: 10 },
    },
    yAxis: {
      type: "category",
      data: props.matrix.row_labels,
      axisLabel: { fontSize: 10 },
    },
    visualMap: {
      min: 0,
      max: 1,
      calculable: true,
      orient: "vertical",
      right: 8,
      top: "center",
      inRange: { color: ["#313695", "#fee090", "#a50026"] },
    },
    series: [
      {
        type: "heatmap",
        data,
        label: { show: false },
      },
    ],
  };
});
</script>

<template>
  <BaseChart
    :option="option"
    height="380px"
  />
</template>
