<script setup lang="ts">
/**
 * Fertilization outcomes as an annotated heatmap.
 *
 * Color encodes a *meaningful* continuous quantity — the probability of the
 * most probable offspring (1 = deterministic outcome, 0 = spread across many
 * zygotes) — while each cell is annotated with the winning zygote's label.
 * Hover still lists every outcome above 1%.  (The legacy view colored cells
 * by the winning zygote's index, whose colorbar was meaningless.)
 */
import { computed } from "vue";
import { NAlert } from "naive-ui";

import BaseChart from "./BaseChart.vue";
import type { FertilizationMatrix } from "../../api/types";

const props = defineProps<{
  matrix: FertilizationMatrix;
}>();

const n = computed(() => props.matrix.col_labels.length);

interface HeatCell {
  value: [number, number, number];
  winner: string;
}

const data = computed<HeatCell[]>(() => {
  const cells: HeatCell[] = [];
  props.matrix.primary_probability.forEach((row, y) => {
    row.forEach((probability, x) => {
      if (!Number.isNaN(probability)) {
        const primary = props.matrix.primary_index[y]?.[x] ?? NaN;
        cells.push({
          value: [x, y, probability],
          winner:
            Number.isNaN(primary)
              ? ""
              : (props.matrix.zygote_labels[primary] ?? ""),
        });
      }
    });
  });
  return cells;
});

const option = computed(() => ({
  animation: false,
  title: {
    text: "Fertilization outcomes (maternal x paternal)",
    subtext: "color = probability of the most probable offspring",
    left: "center",
    textStyle: { fontSize: 14 },
    subtextStyle: { fontSize: 11, color: "#888" },
  },
  tooltip: {
    formatter: (raw: unknown) => {
      const params = raw as { data: HeatCell; value: [number, number, number] };
      const [x, y] = params.data.value;
      const text = props.matrix.cell_text[y]?.[x] ?? "";
      return (
        `<b>${props.matrix.row_labels[y]} x ${props.matrix.col_labels[x]}</b><br/>` +
        (text ? text.replaceAll("\n", "<br/>") : "(no viable outcome)")
      );
    },
  },
  grid: { left: 150, right: 110, top: 60, bottom: 100 },
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
    text: ["certain", "spread"],
    inRange: { color: ["#440154", "#31688e", "#35b779", "#fde725"] },
  },
  series: [
    {
      type: "heatmap",
      data: data.value,
      label: {
        show: true,
        fontSize: 9,
        formatter: (raw: unknown) => {
          const params = raw as { data: HeatCell };
          return params.data.winner;
        },
      },
    },
  ],
}));
</script>

<template>
  <NAlert
    v-if="matrix.too_large"
    type="warning"
  >
    Fertilization matrix too large to render (more than 40 gamete types).
  </NAlert>
  <BaseChart
    v-else
    :option="option"
    :height="`${Math.max(420, n * 32)}px`"
  />
</template>
