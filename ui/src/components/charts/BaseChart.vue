<script setup lang="ts">
/**
 * Shared ECharts wrapper: single place that registers renderers/components
 * and relays chart events (used for click-to-inspect on tick axes).
 */
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { BarChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import VChart from "vue-echarts";

use([
  CanvasRenderer,
  LineChart,
  BarChart,
  HeatmapChart,
  ScatterChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  VisualMapComponent,
]);

defineProps<{
  option: Record<string, unknown>;
  height?: string;
}>();

const emit = defineEmits<{
  "item-click": [params: unknown];
  "data-zoom": [params: unknown];
}>();
</script>

<template>
  <VChart
    class="base-chart"
    :option="option"
    :style="{ height: height ?? '300px' }"
    autoresize
    @click="(params: unknown) => emit('item-click', params)"
    @datazoom="(params: unknown) => emit('data-zoom', params)"
  />
</template>

<style scoped>
.base-chart {
  width: 100%;
}
</style>
