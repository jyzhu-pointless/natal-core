<script setup lang="ts">
/**
 * Landscape map: one glyph per deme (hexagon / square / fallback circle),
 * colored by the selected metric, click to select a deme.
 *
 * Rendered as a standard scatter series with an SVG path symbol — the
 * mature ECharts pipeline (visualMap color mapping, click events) instead
 * of a custom renderItem, which proved unreliable across renderer paths.
 * HexGrid centers are unit-spaced (pointy-top), matching the server xy.
 */
import { computed, onMounted, onUnmounted, ref } from "vue";

import BaseChart from "./BaseChart.vue";
import type { LandscapeMetric, SpatialLandscapePayload } from "../../api/types";

const props = defineProps<{
  landscape: SpatialLandscapePayload;
  metric: LandscapeMetric;
  selected: number | null;
}>();

// Grid layout constants shared between the option and the click resolver.
const GRID = { left: 24, right: 96, top: 24, bottom: 24 };

// Upper bound for the hexagon's on-screen width in px: the isotropic scale
// factor k equals the hex width in px, so capping k caps the glyph size.
// Extra plot area centers the lattice instead of enlarging it.
const MAX_HEX_WIDTH_PX = 72;

const emit = defineEmits<{
  "deme-select": [index: number];
}>();

const values = computed<number[]>(() => {
  const land = props.landscape;
  switch (props.metric.kind) {
    case "total":
      return land.totals;
    case "female":
      return land.females;
    case "male":
      return land.males;
    case "genotype": {
      const index = land.genotype_labels.indexOf(props.metric.label);
      return land.genotype_counts[index] ?? land.totals.map(() => 0);
    }
    case "allele": {
      const index = land.allele_names.indexOf(props.metric.name);
      return land.allele_frequencies[index] ?? land.totals.map(() => 0);
    }
    default:
      return land.totals;
  }
});

const isFrequency = computed(
  () => props.metric.kind === "genotype" || props.metric.kind === "allele",
);

const valueRange = computed<[number, number]>(() => {
  if (isFrequency.value) {
    return [0, 1];
  }
  const defined = values.value.filter((v) => Number.isFinite(v));
  if (!defined.length) {
    return [0, 1];
  }
  return [Math.min(...defined), Math.max(...defined)];
});

/** Fallback layout when the population has no topology: a simple grid. */
const positions = computed<number[][]>(() => {
  const xy = props.landscape.topology.xy;
  if (xy) {
    return xy;
  }
  const n = props.landscape.n_demes;
  const cols = Math.ceil(Math.sqrt(n));
  return Array.from({ length: n }, (_, index) => [
    index % cols,
    Math.floor(index / cols),
  ]);
});

const extent = computed(() => {
  const xs = positions.value.map((p) => p[0] ?? 0);
  const ys = positions.value.map((p) => p[1] ?? 0);
  // Padding must exceed the hex circumradius (0.5 in x, 0.577 in y) so
  // border glyphs are not clipped.
  return {
    xMin: Math.min(0, ...xs) - 0.6,
    xMax: Math.max(1, ...xs) + 0.6,
    yMin: Math.min(0, ...ys) - 0.7,
    yMax: Math.max(1, ...ys) + 0.7,
  };
});

// Measured container size: glyph pixel sizes derive from the actual axis
// scale (pixels per data unit), because a hex tiling only stays seamless
// when each glyph's bounding box matches the on-screen lattice spacing.
const mapEl = ref<HTMLElement | null>(null);
const mapSize = ref({ w: 0, h: 0 });
let resizeObserver: ResizeObserver | null = null;

onMounted(() => {
  resizeObserver = new ResizeObserver((entries) => {
    const rect = entries[0]?.contentRect;
    if (rect) {
      mapSize.value = { w: rect.width, h: rect.height };
    }
  });
  if (mapEl.value) {
    resizeObserver.observe(mapEl.value);
  }
});

onUnmounted(() => {
  resizeObserver?.disconnect();
  resizeObserver = null;
});

// Isotropic axis scale: hexagons must render as REGULAR hexagons, so x and
// y share one pixels-per-data-unit factor k.  k = min(...) guarantees the
// whole lattice fits (the other axis may center with some whitespace); the
// container height adapts to the lattice aspect ratio so that in the common
// case both axes fill exactly and k stays maximal.
const axisScale = computed(() => {
  const innerW = Math.max(1, mapSize.value.w - GRID.left - GRID.right);
  const innerH = Math.max(1, mapSize.value.h - GRID.top - GRID.bottom);
  const xSpan = extent.value.xMax - extent.value.xMin;
  const ySpan = extent.value.yMax - extent.value.yMin;
  const k = Math.min(innerW / xSpan, innerH / ySpan, MAX_HEX_WIDTH_PX);
  const xMid = (extent.value.xMax + extent.value.xMin) / 2;
  const yMid = (extent.value.yMax + extent.value.yMin) / 2;
  return {
    k,
    xMin: xMid - innerW / k / 2,
    xMax: xMid + innerW / k / 2,
    yMin: yMid - innerH / k / 2,
    yMax: yMid + innerH / k / 2,
  };
});

// Container height tracks the lattice aspect ratio (clamped for extreme
// landscapes), so the isotropic scale wastes no space.
const mapHeight = computed(() => {
  const w = mapSize.value.w;
  if (!w) {
    return 420;
  }
  const innerW = Math.max(1, w - GRID.left - GRID.right);
  const xSpan = extent.value.xMax - extent.value.xMin;
  const ySpan = extent.value.yMax - extent.value.yMin;
  // Height that fits the lattice at the capped glyph size.
  const kForHeight = Math.min(innerW / xSpan, MAX_HEX_WIDTH_PX);
  const needed = Math.round(ySpan * kForHeight + GRID.top + GRID.bottom);
  return Math.min(560, Math.max(200, needed));
});

const option = computed(() => {
  const land = props.landscape;
  const kind = land.topology.kind;
  const cells = positions.value.map((position, index) => ({
    value: [position[0] ?? 0, position[1] ?? 0, values.value[index] ?? 0],
    name: land.deme_names[index] ?? String(index),
    itemStyle:
      props.selected === index
        ? { borderColor: "#ff7d00", borderWidth: 3, color: undefined }
        : undefined,
  }));

  return {
    animation: false,
    tooltip: {
      formatter: (raw: unknown) => {
        const params = raw as { dataIndex: number; value: [number, number, number] };
        const index = params.dataIndex;
        const label = land.deme_names[index] ?? `deme ${index}`;
        const grid = land.topology.grid_ij?.[index];
        const coord = grid ? ` @ (${grid[0]}, ${grid[1]})` : "";
        const value = params.value[2] ?? 0;
        const text = isFrequency.value
          ? (value * 100).toFixed(1) + "%"
          : Math.round(value).toLocaleString();
        return `<b>${label}</b>${coord}<br/>${metricLabel.value}: ${text}`;
      },
    },
    grid: { ...GRID },
    xAxis: {
      type: "value",
      min: axisScale.value.xMin,
      max: axisScale.value.xMax,
      show: false,
    },
    yAxis: {
      type: "value",
      min: axisScale.value.yMin,
      max: axisScale.value.yMax,
      show: false,
      inverse: true,
    },
    visualMap: {
      // Explicit dimension: the metric lives in value slot 2; without this
      // ECharts maps the last encode dim (tooltip) and colors come out NaN.
      dimension: 2,
      min: valueRange.value[0],
      max: valueRange.value[1],
      calculable: true,
      orient: "vertical",
      right: 8,
      top: "center",
      inRange: { color: ["#440154", "#31688e", "#35b779", "#fde725"] },
    },
    series: [
      {
        type: "scatter",
        symbol: kind === "hex" ? HEX_SYMBOL : kind === "square" ? "rect" : "circle",
        // Regular hexagon at the isotropic scale k: data-space bounding box
        // is width sqrt(3)*s = 1, height 2s = 2/sqrt(3) (edge s = 1/sqrt(3),
        // centre pitch 1).  The 2% overshoot hides the antialiasing seam.
        symbolSize:
          kind === "hex"
            ? [axisScale.value.k * 1.02, axisScale.value.k * (2 / Math.sqrt(3)) * 1.02]
            : kind === "square"
              ? [axisScale.value.k * 0.99, axisScale.value.k * 0.99]
              : 14,
        data: cells,
        encode: { x: 0, y: 1, tooltip: 2 },
      },
    ],
  };
});

// Pointy-top hexagon (unit bounding box 1x1), filled by itemStyle.color.
const HEX_SYMBOL =
  "path://M0.5 0 L0.933 0.25 L0.933 0.75 L0.5 1 L0.067 0.75 L0.067 0.25 Z";

const metricLabel = computed(() => {
  switch (props.metric.kind) {
    case "total":
      return "Total";
    case "female":
      return "Female";
    case "male":
      return "Male";
    case "genotype":
      return `${props.metric.label} frequency`;
    case "allele":
      return `${props.metric.name} frequency`;
    default:
      return "";
  }
});

function onMapClick(event: MouseEvent): void {
  // Resolve the click to the nearest deme center in grid coordinates.
  // (Self-managed hit test — ECharts click events proved unreliable for
  // scatter symbols across renderer paths.)
  const target = event.currentTarget as HTMLElement;
  const canvas = target.querySelector("canvas");
  if (!canvas) {
    return;
  }
  if (event.offsetX < GRID.left || event.offsetY < GRID.top) {
    return;
  }
  const gx = (event.offsetX - GRID.left) / axisScale.value.k + axisScale.value.xMin;
  const gy = (event.offsetY - GRID.top) / axisScale.value.k + axisScale.value.yMin;

  let bestIndex = -1;
  let bestDist = 0.5; // hit radius in grid units
  positions.value.forEach((position, index) => {
    const dx = (position[0] ?? 0) - gx;
    const dy = (position[1] ?? 0) - gy;
    const dist = Math.hypot(dx, dy);
    if (dist < bestDist) {
      bestDist = dist;
      bestIndex = index;
    }
  });
  if (bestIndex >= 0) {
    emit("deme-select", bestIndex);
  }
}
</script>

<template>
  <div
    ref="mapEl"
    :style="{ height: `${mapHeight}px` }"
    @click="onMapClick"
  >
    <BaseChart
      :option="option"
      height="100%"
    />
  </div>
</template>
