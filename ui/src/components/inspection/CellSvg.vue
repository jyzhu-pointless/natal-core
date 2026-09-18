<script setup lang="ts">
/**
 * Renders the genotype cell SVG produced by the server registry
 * (visualization.py output — trusted, repo-owned markup).
 */
defineProps<{
  svg: string;
  size?: number;
}>();
</script>

<template>
  <div
    class="cell-svg"
    :style="{ width: `${size ?? 80}px` }"
    v-html="svg"
  />
</template>

<style scoped>
.cell-svg :deep(svg) {
  width: 100%;
  height: auto;
  display: block;
}
</style>
