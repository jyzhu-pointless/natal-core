<script setup lang="ts">
/**
 * Inspection tab: mode banner (LIVE vs INSPECTING), tick navigation, sex/age
 * summaries, and the genotype card grid for the displayed snapshot.
 */
import { computed } from "vue";
import {
  NButton,
  NCard,
  NInputNumber,
  NSpace,
  NTag,
  NText,
} from "naive-ui";

import AgeTable from "./AgeTable.vue";
import GenotypeCard from "./GenotypeCard.vue";
import SexSummaryCards from "./SexSummaryCards.vue";
import type { StateSnapshot } from "../../api/types";
import type { RegistryPayload } from "../../api/types";

const props = defineProps<{
  snapshot: StateSnapshot;
  inspectedTick: number | null;
  registry: RegistryPayload;
}>();

const emit = defineEmits<{
  "go-to-tick": [tick: number];
  "back-to-live": [];
  "restore-here": [];
}>();

const svgByLabel = computed(() => {
  const map = new Map<string, string>();
  for (const entry of props.registry.genotypes) {
    map.set(entry.label, entry.svg);
  }
  return map;
});

function goToTick(value: number | null): void {
  if (value !== null) {
    emit("go-to-tick", value);
  }
}
</script>

<template>
  <NSpace
    vertical
    size="large"
  >
    <NSpace align="center">
      <NTag
        :type="inspectedTick === null ? 'success' : 'warning'"
        size="large"
      >
        {{
          inspectedTick === null
            ? "LIVE VIEW"
            : `INSPECTING HISTORY (Tick ${inspectedTick})`
        }}
      </NTag>
      <NInputNumber
        size="small"
        :value="snapshot.tick"
        :min="0"
        style="width: 140px"
        @update:value="goToTick"
      />
      <NButton
        size="small"
        :disabled="inspectedTick === null"
        @click="emit('back-to-live')"
      >
        Back to live
      </NButton>
      <NButton
        size="small"
        type="warning"
        :disabled="inspectedTick === null"
        @click="emit('restore-here')"
      >
        Restore here (time travel)
      </NButton>
    </NSpace>

    <NText
      v-if="inspectedTick !== null && !snapshot.found"
      type="warning"
    >
      Tick {{ inspectedTick }} is not in the recorded history — showing live
      state instead.
    </NText>

    <SexSummaryCards :snapshot="snapshot" />

    <NCard
      v-if="snapshot.is_age_structured"
      title="Age distribution"
      size="small"
    >
      <AgeTable :snapshot="snapshot" />
    </NCard>

    <div class="genotype-grid">
      <GenotypeCard
        v-for="row in snapshot.genotypes"
        :key="row.index"
        :row="row"
        :svg="svgByLabel.get(row.label) ?? ''"
        :is-age-structured="snapshot.is_age_structured"
      />
    </div>
  </NSpace>
</template>

<style scoped>
.genotype-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 12px;
}
</style>
