<script setup lang="ts">
/**
 * Observation tab: build ad-hoc observation groups (genotype pattern, sex,
 * age window) and view the resulting counts at the live tick.
 */
import { computed, reactive, ref } from "vue";
import {
  NButton,
  NCard,
  NCheckbox,
  NInputNumber,
  NSelect,
  NSpace,
  NTable,
  NTag,
  NText,
} from "naive-ui";

import { postJson } from "../../api/http";
import type {
  ObservationGroupBody,
  ObservationResultPayload,
} from "../../api/types";
import { useRegistryStore } from "../../stores/registry";
import { useSimulationStore } from "../../stores/simulation";

const registry = useRegistryStore();
const sim = useSimulationStore();

interface GroupDraft {
  genotype: string[] | null;
  sex: "both" | "female" | "male";
  ageStart: number | null;
  ageEnd: number | null;
}

const groups = reactive<GroupDraft[]>([
  { genotype: null, sex: "both", ageStart: null, ageEnd: null },
]);
const collapseAge = ref(false);
const result = ref<ObservationResultPayload | null>(null);
const busy = ref(false);
const error = ref<string | null>(null);

const genotypeOptions = computed(() =>
  (registry.payload?.unordered_genotype_labels ?? []).map((label) => ({
    label,
    value: label,
  })),
);

const sexOptions = [
  { label: "Both", value: "both" },
  { label: "Female", value: "female" },
  { label: "Male", value: "male" },
];

function addGroup(): void {
  groups.push({ genotype: null, sex: "both", ageStart: null, ageEnd: null });
}

function removeGroup(index: number): void {
  groups.splice(index, 1);
}

async function apply(): Promise<void> {
  busy.value = true;
  error.value = null;
  try {
    const body = {
      groups: groups.map((group): ObservationGroupBody => ({
        genotype: group.genotype ?? undefined,
        sex: group.sex === "both" ? undefined : group.sex,
        age_start: group.ageStart ?? undefined,
        age_end: group.ageEnd ?? undefined,
      })),
      collapse_age: collapseAge.value,
    };
    result.value = await postJson<
      typeof body,
      ObservationResultPayload
    >("/api/observation", body);
  } catch (caught) {
    error.value = String(caught);
  } finally {
    busy.value = false;
  }
}

function fmt(value: number): string {
  return Math.round(value).toLocaleString();
}
</script>

<template>
  <div class="observation-layout">
    <NCard
      title="Observation groups"
      size="small"
    >
      <NSpace vertical>
        <NText
          v-if="!registry.payload"
          depth="3"
        >
          Loading registry…
        </NText>
        <NCard
          v-for="(group, index) in groups"
          :key="index"
          size="small"
          embedded
        >
          <NSpace align="center">
            <NTag
              size="small"
              :bordered="false"
            >
              G{{ index }}
            </NTag>
            <NSelect
              v-model:value="group.genotype"
              multiple
              clearable
              filterable
              tag
              size="small"
              placeholder="genotype pattern (e.g. WT::Dr, WT|*)"
              :options="genotypeOptions"
              style="min-width: 260px"
            />
            <NSelect
              v-model:value="group.sex"
              size="small"
              :options="sexOptions"
              style="width: 96px"
            />
            <NInputNumber
              v-model:value="group.ageStart"
              size="small"
              :min="0"
              placeholder="age ≥"
              style="width: 88px"
            />
            <NInputNumber
              v-model:value="group.ageEnd"
              size="small"
              :min="0"
              placeholder="age ≤"
              style="width: 88px"
            />
            <NButton
              size="small"
              quaternary
              type="error"
              @click="removeGroup(index)"
            >
              ✕
            </NButton>
          </NSpace>
        </NCard>

        <NSpace>
          <NButton
            size="small"
            @click="addGroup"
          >
            Add group
          </NButton>
          <NCheckbox v-model:checked="collapseAge">
            Collapse age
          </NCheckbox>
          <NButton
            size="small"
            type="primary"
            :loading="busy"
            :disabled="sim.wsStatus !== 'open'"
            @click="apply"
          >
            Apply
          </NButton>
        </NSpace>
      </NSpace>
    </NCard>

    <NCard
      title="Observed counts"
      size="small"
    >
      <NText
        v-if="error"
        type="error"
      >
        {{ error }}
      </NText>
      <NText
        v-else-if="!result"
        depth="3"
      >
        Define groups and press Apply.
      </NText>
      <NTable
        v-else
        size="small"
        :single-line="false"
      >
        <thead>
          <tr>
            <th>Group</th>
            <th v-if="!result.collapse_age">
              Age
            </th>
            <th>Female</th>
            <th>Male</th>
            <th>Total</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in result.rows"
            :key="`${row.group}|${row.age}`"
          >
            <td>{{ row.group }}</td>
            <td v-if="!result.collapse_age">
              {{ row.age }}
            </td>
            <td>{{ fmt(row.female) }}</td>
            <td>{{ fmt(row.male) }}</td>
            <td>{{ fmt(row.total) }}</td>
          </tr>
        </tbody>
      </NTable>
    </NCard>
  </div>
</template>

<style scoped>
.observation-layout {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
</style>
