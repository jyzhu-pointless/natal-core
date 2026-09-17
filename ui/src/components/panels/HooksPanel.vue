<script setup lang="ts">
/**
 * Hooks tab: compiled hook descriptors grouped by event, declarative
 * operations rendered as readable rows, callbacks with their source code.
 */
import { computed } from "vue";
import { NCode, NCollapse, NCollapseItem, NEmpty, NTag } from "naive-ui";

import { useDomainStore } from "../../stores/domain";
import type { HookInfo } from "../../api/types";

const domain = useDomainStore();

function describeOp(op: NonNullable<HookInfo["operations"]>[number]): string {
  const parts: string[] = [op.type];
  if (op.genotypes === "*") {
    parts.push("ALL genotypes");
  } else if (Array.isArray(op.genotypes)) {
    parts.push(`genotypes ${op.genotypes.join(", ")}`);
  } else {
    parts.push(`genotype ${op.genotypes}`);
  }
  if (op.sex !== "both") {
    parts.push(`(${op.sex} only)`);
  }
  if (op.type === "scale" || op.type === "kill") {
    parts.push(`by factor ${op.param}`);
  } else if (op.type === "add" || op.type === "subtract") {
    parts.push(`by ${op.param}`);
  } else if (op.type === "set_count") {
    parts.push(`to ${op.param}`);
  }
  if (op.ages !== "*") {
    parts.push(`at ages ${Array.isArray(op.ages) ? op.ages.join(",") : op.ages}`);
  }
  if (op.condition) {
    parts.push(`WHEN ${op.condition}`);
  }
  return parts.join(" ");
}

const hookCount = computed(() => domain.hooks.length);
</script>

<template>
  <NEmpty
    v-if="!hookCount"
    description="No hooks registered."
  />
  <NCollapse
    v-else
    display-directive="show"
  >
    <NCollapseItem
      v-for="hook in domain.hooks"
      :key="`${hook.event}/${hook.name}/${hook.priority}`"
      :title="`${hook.name} (${hook.event}) — priority ${hook.priority}`"
    >
      <NSpace vertical>
        <NTag
          size="small"
          :type="hook.kind === 'declarative' ? 'info' : 'default'"
        >
          {{ hook.kind }}
        </NTag>

        <div
          v-for="(op, index) in hook.operations ?? []"
          :key="index"
          class="op-row"
        >
          {{ describeOp(op) }}
        </div>

        <div v-if="hook.signature">
          <b>Signature:</b>
          <NCode
            :code="hook.signature"
            language="python"
            inline
          />
        </div>

        <NCode
          v-if="hook.source"
          :code="hook.source"
          language="python"
        />
      </NSpace>
    </NCollapseItem>
  </NCollapse>
</template>

<style scoped>
.op-row {
  font-family: monospace;
  font-size: 12px;
  border-bottom: 1px dashed #eee;
  padding: 3px 6px;
  background: #fafafc;
  border-radius: 4px;
}
</style>
