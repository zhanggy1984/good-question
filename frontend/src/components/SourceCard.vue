<template>
  <div class="source-card">
    <div class="source-header">
      <span v-if="index !== undefined" class="source-index">[来源{{ index + 1 }}]</span>
      <span class="doc-icon">📄</span>
      <span class="doc-name">{{ source.document_name }}</span>
      <span v-if="source.chunk_index !== undefined && source.chunk_index !== null" class="chunk-info">
        片段 {{ source.chunk_index + 1 }}/{{ source.total_chunks }}{{ pageLabel }}
      </span>
    </div>
    <div v-if="source.heading_path && source.heading_path.length" class="heading-path">
      {{ source.heading_path.join(' › ') }}
    </div>
    <div class="content">{{ source.chunk_content }}</div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { Source } from '@/api/session'

const props = defineProps<{ source: Source; index?: number }>()

// 编号角标：与回答中 [来源N] 高亮同源（后端 _format_docs 编号）；sources 只含精排
// top-3，故编号 1-3 与卡片一一对应。不传 index 时（无编号场景）不显示角标。
// 注：context 含章节扩充（编号可到 6），回答引用 [来源4-6] 时无对应卡片，属预期取舍。

// 页码：新版 page_range（单页 N / 跨页 A-B）；旧版消息回退到单值 page_number
const pageLabel = computed(() => {
  const r = props.source.page_range ?? (props.source.page_number ? [props.source.page_number, props.source.page_number] : null)
  if (!r || r[1] <= 0) return ''
  return r[0] === r[1] ? ` · 第 ${r[0]} 页` : ` · 第 ${r[0]}-${r[1]} 页`
})
</script>

<style scoped>
.source-card {
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-left: 3px solid var(--primary-color, #63e2b7);
  border-radius: 8px;
  padding: 10px 12px;
  margin-top: 8px;
  background: rgba(255, 255, 255, 0.03);
  font-size: 12px;
}
.source-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 4px;
}
.doc-name {
  font-weight: 600;
  font-size: 12px;
}
.source-index {
  color: var(--primary-color, #63e2b7);
  font-weight: 600;
  font-size: 12px;
  flex-shrink: 0;
}
.chunk-info {
  opacity: 0.6;
  font-size: 11px;
  margin-left: auto;
}
.heading-path {
  opacity: 0.6;
  font-size: 11px;
  margin-bottom: 4px;
}
.content {
  opacity: 0.75;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>
