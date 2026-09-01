<template>
  <div class="source-card" :class="{ expanded }">
    <div class="source-header">
      <span v-if="index !== undefined" class="source-index">[来源{{ index + 1 }}]</span>
      <span v-if="expanded" class="expanded-tag">⤵ 补充上下文</span>
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

const props = defineProps<{ source: Source; index?: number; expanded?: boolean }>()

// 编号角标：与回答中 [来源N] 高亮同源（后端 _format_docs 编号，sources 即全部 context chunk），
// 用户可把回答引用与来源卡片一一对应；不传 index 时（无编号场景）不显示角标。
// expanded：相邻节扩充的补充上下文，降权展示（弱化卡片 + "补充上下文"标签），
// 精排命中不设（undefined 即未扩充，不影响旧版数据渲染）

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
.expanded-tag {
  color: rgba(255, 255, 255, 0.45);
  font-size: 11px;
  flex-shrink: 0;
}
.source-card.expanded {
  opacity: 0.8;
  border-left-color: rgba(255, 255, 255, 0.28);
  background: rgba(255, 255, 255, 0.015);
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
