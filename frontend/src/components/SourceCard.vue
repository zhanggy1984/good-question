<template>
  <div class="source-card">
    <div class="source-header">
      <span class="doc-icon">📄</span>
      <span class="doc-name">{{ source.document_name }}</span>
      <span v-if="source.chunk_index !== undefined && source.chunk_index !== null" class="chunk-info">
        片段 {{ source.chunk_index + 1 }}/{{ source.total_chunks }}
      </span>
    </div>
    <div v-if="source.heading_path && source.heading_path.length" class="heading-path">
      {{ source.heading_path.join(' › ') }}
    </div>
    <div class="content">{{ source.chunk_content }}</div>
  </div>
</template>

<script setup lang="ts">
import type { Source } from '@/api/session'

defineProps<{ source: Source }>()
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
