<template>
  <div>
    <n-h2>数据总览</n-h2>
    <n-grid :cols="3" :x-gap="20" responsive="screen" item-responsive>
      <n-grid-item span="3 m:1">
        <n-card class="stat-card" :bordered="true">
          <div class="stat-icon icon-blue">
            <n-icon :component="LayersOutline" size="28" />
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ stats?.library_count ?? '-' }}</div>
            <div class="stat-label">文档库</div>
          </div>
        </n-card>
      </n-grid-item>
      <n-grid-item span="3 m:1">
        <n-card class="stat-card">
          <div class="stat-icon icon-green">
            <n-icon :component="DocumentOutline" size="28" />
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ stats?.document_count ?? '-' }}</div>
            <div class="stat-label">文档</div>
          </div>
        </n-card>
      </n-grid-item>
      <n-grid-item span="3 m:1">
        <n-card class="stat-card">
          <div class="stat-icon icon-purple">
            <n-icon :component="CubeOutline" size="28" />
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ stats?.chunk_count ?? '-' }}</div>
            <div class="stat-label">片段 Chunk</div>
          </div>
        </n-card>
      </n-grid-item>
    </n-grid>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { NCard, NGrid, NGridItem, NH2, NIcon } from 'naive-ui'
import { LayersOutline, DocumentOutline, CubeOutline } from '@vicons/ionicons5'
import api from '@/api'

const stats = ref<{ library_count: number; document_count: number; chunk_count: number } | null>(null)

onMounted(async () => {
  const res = await api.get('/dashboard')
  stats.value = res.data
})
</script>

<style scoped>
.stat-card {
  border-radius: 10px;
}
.stat-card :deep(.n-card__content) {
  display: flex;
  align-items: center;
  gap: 16px;
}
.stat-icon {
  width: 52px;
  height: 52px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.icon-blue {
  background: rgba(24, 160, 251, 0.12);
  color: #18a0fb;
}
.icon-green {
  background: rgba(24, 160, 88, 0.12);
  color: #18a058;
}
.icon-purple {
  background: rgba(144, 98, 246, 0.12);
  color: #9062f6;
}
.stat-value {
  font-size: 26px;
  font-weight: 700;
  line-height: 1.2;
}
.stat-label {
  font-size: 13px;
  opacity: 0.6;
}
</style>
