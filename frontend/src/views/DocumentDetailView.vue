<template>
  <div>
    <div class="page-header">
      <n-breadcrumb>
        <n-breadcrumb-item><a @click="router.push('/libraries')">文档库</a></n-breadcrumb-item>
        <n-breadcrumb-item>
          <a @click="router.push(`/libraries/${libraryId}`)">{{ libraryName }}</a>
        </n-breadcrumb-item>
        <n-breadcrumb-item>{{ docFilename }}</n-breadcrumb-item>
      </n-breadcrumb>
    </div>

    <div class="chunk-list">
      <n-card v-for="c in chunks" :key="c.id" class="chunk-card" :title="`片段 ${c.chunk_index + 1}`">
        <template #header-extra>
          <span class="chunk-meta">{{ c.token_count }} tokens</span>
        </template>
        <div class="chunk-content">{{ c.content }}</div>
        <div v-if="headingPath(c)" class="chunk-heading">
          📍 {{ headingPath(c) }}
        </div>
      </n-card>
    </div>

    <div class="pagination">
      <n-pagination
        :page="pagination.page"
        :page-size="pagination.pageSize"
        :item-count="pagination.itemCount"
        @update:page="onPageChange"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { NCard, NBreadcrumb, NBreadcrumbItem, NPagination } from 'naive-ui'
import { listDocumentChunks, type ChunkItem } from '@/api/document'
import { listLibraries } from '@/api/library'

const route = useRoute()
const router = useRouter()
const docId = Number(route.params.docId)
const libraryId = Number(route.params.id)

const chunks = ref<ChunkItem[]>([])
const libraryName = ref('')
const docFilename = ref('')
const pagination = ref({ page: 1, pageSize: 10, itemCount: 0 })

function headingPath(c: ChunkItem): string {
  const h = c.metadata_json?.heading_path
  if (Array.isArray(h) && h.length) return h.filter(Boolean).join(' › ')
  return ''
}

async function fetchData() {
  const res = await listDocumentChunks(docId, { page: pagination.value.page, page_size: pagination.value.pageSize })
  chunks.value = res.data.items
  pagination.value.itemCount = res.data.total
}

function onPageChange(page: number) {
  pagination.value.page = page
  fetchData()
}

onMounted(async () => {
  // 库名从列表查，文档名从路由 query 传入
  const libs = await listLibraries({ page_size: 100 })
  const lib = libs.data.items.find((l) => l.id === libraryId)
  libraryName.value = lib?.name || '文档库'
  docFilename.value = (route.query.name as string) || `文档 #${docId}`
  await fetchData()
})
</script>

<style scoped>
.page-header {
  margin-bottom: 16px;
}
.chunk-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.chunk-card {
  border-radius: 10px;
}
.chunk-meta {
  font-size: 12px;
  opacity: 0.5;
}
.chunk-content {
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.7;
  font-size: 14px;
}
.chunk-heading {
  margin-top: 8px;
  font-size: 12px;
  opacity: 0.6;
  color: var(--primary-color, #63e2b7);
}
.pagination {
  margin-top: 20px;
  display: flex;
  justify-content: center;
}
</style>
