<template>
  <div>
    <div class="page-header">
      <div>
        <n-breadcrumb>
          <n-breadcrumb-item><a @click="router.push('/libraries')">文档库</a></n-breadcrumb-item>
          <n-breadcrumb-item>{{ libraryName }}</n-breadcrumb-item>
        </n-breadcrumb>
      </div>
      <div class="upload-config">
        <span class="cfg-label">chunk 长度</span>
        <n-input-number v-model:value="chunkSize" :min="128" :max="8192" size="small" style="width: 110px" />
        <span class="cfg-label">重叠 token</span>
        <n-input-number v-model:value="overlapToken" :min="0" :max="Math.max(chunkSize - 1, 0)" size="small" style="width: 110px" />
        <n-button v-if="isAdmin" type="primary" :loading="uploading" @click="fileInput?.click()">
          上传文档
        </n-button>
      </div>
      <input
        ref="fileInput"
        type="file"
        accept=".pdf,.docx,.txt,.md"
        hidden
        @change="handleUpload"
      />
    </div>

    <n-data-table
      :columns="columns"
      :data="documents"
      :loading="loading"
      :bordered="true"
      :pagination="pagination"
      @update:page="onPageChange"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, h, onMounted, onUnmounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  NButton,
  NTag,
  NPopconfirm,
  NDataTable,
  NBreadcrumb,
  NBreadcrumbItem,
  NInputNumber,
} from 'naive-ui'
import type { DataTableColumns } from 'naive-ui'
import { listDocuments, uploadDocument, deleteDocument, getDocumentStatus, type Document } from '@/api/document'
import { listLibraries } from '@/api/library'
import { useAuthStore } from '@/stores/auth'
import { message } from '@/utils/naive'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const isAdmin = computed(() => auth.isAdmin)

const libraryId = Number(route.params.id)
const libraryName = ref('')
const documents = ref<Document[]>([])
const loading = ref(false)
const uploading = ref(false)
const fileInput = ref<HTMLInputElement>()
// 上传切分配置（默认 1024 chunk / 102 重叠 token，上传前可调整）
const chunkSize = ref(1024)
const overlapToken = ref(102)
const pagination = ref({ page: 1, pageSize: 10, itemCount: 0 })
let pollTimer: number | null = null

const columns = computed<DataTableColumns<Document>>(() => {
  const cols: DataTableColumns<Document> = [
    {
      title: '文件名',
      key: 'filename',
      render: (row) =>
        h(
          NButton,
          {
            text: true,
            type: 'primary',
            onClick: () =>
              router.push(`/libraries/${libraryId}/documents/${row.id}?name=${encodeURIComponent(row.filename)}`),
          },
          { default: () => row.filename },
        ),
    },
    { title: '类型', key: 'file_type', width: 80 },
    {
      title: '大小',
      key: 'file_size',
      width: 100,
      render: (row) => formatSize(row.file_size),
    },
    { title: 'Chunk 数', key: 'chunk_count', width: 90 },
    {
      title: '状态',
      key: 'status',
      width: 110,
      render: (row) => renderStatus(row),
    },
    { title: '上传时间', key: 'created_at', width: 180 },
  ]
  // 仅 admin 显示操作列（普通用户完全隐藏删除入口）
  if (isAdmin.value) {
    cols.push({
      title: '操作',
      key: 'actions',
      width: 80,
      render: (row) =>
        h(
          NPopconfirm,
          { onPositiveClick: () => handleDelete(row) },
          {
            trigger: () => h(NButton, { text: true, type: 'error' }, { default: () => '删除' }),
            default: () => `删除文档「${row.filename}」？`,
          },
        ),
    })
  }
  return cols
})

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes}B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`
  return `${(bytes / 1024 / 1024).toFixed(1)}MB`
}

function renderStatus(row: Document) {
  // 处理中带实时进度（大文档向量化期间 processed_chunks 持续增长）
  if (row.status === 'processing') {
    return h(NTag, { type: 'info', size: 'small' }, {
      default: () => (row.processed_chunks > 0 ? `处理中 ${row.processed_chunks} 段` : '处理中'),
    })
  }
  const map: Record<string, { type: 'default' | 'info' | 'success' | 'error'; text: string }> = {
    ready: { type: 'success', text: '就绪' },
    failed: { type: 'error', text: '失败' },
  }
  const s = map[row.status] || { type: 'default', text: row.status }
  return h(NTag, { type: s.type, size: 'small' }, { default: () => s.text })
}

async function fetchLibrary() {
  const res = await listLibraries({ page_size: 100 })
  const lib = res.data.items.find((l) => l.id === libraryId)
  libraryName.value = lib?.name || '文档库'
}

async function fetchData() {
  loading.value = true
  try {
    const res = await listDocuments(libraryId, { page: pagination.value.page, page_size: pagination.value.pageSize })
    documents.value = res.data.items
    pagination.value.itemCount = res.data.total
  } finally {
    loading.value = false
  }
}

function onPageChange(page: number) {
  pagination.value.page = page
  fetchData()
}

async function handleUpload(e: Event) {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  uploading.value = true
  try {
    const res = await uploadDocument(libraryId, file, chunkSize.value, overlapToken.value)
    message.success('上传成功，开始处理')
    // 轮询新文档状态
    startPolling(res.data.id)
    await fetchData()
  } catch (err: any) {
    message.error(err.response?.data?.error?.message || '上传失败')
  } finally {
    uploading.value = false
    input.value = ''
  }
}

async function handleDelete(row: Document) {
  try {
    await deleteDocument(row.id)
    message.success('已删除')
    fetchData()
  } catch (e: any) {
    message.error(e.response?.data?.error?.message || '删除失败')
  }
}

// 轮询处理中的文档
function startPolling(initialId?: number) {
  if (pollTimer) return
  pollTimer = window.setInterval(async () => {
    let changed = false
    for (const doc of documents.value) {
      if (doc.status === 'processing') {
        try {
          const st = await getDocumentStatus(doc.id)
          if (st.data.status !== 'processing') {
            doc.status = st.data.status
            doc.chunk_count = st.data.chunk_count
            changed = true
          } else if (st.data.processed_chunks !== doc.processed_chunks) {
            // 处理中：仅同步进度，避免整表无谓刷新
            doc.processed_chunks = st.data.processed_chunks
            changed = true
          }
        } catch {}
      }
    }
    if (changed) await fetchData()
  }, 3000)
}

onMounted(async () => {
  await fetchLibrary()
  await fetchData()
  startPolling()
})

onUnmounted(() => {
  if (pollTimer) window.clearInterval(pollTimer)
})
</script>

<style scoped>
.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}
.upload-config {
  display: flex;
  align-items: center;
  gap: 8px;
}
.cfg-label {
  font-size: 13px;
  color: #666;
  white-space: nowrap;
}
</style>
