<template>
  <div>
    <div class="page-header">
      <n-h2>文档库管理</n-h2>
      <n-button v-if="isAdmin" type="primary" @click="showCreate = true">新增文档库</n-button>
    </div>

    <n-data-table
      :columns="columns"
      :data="libraries"
      :loading="loading"
      :bordered="true"
      :pagination="pagination"
      @update:page="onPageChange"
    />

    <n-modal v-model:show="showCreate" preset="card" title="新增文档库" style="width: 440px">
      <n-form :model="createForm">
        <n-form-item label="名称" required>
          <n-input v-model:value="createForm.name" placeholder="文档库名称" />
        </n-form-item>
        <n-form-item label="描述">
          <n-input v-model:value="createForm.description" type="textarea" placeholder="可选" />
        </n-form-item>
      </n-form>
      <template #footer>
        <n-space justify="end">
          <n-button @click="showCreate = false">取消</n-button>
          <n-button type="primary" :loading="creating" @click="handleCreate">创建</n-button>
        </n-space>
      </template>
    </n-modal>
  </div>
</template>

<script setup lang="ts">
import { computed, h, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import {
  NButton,
  NSpace,
  NTag,
  NDataTable,
  NModal,
  NForm,
  NFormItem,
  NInput,
} from 'naive-ui'
import type { DataTableColumns } from 'naive-ui'
import { listLibraries, createLibrary, deleteLibrary, type Library } from '@/api/library'
import { useAuthStore } from '@/stores/auth'
import { message, dialog } from '@/utils/naive'

const router = useRouter()
const auth = useAuthStore()
const isAdmin = computed(() => auth.isAdmin)

const libraries = ref<Library[]>([])
const loading = ref(false)
const showCreate = ref(false)
const creating = ref(false)
const createForm = ref({ name: '', description: '' })

const pagination = ref({ page: 1, pageSize: 10, itemCount: 0, showSizePicker: false })

const columns = computed<DataTableColumns<Library>>(() => {
  const cols: DataTableColumns<Library> = [
    { title: 'ID', key: 'id', width: 60 },
    {
      title: '名称',
      key: 'name',
      render: (row) =>
        h(
          NButton,
          { text: true, type: 'primary', onClick: () => router.push(`/libraries/${row.id}`) },
          { default: () => row.name },
        ),
    },
    { title: '描述', key: 'description' },
    { title: '创建时间', key: 'created_at', width: 180 },
  ]
  // 仅 admin 显示操作列（普通用户完全隐藏删除入口）
  if (isAdmin.value) {
    cols.push({
      title: '操作',
      key: 'actions',
      width: 120,
      render: (row) =>
        h(
          NButton,
          { text: true, type: 'error', onClick: () => handleDelete(row) },
          { default: () => '删除' },
        ),
    })
  }
  return cols
})

async function fetchData() {
  loading.value = true
  try {
    const res = await listLibraries({ page: pagination.value.page, page_size: pagination.value.pageSize })
    libraries.value = res.data.items
    pagination.value.itemCount = res.data.total
  } finally {
    loading.value = false
  }
}

function onPageChange(page: number) {
  pagination.value.page = page
  fetchData()
}

async function handleCreate() {
  if (!createForm.value.name.trim()) {
    message.warning('请输入名称')
    return
  }
  creating.value = true
  try {
    await createLibrary(createForm.value)
    message.success('创建成功')
    showCreate.value = false
    createForm.value = { name: '', description: '' }
    fetchData()
  } catch (e: any) {
    message.error(e.response?.data?.error?.message || '创建失败')
  } finally {
    creating.value = false
  }
}

async function handleDelete(row: Library) {
  dialog.warning({
    title: '确认删除',
    content: `确定删除文档库「${row.name}」？将级联删除其全部文档与索引。`,
    positiveText: '删除',
    negativeText: '取消',
    onPositiveClick: async () => {
      try {
        await deleteLibrary(row.id)
        message.success('已删除')
        fetchData()
      } catch (e: any) {
        message.error(e.response?.data?.error?.message || '删除失败')
      }
    },
  })
}

onMounted(fetchData)
</script>

<style scoped>
.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
</style>
