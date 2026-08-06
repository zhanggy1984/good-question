<template>
  <div class="session-list">
    <div class="session-header">
      <span>会话</span>
      <n-button text size="small" type="primary" :disabled="!canCreate" @click="create">
        ＋ 新会话
      </n-button>
    </div>
    <div v-if="sessions.length === 0" class="empty">
      <p>暂无会话</p>
      <p class="hint">选择下方文档库后新建会话</p>
    </div>
    <div
      v-for="s in sessions"
      :key="s.id"
      class="session-item"
      :class="{ active: s.id === currentId }"
      @click="select(s.id)"
    >
      <div class="session-title">{{ s.title || '新会话' }}</div>
      <div class="session-meta">{{ s.message_count }} 条消息</div>
      <n-button v-if="canDelete" class="del" text size="tiny" type="error" @click.stop="remove(s.id)">
        删除
      </n-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { NButton } from 'naive-ui'
import type { ChatSession } from '@/api/session'

defineProps<{
  sessions: ChatSession[]
  currentId: number | null
  canCreate: boolean
  canDelete: boolean
}>()

const emit = defineEmits<{
  (e: 'create'): void
  (e: 'select', id: number): void
  (e: 'remove', id: number): void
}>()

const create = () => emit('create')
const select = (id: number) => emit('select', id)
const remove = (id: number) => emit('remove', id)
</script>

<style scoped>
.session-list {
  height: 100%;
  overflow-y: auto;
}
.session-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  font-size: 13px;
  font-weight: 600;
}
.empty {
  text-align: center;
  padding: 32px 16px;
  opacity: 0.5;
  font-size: 13px;
}
.empty .hint {
  font-size: 12px;
}
.session-item {
  position: relative;
  padding: 10px 16px;
  cursor: pointer;
  border-left: 3px solid transparent;
  transition: background 0.2s;
}
.session-item:hover {
  background: rgba(255, 255, 255, 0.04);
}
.session-item.active {
  background: rgba(99, 226, 183, 0.08);
  border-left-color: var(--primary-color, #63e2b7);
}
.session-title {
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 30px;
}
.session-meta {
  font-size: 11px;
  opacity: 0.5;
  margin-top: 2px;
}
.del {
  position: absolute;
  right: 8px;
  top: 50%;
  transform: translateY(-50%);
  opacity: 0;
}
.session-item:hover .del {
  opacity: 1;
}
</style>
