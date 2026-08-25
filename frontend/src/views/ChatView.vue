<template>
  <div class="chat-page">
    <!-- 左侧：库选择 + 会话列表 -->
    <div class="chat-sidebar">
      <div class="lib-select">
        <n-select
          v-model:value="selectedLibrary"
          :options="libraryOptions"
          placeholder="选择文档库"
          size="small"
          @update:value="onLibraryChange"
        />
      </div>
      <SessionList
        :sessions="sessions"
        :current-id="currentSessionId"
        :can-create="!!selectedLibrary"
        :can-delete="true"
        @create="createSession"
        @select="selectSession"
        @remove="removeSession"
      />
    </div>

    <!-- 右侧：聊天区 -->
    <div class="chat-main">
      <div v-if="!selectedLibrary" class="chat-placeholder">
        <p>请先选择左侧文档库</p>
        <p class="hint">选择后即可基于该库的文档进行问答</p>
      </div>
      <template v-else>
        <div ref="msgRef" class="messages" :native-scrollbar="false">
          <div v-if="messages.length === 0" class="chat-placeholder">
            <p>开始提问吧</p>
            <p class="hint">回答将流式返回，并附带引用来源</p>
          </div>
          <ChatMessage v-for="(m, i) in messages" :key="i" :message="m" />
          <div v-if="chat.error" class="error-bar">
            <span>{{ chat.error }}</span>
            <n-button size="tiny" type="warning" @click="retryLast">重试</n-button>
          </div>
        </div>
        <ChatInput :disabled="chat.streaming" @send="send" />
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { NSelect, NButton } from 'naive-ui'
import SessionList from '@/components/SessionList.vue'
import ChatMessage from '@/components/ChatMessage.vue'
import ChatInput from '@/components/ChatInput.vue'
import { useAuthStore } from '@/stores/auth'
import { useChatStore } from '@/stores/chat'
import { message, dialog } from '@/utils/naive'
import { listLibraries, type Library } from '@/api/library'
import {
  listSessions,
  createSession as apiCreateSession,
  getSession,
  deleteSession,
  type ChatSession,
} from '@/api/session'
import { streamChat } from '@/utils/sse'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const chat = useChatStore()

const msgRef = ref<HTMLElement>()
const libraries = ref<Library[]>([])
const selectedLibrary = ref<number | null>(null)
const sessions = ref<ChatSession[]>([])

const libraryOptions = computed(() =>
  libraries.value.map((l) => ({ label: l.name, value: l.id })),
)
const currentSessionId = computed(() => chat.currentSessionId)
const messages = computed(() => chat.messages)

onMounted(async () => {
  // 加载库列表
  const res = await listLibraries({ page_size: 100 })
  libraries.value = res.data.items
  // 若路由指定了会话，加载对应库和会话
  const sid = Number(route.params.sessionId)
  if (sid) {
    await openSession(sid)
  }
})

watch(() => route.params.sessionId, async (v) => {
  const sid = Number(v)
  if (sid && sid !== chat.currentSessionId) {
    await openSession(sid)
  }
})

async function onLibraryChange() {
  chat.reset()
  await loadSessions()
  if (route.path !== '/chat') router.replace('/chat')
}

async function loadSessions() {
  if (!selectedLibrary.value) return
  const res = await listSessions({ library_id: selectedLibrary.value, page_size: 50 })
  sessions.value = res.data.items
}

async function createSession() {
  if (!selectedLibrary.value) {
    message.warning('请先选择文档库')
    return
  }
  const res = await apiCreateSession(selectedLibrary.value)
  router.push(`/chat/${res.data.id}`)
  await loadSessions()
}

async function selectSession(id: number) {
  if (id !== chat.currentSessionId) {
    router.push(`/chat/${id}`)
  }
}

async function openSession(id: number) {
  const detail = await getSession(id)
  // 设置库并加载
  selectedLibrary.value = detail.data.library_id
  chat.setCurrent(detail.data.id)
  chat.setMessages(
    detail.data.messages.map((m) => ({
      id: m.id,
      role: m.role as 'user' | 'assistant',
      content: m.content,
      sources: m.sources_json || undefined,
    })),
  )
  await loadSessions()
  scrollBottom()
}

async function removeSession(id: number) {
  const target = sessions.value.find((s) => s.id === id)
  dialog.warning({
    title: '删除会话',
    content: `确定删除会话「${target?.title || '新会话'}」？该会话的所有聊天记录将被删除。`,
    positiveText: '删除',
    negativeText: '取消',
    onPositiveClick: async () => {
      try {
        await deleteSession(id)
        message.success('会话已删除')
        if (chat.currentSessionId === id) {
          chat.reset()
          router.replace('/chat')
        }
        await loadSessions()
      } catch (e: any) {
        message.error(e.response?.data?.error?.message || '删除失败')
      }
    },
  })
}

async function send(content: string) {
  // 确保有会话
  let sid = chat.currentSessionId
  if (!sid) {
    if (!selectedLibrary.value) {
      message.warning('请先选择文档库')
      return
    }
    const res = await apiCreateSession(selectedLibrary.value)
    sid = res.data.id
    router.replace(`/chat/${sid}`)
    await loadSessions()
  }

  chat.setError(null)
  chat.addUserMessage(content)
  chat.startAssistantMessage()
  chat.setStreaming(true)
  scrollBottom()

  try {
    await streamChat(sid as number, content, auth.token, (ev) => {
      if (ev.type === 'tool_call') {
        // 检索状态三态：failed=服务不可用（显示"可信度偏低"警告）、empty=已检索但空命中
        // （显示"本次未检索到相关内容"）、hit=命中（走 sources 展示）。
        // 无 tool_call 事件 = LLM 未检索直接答（问候/闲聊等），不标记。
        // empty 提示仅对意图与文档相关（query/unknown，且非计算/常识豁免）显示——
        // smalltalk 问候、non_doc 计算题的空命中后端交 LLM 自然作答，提示"未找到"会突兀。
        const st = ev.data.status
        if (st === 'error' || st === 'rule_override_error') {
          chat.setRetrievalFailed()
        } else if (
          ev.data.result?.source_count === 0 &&
          ev.data.intent !== 'smalltalk' &&
          !ev.data.non_doc_question
        ) {
          chat.setRetrievalEmpty()
        }
      } else if (ev.type === 'sources') {
        chat.setAssistantSources(ev.data.sources)
      } else if (ev.type === 'reasoning') {
        chat.appendReasoning(ev.data.content)
      } else if (ev.type === 'token') {
        chat.appendToken(ev.data.content)
        scrollBottom()
      } else if (ev.type === 'done') {
        chat.finishAssistantMessage()
        chat.setStreaming(false)
        scrollBottom()
      } else if (ev.type === 'error') {
        chat.setError(ev.data.message)
        chat.finishAssistantMessage()
        chat.setStreaming(false)
      }
    })
  } catch (e: any) {
    chat.setError(e.message || '连接中断')
    chat.finishAssistantMessage()
    chat.setStreaming(false)
  }
}

// 重试：重新发送最后一条用户消息
async function retryLast() {
  const lastUser = [...messages.value].reverse().find((m) => m.role === 'user')
  if (lastUser) {
    // 移除未完成的助手消息
    chat.messages.pop()
    chat.setError(null)
    await send(lastUser.content)
  }
}

function scrollBottom() {
  nextTick(() => {
    if (msgRef.value) msgRef.value.scrollTop = msgRef.value.scrollHeight
  })
}
</script>

<style scoped>
.chat-page {
  display: flex;
  height: calc(100vh - 96px);
  gap: 16px;
}
.chat-sidebar {
  width: 260px;
  flex-shrink: 0;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.lib-select {
  padding: 12px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.chat-sidebar :deep(.session-list) {
  flex: 1;
  overflow-y: auto;
}
.chat-main {
  flex: 1;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}
.chat-placeholder {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--text-color);
  opacity: 0.6;
}
.chat-placeholder p {
  margin: 4px 0;
  font-size: 15px;
}
.chat-placeholder .hint {
  font-size: 12px;
}
.error-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 8px;
  background: rgba(208, 48, 80, 0.1);
  border-radius: 8px;
  margin-top: 8px;
  color: #d03050;
  font-size: 13px;
}
</style>
