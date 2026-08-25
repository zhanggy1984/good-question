import { defineStore } from 'pinia'
import type { ChatSession, Source } from '@/api/session'

export interface Message {
  id?: number
  role: 'user' | 'assistant'
  content: string
  reasoning?: string
  sources?: Source[]
  streaming?: boolean
  /** 检索服务不可用（tool_call status=error/rule_override_error）时置位，UI 显示"可信度偏低"提示 */
  retrievalFailed?: boolean
  /** 检索正常但空命中（tool_call source_count=0）时置位，UI 显示"本次未检索到相关内容"提示 */
  retrievalEmpty?: boolean
}

export const useChatStore = defineStore('chat', {
  state: () => ({
    sessions: [] as ChatSession[],
    currentSessionId: null as number | null,
    messages: [] as Message[],
    streaming: false,
    error: null as string | null,
  }),
  actions: {
    setSessions(list: ChatSession[]) {
      this.sessions = list
    },
    setCurrent(id: number | null) {
      this.currentSessionId = id
    },
    setMessages(list: Message[]) {
      this.messages = list
    },
    addUserMessage(content: string) {
      this.messages.push({ role: 'user', content })
    },
    startAssistantMessage() {
      this.messages.push({ role: 'assistant', content: '', reasoning: '', streaming: true })
    },
    appendToken(token: string) {
      const last = this.messages[this.messages.length - 1]
      if (last && last.role === 'assistant' && last.streaming) {
        last.content += token
      }
    },
    appendReasoning(reasoning: string) {
      const last = this.messages[this.messages.length - 1]
      if (last && last.role === 'assistant' && last.streaming) {
        last.reasoning += reasoning
      }
    },
    setAssistantSources(sources: Source[]) {
      const last = this.messages[this.messages.length - 1]
      if (last && last.role === 'assistant') {
        last.sources = sources
      }
    },
    setRetrievalFailed() {
      const last = this.messages[this.messages.length - 1]
      if (last && last.role === 'assistant') {
        last.retrievalFailed = true
      }
    },
    setRetrievalEmpty() {
      const last = this.messages[this.messages.length - 1]
      if (last && last.role === 'assistant') {
        last.retrievalEmpty = true
      }
    },
    finishAssistantMessage() {
      const last = this.messages[this.messages.length - 1]
      if (last) last.streaming = false
      this.streaming = false
    },
    setStreaming(v: boolean) {
      this.streaming = v
    },
    setError(msg: string | null) {
      this.error = msg
    },
    reset() {
      this.sessions = []
      this.currentSessionId = null
      this.messages = []
      this.streaming = false
      this.error = null
    },
  },
})
