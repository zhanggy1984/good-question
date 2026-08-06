import { defineStore } from 'pinia'
import type { ChatSession, Source } from '@/api/session'

export interface Message {
  id?: number
  role: 'user' | 'assistant'
  content: string
  reasoning?: string
  sources?: Source[]
  streaming?: boolean
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
