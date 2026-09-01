import api from './index'
import type { Page } from './types'

export interface ChatSession {
  id: number
  library_id: number
  title: string | null
  summary: string | null
  message_count: number
  created_at: string
  updated_at: string
}

export interface Source {
  document_name: string
  heading_path: string[]
  chunk_content: string
  chunk_index: number
  total_chunks: number
  page_range?: number[]
  page_number?: number // 旧版消息兼容（sources_json 历史数据）
  heading_level?: number
}

export interface ChatMessage {
  id: number
  session_id: number
  role: string
  content: string
  sources_json: Source[] | null
  created_at: string
}

export interface SessionDetail extends ChatSession {
  messages: ChatMessage[]
}

export const listSessions = (params?: { library_id?: number; page?: number; page_size?: number }) =>
  api.get<Page<ChatSession>>('/sessions', { params })

export const createSession = (libraryId: number) =>
  api.post<ChatSession>('/sessions', { library_id: libraryId })

export const getSession = (id: number) => api.get<SessionDetail>(`/sessions/${id}`)

export const deleteSession = (id: number) => api.delete(`/sessions/${id}`)
