import api from './index'
import type { Page } from './types'

export interface Document {
  id: number
  library_id: number
  filename: string
  file_type: string
  file_size: number
  chunk_count: number
  processed_chunks: number
  chunk_size: number
  overlap_token: number
  status: string
  error_message: string | null
  created_at: string
}

export interface DocumentStatus {
  status: string
  chunk_count: number
  processed_chunks: number
  error_message: string | null
}

export const listDocuments = (libraryId: number, params?: { page?: number; page_size?: number }) =>
  api.get<Page<Document>>(`/libraries/${libraryId}/documents`, { params })

export const uploadDocument = (libraryId: number, file: File, chunkSize = 1024, overlapToken = 102) => {
  const form = new FormData()
  form.append('file', file)
  form.append('chunk_size', String(chunkSize))
  form.append('overlap_token', String(overlapToken))
  return api.post<Document>(`/libraries/${libraryId}/documents`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export const getDocumentStatus = (id: number) => api.get<DocumentStatus>(`/documents/${id}/status`)

export const deleteDocument = (id: number) => api.delete(`/documents/${id}`)

export interface ChunkItem {
  id: number
  chunk_index: number
  content: string
  token_count: number
  metadata_json: Record<string, any>
}

export const listDocumentChunks = (id: number, params?: { page?: number; page_size?: number }) =>
  api.get<Page<ChunkItem>>(`/documents/${id}/chunks`, { params })
