import api from './index'
import type { Page } from './types'

export interface Library {
  id: number
  name: string
  description: string | null
  created_by: number | null
  created_at: string
}

export const listLibraries = (params?: { page?: number; page_size?: number }) =>
  api.get<Page<Library>>('/libraries', { params })

export const createLibrary = (data: { name: string; description?: string }) =>
  api.post<Library>('/libraries', data)

export const deleteLibrary = (id: number) => api.delete(`/libraries/${id}`)
