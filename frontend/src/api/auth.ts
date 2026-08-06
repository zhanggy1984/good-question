import api from './index'

export interface User {
  id: number
  username: string
  role: string
  created_at: string
}

export const register = (username: string, password: string) =>
  api.post('/auth/register', { username, password })

export const login = (username: string, password: string) =>
  api.post<{ access_token: string; token_type: string }>('/auth/login', { username, password })

export const getMe = () => api.get<User>('/auth/me')
