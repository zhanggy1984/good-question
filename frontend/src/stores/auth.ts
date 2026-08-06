import { defineStore } from 'pinia'
import { getMe, login as apiLogin, register as apiRegister } from '@/api/auth'

export const useAuthStore = defineStore('auth', {
  state: () => ({
    token: localStorage.getItem('access_token') || '',
    user: null as { id: number; username: string; role: string } | null,
  }),
  getters: {
    isLoggedIn: (s) => !!s.token,
    isAdmin: (s) => s.user?.role === 'admin',
  },
  actions: {
    async login(username: string, password: string) {
      const res = await apiLogin(username, password)
      this.token = res.data.access_token
      localStorage.setItem('access_token', this.token)
      await this.fetchMe()
    },
    async fetchMe() {
      const res = await getMe()
      this.user = res.data
    },
    async register(username: string, password: string) {
      await apiRegister(username, password)
    },
    logout() {
      this.token = ''
      this.user = null
      localStorage.removeItem('access_token')
    },
  },
})
