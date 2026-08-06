import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', name: 'login', component: () => import('@/views/LoginView.vue') },
    { path: '/register', name: 'register', component: () => import('@/views/RegisterView.vue') },
    {
      path: '/',
      component: () => import('@/components/AppLayout.vue'),
      children: [
        { path: '', name: 'dashboard', component: () => import('@/views/DashboardView.vue') },
        { path: 'libraries', name: 'libraries', component: () => import('@/views/LibraryListView.vue') },
        { path: 'libraries/:id', name: 'library-docs', component: () => import('@/views/DocumentListView.vue') },
        { path: 'libraries/:id/documents/:docId', name: 'doc-chunks', component: () => import('@/views/DocumentDetailView.vue') },
        { path: 'chat', name: 'chat', component: () => import('@/views/ChatView.vue') },
        { path: 'chat/:sessionId', name: 'chat-session', component: () => import('@/views/ChatView.vue') },
      ],
    },
    { path: '/:pathMatch(.*)*', redirect: '/' },
  ],
})

// 路由守卫：未登录跳登录页；非 admin 用户只能访问聊天问答页
router.beforeEach(async (to) => {
  const auth = useAuthStore()
  const publicPages = ['/login', '/register']
  if (!auth.isLoggedIn && !publicPages.includes(to.path)) {
    return { path: '/login' }
  }
  if (auth.isLoggedIn && publicPages.includes(to.path)) {
    return { path: '/' }
  }
  // 非 admin：仅 /chat 相关页面放行，其余（仪表盘/文档库/文档）一律重定向到聊天
  if (auth.isLoggedIn && !to.path.startsWith('/chat')) {
    if (!auth.user) {
      try {
        await auth.fetchMe()
      } catch {
        auth.logout()
        return { path: '/login' }
      }
    }
    if (!auth.isAdmin) {
      return { path: '/chat' }
    }
  }
})

export default router
