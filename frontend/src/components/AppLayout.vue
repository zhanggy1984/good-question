<template>
  <n-layout has-sider style="height: 100vh">
    <n-layout-sider
      bordered
      :width="220"
      :collapsed-width="64"
      collapse-mode="width"
      :collapsed="collapsed"
      show-trigger
      @collapse="collapsed = true"
      @expand="collapsed = false"
    >
      <div class="logo" @click="router.push('/')">
        <span class="logo-dot"></span>
        <span v-if="!collapsed">不懂就问</span>
      </div>
      <n-menu
        :value="activeKey"
        :options="menuOptions"
        :collapsed="collapsed"
        :collapsed-width="64"
        @update:value="onSelect"
      />
    </n-layout-sider>

    <n-layout>
      <n-layout-header bordered class="header">
        <div class="header-title">{{ pageTitle }}</div>
        <div class="header-user">
          <n-tag v-if="isAdmin" type="warning" size="small" round>管理员</n-tag>
          <span class="username">{{ auth.user?.username }}</span>
          <n-button quaternary size="small" @click="handleLogout">退出登录</n-button>
        </div>
      </n-layout-header>
      <n-layout-content class="content" :native-scrollbar="false">
        <router-view />
      </n-layout-content>
    </n-layout>
  </n-layout>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { NMenu, NLayout, NLayoutSider, NLayoutHeader, NLayoutContent, NTag, NButton } from 'naive-ui'
import { useAuthStore } from '@/stores/auth'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()

const collapsed = ref(false)

// 刷新后恢复用户信息（token 在 localStorage，user 需重新加载）
onMounted(async () => {
  if (auth.isLoggedIn && !auth.user) {
    try {
      await auth.fetchMe()
    } catch {
      auth.logout()
      router.push('/login')
    }
  }
})

const isAdmin = computed(() => auth.isAdmin)

const activeKey = computed(() => {
  const p = route.path
  if (p.startsWith('/libraries')) return 'libraries'
  if (p.startsWith('/chat')) return 'chat'
  return 'dashboard'
})

const pageTitle = computed(() => {
  const map: Record<string, string> = {
    dashboard: '仪表盘',
    libraries: '文档库',
    'library-docs': '文档管理',
    chat: '聊天问答',
    'chat-session': '聊天问答',
  }
  return map[route.name as string] || '不懂就问'
})

// 菜单权限：非 admin 用户只显示"聊天问答"，管理员才显示仪表盘/文档库
const menuOptions = computed(() => {
  if (!isAdmin.value) return [{ label: '聊天问答', key: 'chat' }]
  return [
    { label: '仪表盘', key: 'dashboard' },
    { label: '文档库', key: 'libraries' },
    { label: '聊天问答', key: 'chat' },
  ]
})

function onSelect(key: string) {
  if (key === 'dashboard') router.push('/')
  else if (key === 'libraries') router.push('/libraries')
  else if (key === 'chat') router.push('/chat')
}

function handleLogout() {
  auth.logout()
  router.push('/login')
}
</script>

<style scoped>
.logo {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 16px 20px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  color: var(--primary-color, #63e2b7);
}
.logo-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: linear-gradient(135deg, #63e2b7, #18a058);
  box-shadow: 0 0 8px rgba(99, 226, 183, 0.6);
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  height: 56px;
}
.header-title {
  font-size: 15px;
  font-weight: 500;
  color: var(--text-color);
}
.header-user {
  display: flex;
  align-items: center;
  gap: 12px;
}
.username {
  font-size: 13px;
  opacity: 0.8;
}
.content {
  padding: 20px 24px;
}
</style>
