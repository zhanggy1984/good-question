<template>
  <div class="auth-page">
    <n-card class="auth-card" :bordered="false">
      <div class="brand">
        <span class="logo-dot"></span>
        <h1>不懂就问</h1>
        <p>文档问答系统</p>
      </div>
      <n-form ref="formRef" :model="form" :rules="rules" size="large">
        <n-form-item path="username" label="用户名">
          <n-input v-model:value="form.username" placeholder="请输入用户名" @keyup.enter="handleLogin" />
        </n-form-item>
        <n-form-item path="password" label="密码">
          <n-input
            v-model:value="form.password"
            type="password"
            show-password-on="click"
            placeholder="请输入密码"
            @keyup.enter="handleLogin"
          />
        </n-form-item>
        <n-button type="primary" block :loading="loading" @click="handleLogin">登 录</n-button>
      </n-form>
      <div class="auth-footer">
        还没有账号？
        <n-button text type="primary" @click="router.push('/register')">去注册</n-button>
      </div>
    </n-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { NForm, NFormItem, NInput, NButton, NCard } from 'naive-ui'
import { useAuthStore } from '@/stores/auth'
import { message } from '@/utils/naive'

const router = useRouter()
const auth = useAuthStore()

const formRef = ref()
const loading = ref(false)
const form = ref({ username: '', password: '' })

const rules = {
  username: { required: true, message: '请输入用户名', trigger: 'blur' },
  password: { required: true, message: '请输入密码', trigger: 'blur' },
}

async function handleLogin() {
  await formRef.value?.validate().catch(() => {})
  loading.value = true
  try {
    await auth.login(form.value.username, form.value.password)
    message.success('登录成功')
    router.push('/')
  } catch (e: any) {
    message.error(e.response?.data?.error?.message || '登录失败')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.auth-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: radial-gradient(ellipse at top, rgba(24, 160, 88, 0.08), transparent 60%),
    var(--body-color);
}
.auth-card {
  width: 380px;
  padding: 20px;
  border-radius: 12px;
}
.brand {
  text-align: center;
  margin-bottom: 24px;
}
.brand h1 {
  margin: 8px 0 4px;
  font-size: 24px;
  color: var(--primary-color, #63e2b7);
}
.brand p {
  margin: 0;
  font-size: 13px;
  opacity: 0.6;
}
.logo-dot {
  display: inline-block;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: linear-gradient(135deg, #63e2b7, #18a058);
  box-shadow: 0 0 12px rgba(99, 226, 183, 0.7);
}
.auth-footer {
  margin-top: 16px;
  text-align: center;
  font-size: 13px;
}
</style>
