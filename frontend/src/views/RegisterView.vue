<template>
  <div class="auth-page">
    <n-card class="auth-card" :bordered="false">
      <div class="brand">
        <span class="logo-dot"></span>
        <h1>注册账号</h1>
        <p>加入不懂就问文档问答</p>
      </div>
      <n-form ref="formRef" :model="form" :rules="rules" size="large">
        <n-form-item path="username" label="用户名">
          <n-input v-model:value="form.username" placeholder="3-50 位字符" />
        </n-form-item>
        <n-form-item path="password" label="密码">
          <n-input
            v-model:value="form.password"
            type="password"
            show-password-on="click"
            placeholder="至少 6 位"
          />
        </n-form-item>
        <n-form-item path="confirm" label="确认密码">
          <n-input
            v-model:value="form.confirm"
            type="password"
            show-password-on="click"
            placeholder="再次输入密码"
          />
        </n-form-item>
        <n-button type="primary" block :loading="loading" @click="handleRegister">注 册</n-button>
      </n-form>
      <div class="auth-footer">
        已有账号？
        <n-button text type="primary" @click="router.push('/login')">去登录</n-button>
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
const form = ref({ username: '', password: '', confirm: '' })

const rules = {
  username: {
    required: true,
    min: 3,
    max: 50,
    message: '用户名 3-50 位',
    trigger: 'blur',
  },
  password: { required: true, min: 6, message: '密码至少 6 位', trigger: 'blur' },
  confirm: {
    required: true,
    validator: (_: any, value: string) => value === form.value.password,
    message: '两次密码不一致',
    trigger: 'blur',
  },
}

async function handleRegister() {
  await formRef.value?.validate().catch(() => {})
  loading.value = true
  try {
    await auth.register(form.value.username, form.value.password)
    message.success('注册成功，请登录')
    router.push('/login')
  } catch (e: any) {
    message.error(e.response?.data?.error?.message || '注册失败')
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
