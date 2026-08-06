<template>
  <div class="chat-input">
    <n-input
      v-model:value="text"
      type="textarea"
      :autosize="{ minRows: 1, maxRows: 5 }"
      placeholder="输入问题，Enter 发送，Shift+Enter 换行"
      :disabled="disabled"
      @keydown="onKeydown"
    />
    <n-button type="primary" :disabled="disabled || !text.trim()" :loading="disabled" @click="send">
      发送
    </n-button>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { NInput, NButton } from 'naive-ui'

const props = defineProps<{ disabled?: boolean }>()
const emit = defineEmits<{ (e: 'send', content: string): void }>()

const text = ref('')

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault()
    if (text.value.trim()) send()
  }
}

function send() {
  const content = text.value.trim()
  if (!content || props.disabled) return
  emit('send', content)
  text.value = ''
}
</script>

<style scoped>
.chat-input {
  display: flex;
  gap: 12px;
  padding: 12px 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.chat-input :deep(.n-input) {
  flex: 1;
}
</style>
