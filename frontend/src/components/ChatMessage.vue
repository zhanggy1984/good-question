<template>
  <div class="msg" :class="message.role">
    <div class="avatar" :class="message.role">{{ message.role === 'user' ? '我' : 'AI' }}</div>
    <div class="body">
      <div v-if="message.reasoning" class="reasoning">
        <button class="reasoning-toggle" @click="showReasoning = !showReasoning">
          💭 思考过程 {{ showReasoning ? '收起' : '展开' }}
        </button>
        <div v-if="showReasoning" class="reasoning-content">{{ message.reasoning }}</div>
      </div>
      <div v-if="message.retrievalFailed" class="retrieval-warning">
        ⚠️ 检索服务暂不可用，以下回答未经文档验证，可信度偏低
      </div>
      <div class="bubble">
        <div v-if="message.content" class="answer" :class="{ collapsed: answerCollapsed }">
          <template v-for="(seg, i) in parsedContent" :key="i">
            <span v-if="seg.ref" class="src-ref">{{ seg.text }}</span>
            <template v-else>{{ seg.text }}</template>
          </template>
        </div>
        <span v-else-if="message.streaming && !message.sources?.length" class="retrieving">
          🔍 检索中…
        </span>
        <button v-if="answerCollapsible" class="answer-toggle" @click="showFullAnswer = !showFullAnswer">
          {{ showFullAnswer ? '收起' : '展开完整回答' }}
        </button>
      </div>
      <div v-if="message.retrievalEmpty" class="retrieval-empty">
        ℹ️ 本次未检索到相关内容
      </div>
      <div v-if="message.sources && message.sources.length" class="sources">
        <button class="sources-toggle" @click="showSources = !showSources">
          📎 引用来源（{{ message.sources.length }}）{{ showSources ? '收起' : '展开' }}
        </button>
        <div v-if="showSources">
          <SourceCard v-for="(s, i) in message.sources" :key="i" :source="s" :index="i" :expanded="s.expanded" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import SourceCard from './SourceCard.vue'
import type { Message } from '@/stores/chat'

const props = defineProps<{ message: Message }>()
const showReasoning = ref(false)
const showSources = ref(true)
// 长答案折叠：仅 assistant 且流式结束且内容超阈值时可折叠（流式生成中始终展开，
// 保证输出过程可见 + 自动滚动正常）；折叠态限高可滚动，内容不丢失
const showFullAnswer = ref(false)
const answerCollapsible = computed(
  () => props.message.role === 'assistant' && props.message.content.length > 500 && !props.message.streaming,
)
const answerCollapsed = computed(() => answerCollapsible.value && !showFullAnswer.value)
// 回答中 [来源N] 引用高亮：纯视觉强调（可对应卡片角标编号），不建立跳转。
// 流式生成中 content 增量变化，computed 实时重算；半截未闭合的 "[来源" 不匹配正则，
// 按普通文本展示，不会出现残留高亮
const parsedContent = computed<{ text: string; ref: boolean }[]>(() => {
  const content = props.message.content
  const parts: { text: string; ref: boolean }[] = []
  const re = /\[来源\d+\]/g
  let last = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(content))) {
    if (m.index > last) parts.push({ text: content.slice(last, m.index), ref: false })
    parts.push({ text: m[0], ref: true })
    last = m.index + m[0].length
  }
  if (last < content.length) parts.push({ text: content.slice(last), ref: false })
  return parts.length ? parts : [{ text: content, ref: false }]
})
</script>

<style scoped>
.msg {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
}
.msg.user {
  flex-direction: row-reverse;
}
.avatar {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  flex-shrink: 0;
}
.avatar.user {
  background: linear-gradient(135deg, #18a058, #36ad6a);
  color: #fff;
}
.avatar.assistant {
  background: linear-gradient(135deg, #4b6bff, #63e2b7);
  color: #fff;
}
.body {
  max-width: 72%;
}
.msg.user .body {
  text-align: right;
}
.bubble {
  padding: 10px 14px;
  border-radius: 10px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 14px;
}
.msg.user .bubble {
  background: linear-gradient(135deg, rgba(24, 160, 88, 0.25), rgba(24, 160, 88, 0.15));
  border-top-right-radius: 2px;
}
.msg.assistant .bubble {
  background: rgba(255, 255, 255, 0.06);
  border-top-left-radius: 2px;
}
.answer {
  white-space: pre-wrap;
  word-break: break-word;
}
.answer.collapsed {
  max-height: 240px;
  overflow-y: auto;
}
.src-ref {
  color: var(--primary-color, #63e2b7);
  font-weight: 600;
  background: rgba(99, 226, 183, 0.12);
  border-radius: 3px;
  padding: 0 2px;
}
.answer-toggle {
  display: block;
  background: none;
  border: 1px solid rgba(255, 255, 255, 0.15);
  color: inherit;
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 12px;
  cursor: pointer;
  opacity: 0.7;
  margin-top: 6px;
  text-align: left;
}
.answer-toggle:hover {
  opacity: 1;
}
.retrieving {
  opacity: 0.5;
  font-size: 13px;
}
.retrieval-warning {
  margin-bottom: 8px;
  padding: 8px 12px;
  border-left: 3px solid rgba(217, 160, 46, 0.8);
  background: rgba(217, 160, 46, 0.12);
  border-radius: 0 6px 6px 0;
  font-size: 12px;
  line-height: 1.5;
  color: #e0b64c;
  text-align: left;
}
.retrieval-empty {
  margin-top: 8px;
  font-size: 12px;
  opacity: 0.55;
  text-align: left;
}
.sources {
  margin-top: 8px;
  text-align: left;
}
.sources-label {
  font-size: 11px;
  opacity: 0.5;
  margin-bottom: 4px;
}
.sources-toggle {
  background: none;
  border: 1px solid rgba(255, 255, 255, 0.15);
  color: inherit;
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 12px;
  cursor: pointer;
  opacity: 0.7;
  margin-bottom: 6px;
}
.sources-toggle:hover {
  opacity: 1;
}
.reasoning {
  margin-bottom: 8px;
  text-align: left;
}
.reasoning-toggle {
  background: none;
  border: 1px solid rgba(255, 255, 255, 0.15);
  color: inherit;
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 12px;
  cursor: pointer;
  opacity: 0.7;
}
.reasoning-toggle:hover {
  opacity: 1;
}
.reasoning-content {
  margin-top: 6px;
  padding: 10px 12px;
  border-left: 2px solid rgba(99, 226, 183, 0.4);
  background: rgba(255, 255, 255, 0.03);
  border-radius: 0 6px 6px 0;
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  opacity: 0.75;
  max-height: 300px;
  overflow-y: auto;
}
</style>
