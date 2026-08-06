/** SSE 流式接收：POST /api/chat/{sessionId}，解析 event/data 块 */

export interface ChatEvent {
  type: 'sources' | 'token' | 'reasoning' | 'done' | 'error' | 'message'
  data: Record<string, any>
}

/**
 * 流式聊天，逐事件回调 onEvent
 * 网络中断时抛 StreamError，由上层处理"重试"
 */
export async function streamChat(
  sessionId: number,
  content: string,
  token: string,
  onEvent: (ev: ChatEvent) => void,
): Promise<void> {
  const resp = await fetch(`/api/chat/${sessionId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ content }),
  })

  if (!resp.ok) {
    throw new Error(`请求失败：HTTP ${resp.status}`)
  }
  if (!resp.body) {
    throw new Error('响应无 body')
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      // 按空行切分 SSE 块
      const parts = buffer.split('\n\n')
      buffer = parts.pop() || ''
      for (const part of parts) {
        parseBlock(part, onEvent)
      }
    }
  } catch (e) {
    throw new Error('连接中断，请重试')
  }
}

function parseBlock(block: string, onEvent: (ev: ChatEvent) => void) {
  let event = 'message'
  let data = ''
  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      data += line.slice(5).trim()
    }
  }
  if (data) {
    try {
      onEvent({ type: event as ChatEvent['type'], data: JSON.parse(data) })
    } catch {
      // 忽略无法解析的数据
    }
  }
}
