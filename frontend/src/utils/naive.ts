/**
 * Naive UI 离散 API：message/dialog 无需组件树中的 provider 包裹即可使用。
 * 解决懒加载路由组件中 useMessage/useDialog 找不到 provider 的问题。
 */
import { createDiscreteApi } from 'naive-ui'

export const { message, dialog } = createDiscreteApi(['message', 'dialog'])
