/** 通用分页响应 */
export interface Page<T> {
  items: T[]
  total: number
  page: number
  page_size: number
}
