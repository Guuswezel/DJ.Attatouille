import type { Mix, MixOptions, Overview, Preparation, SkipPlan } from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({})) as { error?: string }
    throw new Error(body.error ?? `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export const api = {
  overview: () => request<Overview>('/api/overview'),
  createPreparation: (payload: { sourcePath: string; label: string }) => request<Preparation>('/api/preparations', {
    method: 'POST', body: JSON.stringify(payload),
  }),
  createMix: (payload: MixOptions) => request<Mix>('/api/mixes', {
    method: 'POST', body: JSON.stringify(payload),
  }),
  deletePreparation: (preparationId: string) => request<{ deletedPreparationId: string; deletedMixIds: string[] }>(`/api/preparations/${preparationId}`, { method: 'DELETE' }),
  retryPreparation: (preparationId: string) => request<Preparation>(`/api/preparations/${preparationId}/retry`, { method: 'POST' }),
  deleteMix: (mixId: string) => request<{ deletedMixId: string }>(`/api/mixes/${mixId}`, { method: 'DELETE' }),
  nextTransition: (mixId: string, currentTrackIndex: number, playbackSeconds: number) => request<SkipPlan>(
    `/api/mixes/${mixId}/next-transition`,
    { method: 'POST', body: JSON.stringify({ currentTrackIndex, playbackSeconds }) },
  ),
}
