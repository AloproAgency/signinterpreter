const BASE = '/api';

async function request(path: string, options?: RequestInit) {
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const api = {
  // Vocabulary
  getVocabulary: () => request('/vocabulary'),
  getWord: (name: string) => request(`/vocabulary/${encodeURIComponent(name)}`),
  createWord: (name: string, description = '') =>
    request('/vocabulary', { method: 'POST', body: JSON.stringify({ name, description }) }),
  deleteWord: (name: string) =>
    request(`/vocabulary/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  updateWord: (name: string, data: { description?: string; is_active?: boolean }) =>
    request(`/vocabulary/${encodeURIComponent(name)}`, { method: 'PATCH', body: JSON.stringify(data) }),

  // Contributions
  getContributions: (status?: string) =>
    request(`/contributions${status ? `?status=${status}` : ''}`),
  reviewContribution: (id: number, status: string, notes = '') =>
    request(`/contributions/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ status, notes, reviewed_by: 'admin' }),
    }),
  deleteContribution: (id: number) =>
    request(`/contributions/${id}`, { method: 'DELETE' }),

  // Contribution video preview
  getContributionVideoUrl: (id: number) => `${BASE}/contributions/${id}/video`,

  // Dataset words (SL reference)
  getDatasetWords: () => request('/dataset-words'),

  // Reference videos
  getReference: (word: string) =>
    request(`/reference/${encodeURIComponent(word)}`),
  getReferenceVideoUrl: (word: string) =>
    `${BASE}/reference/${encodeURIComponent(word)}/video`,

  // Team
  joinTeam: (name: string) =>
    request('/team/join', { method: 'POST', body: JSON.stringify({ name }) }),
  getMembers: () => request('/team/members'),
  getMyTasks: (memberName: string) =>
    request(`/team/my-tasks?member_name=${encodeURIComponent(memberName)}`),
  completeTask: (taskId: number) =>
    request(`/team/tasks/${taskId}/complete`, { method: 'POST' }),
  updateTaskProgress: (taskId: number, recorded: number) =>
    request(`/team/tasks/${taskId}/progress?templates_recorded=${recorded}`, { method: 'POST' }),

  // Team admin
  assignWords: (wordNames: string[], assignedDate?: string, templatesRequired = 8) =>
    request('/team/admin/assign', {
      method: 'POST',
      body: JSON.stringify({ word_names: wordNames, assigned_date: assignedDate, templates_required: templatesRequired }),
    }),
  getAssignments: (targetDate?: string) =>
    request(`/team/admin/assignments${targetDate ? `?target_date=${targetDate}` : ''}`),
  rejectTask: (taskId: number, reason = '') =>
    request(`/team/admin/tasks/${taskId}/reject`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    }),
  deleteAssignment: (id: number) =>
    request(`/team/admin/assignments/${id}`, { method: 'DELETE' }),

  // Admin
  login: (password: string) =>
    request('/admin/login', { method: 'POST', body: JSON.stringify({ password }) }),
  buildIndex: () => request('/admin/build-index', { method: 'POST' }),
  getStats: () => request('/admin/stats'),
};
