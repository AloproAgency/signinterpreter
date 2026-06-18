import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../lib/api';
import { useApp } from '../lib/context';
import { RotateCcw, ChevronRight, Users } from 'lucide-react';

interface Task {
  task_id: number;
  word: string;
  word_id: number;
  assigned_date: string;
  templates_required: number;
  templates_recorded: number;
  status: 'pending' | 'done' | 'rejected';
  rejected_reason: string | null;
  completed_at: string | null;
}

interface TaskSummary { total: number; pending: number; rejected: number; done: number; }

export default function TeamPage() {
  const { addToast } = useApp();
  const navigate = useNavigate();

  const [memberName, setMemberName] = useState(() => localStorage.getItem('team_member') || '');
  const [isJoined, setIsJoined]     = useState(() => !!localStorage.getItem('team_member'));
  const [joinInput, setJoinInput]   = useState('');
  const [tasks, setTasks]           = useState<Task[]>([]);
  const [summary, setSummary]       = useState<TaskSummary>({ total: 0, pending: 0, rejected: 0, done: 0 });
  const [loading, setLoading]       = useState(false);

  const loadTasks = useCallback(async () => {
    if (!memberName) return;
    setLoading(true);
    try {
      const data = await api.getMyTasks(memberName);
      setTasks(data.tasks);
      setSummary(data.summary);
    } catch { /* member may not exist yet */ }
    setLoading(false);
  }, [memberName]);

  useEffect(() => { if (isJoined && memberName) loadTasks(); }, [isJoined, memberName, loadTasks]);

  const handleJoin = async () => {
    const name = joinInput.trim();
    if (!name) return;
    try {
      await api.joinTeam(name);
      setMemberName(name);
      localStorage.setItem('team_member', name);
      setIsJoined(true);
      addToast('success', `Bienvenue ${name}`);
    } catch { addToast('error', 'Erreur'); }
  };

  const handleLeave = () => {
    localStorage.removeItem('team_member');
    setMemberName(''); setIsJoined(false); setTasks([]);
  };

  const goRecord = (task: Task) => {
    localStorage.setItem('contributor', memberName);
    localStorage.setItem('prefill_word', task.word);
    localStorage.setItem('prefill_task_id', String(task.task_id));
    localStorage.setItem('prefill_count', String(task.templates_required));
    navigate('/contribute');
  };

  const tasksByDate = tasks.reduce<Record<string, Task[]>>((acc, t) => {
    if (!acc[t.assigned_date]) acc[t.assigned_date] = [];
    acc[t.assigned_date].push(t);
    return acc;
  }, {});
  const sortedDates = Object.keys(tasksByDate).sort().reverse();
  const today = new Date().toISOString().slice(0, 10);

  /* ── Join screen ───────────────────────────────── */
  if (!isJoined) {
    return (
      <div className="h-full flex items-center justify-center p-6 bg-zinc-50 dark:bg-zinc-950">
        <div className="w-full max-w-sm animate-scale-in">
          <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-xl shadow-sm p-8">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-blue-50 dark:bg-blue-950/50 rounded-lg">
                <Users className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-zinc-900 dark:text-zinc-50">Rejoindre l'équipe</h2>
                <p className="text-sm text-zinc-500">Signez les mots assignés</p>
              </div>
            </div>
            <input
              type="text"
              value={joinInput}
              onChange={e => setJoinInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleJoin()}
              placeholder="Votre nom"
              className="w-full px-3 py-2.5 text-sm border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 rounded-md placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 mb-3 text-zinc-900 dark:text-zinc-50"
            />
            <button
              onClick={handleJoin}
              disabled={!joinInput.trim()}
              className="w-full py-2.5 text-sm font-medium bg-blue-600 hover:bg-blue-700 text-white rounded-md disabled:opacity-40 transition-colors cursor-pointer"
            >
              Rejoindre
            </button>
          </div>
        </div>
      </div>
    );
  }

  /* ── Dashboard ─────────────────────────────────── */
  return (
    <div className="h-full flex flex-col bg-white dark:bg-zinc-950">

      {/* Header */}
      <div className="shrink-0 h-12 px-4 flex items-center justify-between border-b border-zinc-200 dark:border-zinc-800">
        <div className="flex items-center gap-4">
          <span className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">{memberName}</span>
          <div className="hidden sm:flex items-center gap-3 text-xs font-mono text-zinc-400">
            {(summary.pending + summary.rejected) > 0 && (
              <span className="bg-blue-50 dark:bg-blue-950/40 text-blue-600 dark:text-blue-400 px-2 py-0.5 rounded font-medium">
                {summary.pending + summary.rejected} à faire
              </span>
            )}
            <span>{summary.done} faits</span>
            <span className="text-zinc-300 dark:text-zinc-700">·</span>
            <span>{summary.total} total</span>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={loadTasks}
            className="p-2 text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-md transition-colors cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleLeave}
            className="px-3 py-1.5 text-xs border border-zinc-200 dark:border-zinc-700 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-zinc-800 rounded-md transition-colors cursor-pointer"
          >
            Quitter
          </button>
        </div>
      </div>

      {/* Task list */}
      <div className="flex-1 min-h-0 overflow-auto">
        {loading ? (
          <div className="p-6 space-y-2">
            {[1,2,3,4].map(i => <div key={i} className="h-14 animate-pulse bg-zinc-100 dark:bg-zinc-800/50 rounded-lg" />)}
          </div>
        ) : tasks.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-3">
            <p className="text-sm text-zinc-400">Aucune tâche pour le moment</p>
            <p className="text-xs text-zinc-300 dark:text-zinc-600">L'admin n'a pas encore assigné de mots</p>
          </div>
        ) : (
          <table className="w-full text-sm border-collapse">
            <thead className="sticky top-0 bg-zinc-50 dark:bg-zinc-900 border-b border-zinc-200 dark:border-zinc-800">
              <tr>
                <th className="text-left px-4 py-2.5 text-[10px] uppercase tracking-widest font-medium text-zinc-400">Date</th>
                <th className="text-left px-4 py-2.5 text-[10px] uppercase tracking-widest font-medium text-zinc-400">Mot</th>
                <th className="text-left px-4 py-2.5 text-[10px] uppercase tracking-widest font-medium text-zinc-400">Progrès</th>
                <th className="text-center px-4 py-2.5 text-[10px] uppercase tracking-widest font-medium text-zinc-400">Statut</th>
                <th className="px-4 py-2.5" />
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60">
              {sortedDates.flatMap(dateStr =>
                tasksByDate[dateStr].map(task => (
                  <tr key={task.task_id} className={`hover:bg-zinc-50 dark:hover:bg-zinc-900/40 transition-colors ${task.status === 'done' ? 'opacity-50' : ''}`}>
                    <td className="px-4 py-3 text-xs font-mono text-zinc-400 whitespace-nowrap">
                      {dateStr === today
                        ? <span className="text-blue-600 dark:text-blue-400 font-medium">Aujourd'hui</span>
                        : new Date(dateStr).toLocaleDateString('fr-FR', { day: 'numeric', month: 'short' })}
                    </td>
                    <td className="px-4 py-3 font-medium text-zinc-900 dark:text-zinc-50">{task.word}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2.5">
                        <span className="text-xs font-mono text-zinc-500 tabular-nums">
                          {task.templates_recorded}/{task.templates_required}
                        </span>
                        <div className="w-20 h-1.5 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-500 rounded-full"
                            style={{ width: `${(task.templates_recorded / task.templates_required) * 100}%` }}
                          />
                        </div>
                      </div>
                      {task.status === 'rejected' && task.rejected_reason && (
                        <p className="text-xs text-red-500 dark:text-red-400 mt-0.5">{task.rejected_reason}</p>
                      )}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium uppercase tracking-wide ${
                        task.status === 'done'
                          ? 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-400'
                          : task.status === 'rejected'
                          ? 'bg-red-50 dark:bg-red-950/40 text-red-600 dark:text-red-400'
                          : 'bg-blue-50 dark:bg-blue-950/40 text-blue-600 dark:text-blue-400'
                      }`}>
                        {task.status === 'done' ? 'fait'
                         : task.status === 'rejected' ? 'rejeté'
                         : 'en attente'}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      {task.status !== 'done' && (
                        <button
                          onClick={() => goRecord(task)}
                          className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors cursor-pointer whitespace-nowrap"
                        >
                          {task.status === 'rejected' ? 'Reprendre' : 'Signer'}
                          <ChevronRight className="w-3 h-3" />
                        </button>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
