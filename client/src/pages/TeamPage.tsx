import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../lib/api';
import { useApp } from '../lib/context';
import {
  Users, UserPlus, CheckCircle, Clock, AlertTriangle,
  ChevronRight, Calendar, RotateCcw,
} from 'lucide-react';

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

interface TaskSummary {
  total: number;
  pending: number;
  rejected: number;
  done: number;
}

export default function TeamPage() {
  const { addToast } = useApp();
  const navigate = useNavigate();

  const [memberName, setMemberName] = useState(() => localStorage.getItem('team_member') || '');
  const [isJoined, setIsJoined] = useState(() => !!localStorage.getItem('team_member'));
  const [joinInput, setJoinInput] = useState('');

  const [tasks, setTasks] = useState<Task[]>([]);
  const [summary, setSummary] = useState<TaskSummary>({ total: 0, pending: 0, rejected: 0, done: 0 });
  const [loading, setLoading] = useState(false);

  const loadTasks = useCallback(async () => {
    if (!memberName) return;
    setLoading(true);
    try {
      const data = await api.getMyTasks(memberName);
      setTasks(data.tasks);
      setSummary(data.summary);
    } catch {
      // Member might not exist yet
    }
    setLoading(false);
  }, [memberName]);

  useEffect(() => {
    if (isJoined && memberName) loadTasks();
  }, [isJoined, memberName, loadTasks]);

  const handleJoin = async () => {
    const name = joinInput.trim();
    if (!name) return;
    try {
      await api.joinTeam(name);
      setMemberName(name);
      localStorage.setItem('team_member', name);
      setIsJoined(true);
      addToast('success', `Bienvenue ${name} !`);
    } catch {
      addToast('error', 'Erreur');
    }
  };

  const handleLeave = () => {
    localStorage.removeItem('team_member');
    setMemberName('');
    setIsJoined(false);
    setTasks([]);
  };

  const goRecord = (task: Task) => {
    // Navigate to contribution page with word pre-filled
    localStorage.setItem('contributor', memberName);
    localStorage.setItem('prefill_word', task.word);
    localStorage.setItem('prefill_task_id', String(task.task_id));
    localStorage.setItem('prefill_count', String(task.templates_required));
    navigate('/contribute');
  };

  // Group tasks by date
  const tasksByDate = tasks.reduce<Record<string, Task[]>>((acc, t) => {
    const d = t.assigned_date;
    if (!acc[d]) acc[d] = [];
    acc[d].push(t);
    return acc;
  }, {});
  const sortedDates = Object.keys(tasksByDate).sort().reverse();

  const today = new Date().toISOString().slice(0, 10);

  // Join screen
  if (!isJoined) {
    return (
      <div className="h-full flex items-center justify-center p-4 bg-white dark:bg-[#0d1117]">
        <div className="w-full max-w-sm animate-scale-in">
          <div className="rounded-lg border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22] p-8 text-center">
            <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4 bg-[rgba(19,150,186,0.1)] border border-[rgba(19,150,186,0.15)]">
              <Users className="w-8 h-8 text-[#1396ba]" />
            </div>
            <h2 className="text-xl font-bold mb-2 text-[#1f2328] dark:text-[#e6edf3]">Rejoindre l'equipe</h2>
            <p className="text-sm mb-6 text-[#656d76] dark:text-[#8b949e]">
              Contribuez au vocabulaire en signant les mots du jour
            </p>
            <input
              type="text"
              value={joinInput}
              onChange={e => setJoinInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleJoin()}
              placeholder="Votre nom"
              className="w-full rounded-md px-4 py-3 text-base mb-4 bg-white dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] text-[#1f2328] dark:text-[#e6edf3] placeholder-[#8b949e] focus:outline-none focus:border-[#1396ba] focus:ring-1 focus:ring-[#1396ba]/30"
            />
            <button
              onClick={handleJoin}
              disabled={!joinInput.trim()}
              className="w-full py-3 rounded-md text-base font-medium bg-[#1396ba] hover:bg-[#17b8e3] text-white disabled:opacity-40 cursor-pointer transition-colors flex items-center justify-center gap-2"
            >
              <UserPlus className="w-5 h-5" />
              Rejoindre
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full p-4 md:p-6 overflow-auto bg-white dark:bg-[#0d1117]">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-[#1f2328] dark:text-[#e6edf3]">
            Bonjour, {memberName}
          </h1>
          <p className="text-sm mt-1 text-[#656d76] dark:text-[#8b949e]">
            Vos taches de contribution
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={loadTasks}
            className="p-2 rounded-md text-[#8b949e] hover:text-[#1396ba] hover:bg-[rgba(19,150,186,0.1)] transition-colors cursor-pointer"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <button
            onClick={handleLeave}
            className="px-3 py-1.5 rounded-md text-sm border border-[#d0d7de] dark:border-[#30363d] text-[#656d76] dark:text-[#8b949e] hover:bg-[#f6f8fa] dark:hover:bg-[#161b22] cursor-pointer transition-colors"
          >
            Quitter
          </button>
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-4 mb-8 max-w-lg">
        <div className="rounded-lg p-4 border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22] text-center">
          <p className="text-2xl font-bold font-mono text-[#f59e0b]">{summary.pending + summary.rejected}</p>
          <p className="text-sm mt-1 text-[#656d76] dark:text-[#8b949e]">A faire</p>
        </div>
        <div className="rounded-lg p-4 border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22] text-center">
          <p className="text-2xl font-bold font-mono text-[#10b981]">{summary.done}</p>
          <p className="text-sm mt-1 text-[#656d76] dark:text-[#8b949e]">Fait</p>
        </div>
        <div className="rounded-lg p-4 border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22] text-center">
          <p className="text-2xl font-bold font-mono text-[#1396ba]">{summary.total}</p>
          <p className="text-sm mt-1 text-[#656d76] dark:text-[#8b949e]">Total</p>
        </div>
      </div>

      {/* Tasks by date */}
      {loading ? (
        <p className="text-sm text-[#8b949e]">Chargement...</p>
      ) : tasks.length === 0 ? (
        <div className="text-center py-16">
          <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4 bg-[rgba(19,150,186,0.1)] border border-[rgba(19,150,186,0.15)]">
            <Calendar className="w-8 h-8 text-[#1396ba]/50" />
          </div>
          <p className="text-base text-[#8b949e]">Aucune tache pour le moment</p>
          <p className="text-sm mt-1 text-[#656d76]">L'admin vous assignera des mots bientot</p>
        </div>
      ) : (
        <div className="space-y-6 max-w-2xl">
          {sortedDates.map(dateStr => (
            <div key={dateStr}>
              <h3 className="text-sm font-medium uppercase tracking-wider mb-3 flex items-center gap-2 text-[#656d76] dark:text-[#8b949e]">
                <Calendar className="w-3.5 h-3.5" />
                {dateStr === today ? "Aujourd'hui" : new Date(dateStr).toLocaleDateString('fr-FR', { weekday: 'long', day: 'numeric', month: 'long' })}
              </h3>
              <div className="space-y-2">
                {tasksByDate[dateStr].map(task => (
                  <div
                    key={task.task_id}
                    className={`rounded-lg border p-4 flex items-center gap-4 transition-colors ${
                      task.status === 'done'
                        ? 'border-[#10b981]/20 bg-[#10b981]/5'
                        : task.status === 'rejected'
                        ? 'border-[#ef4444]/20 bg-[#ef4444]/5'
                        : 'border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22]'
                    }`}
                  >
                    {/* Status icon */}
                    {task.status === 'done' ? (
                      <CheckCircle className="w-6 h-6 text-[#10b981] shrink-0" />
                    ) : task.status === 'rejected' ? (
                      <AlertTriangle className="w-6 h-6 text-[#ef4444] shrink-0" />
                    ) : (
                      <Clock className="w-6 h-6 text-[#f59e0b] shrink-0" />
                    )}

                    {/* Word info */}
                    <div className="flex-1">
                      <p className="text-base font-bold text-[#1f2328] dark:text-[#e6edf3]">{task.word}</p>
                      <p className="text-sm text-[#656d76] dark:text-[#8b949e]">
                        {task.templates_recorded}/{task.templates_required} templates
                      </p>
                      {task.status === 'rejected' && task.rejected_reason && (
                        <p className="text-sm mt-1 text-[#ef4444]">
                          Rejete : {task.rejected_reason}
                        </p>
                      )}
                    </div>

                    {/* Action */}
                    {task.status !== 'done' && (
                      <button
                        onClick={() => goRecord(task)}
                        className="px-4 py-2 rounded-md text-sm font-medium bg-[#1396ba] hover:bg-[#17b8e3] text-white cursor-pointer transition-colors flex items-center gap-1.5 shrink-0"
                      >
                        {task.status === 'rejected' ? 'Reprendre' : 'Signer'}
                        <ChevronRight className="w-4 h-4" />
                      </button>
                    )}

                    {task.status === 'done' && (
                      <span className="text-sm font-medium text-[#10b981]">Termine</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
