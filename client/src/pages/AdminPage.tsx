import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '../lib/api';
import { useApp } from '../lib/context';
import {
  Database, CheckCircle, XCircle, Clock,
  RefreshCw, Trash2, Plus, Lock,
  Download, Edit3, Save, X,
  Cpu, HardDrive, Zap, Eye, BarChart3, Play, Users, Calendar, AlertTriangle, Activity,
} from 'lucide-react';
import type { StatsInfo, ContributionInfo, WordInfo } from '../lib/types';
import { SkeletonPlayer } from '../components/SkeletonPlayer';

type Tab = 'dashboard' | 'contributions' | 'index' | 'words' | 'team';
type ContribFilter = 'all' | 'pending' | 'approved' | 'rejected';

export default function AdminPage() {
  const { addToast } = useApp();

  const [logged, setLogged] = useState(() => sessionStorage.getItem('admin_logged') === 'true');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');

  const [stats, setStats] = useState<StatsInfo | null>(null);
  const [contributions, setContributions] = useState<ContributionInfo[]>([]);
  const [words, setWords] = useState<WordInfo[]>([]);
  const [building, setBuilding] = useState(false);
  const [buildProgress, setBuildProgress] = useState(0);
  const [trainLogs, setTrainLogs] = useState<{epoch:number;total:number;acc:number;val_acc:number;loss:number;val_loss:number}[]>([]);
  const [trainDone, setTrainDone] = useState<null | {meta: any}>(null);
  const trainLogRef = useRef<HTMLDivElement>(null);
  const [newWordName, setNewWordName] = useState('');
  const [tab, setTab] = useState<Tab>('dashboard');
  const [contribFilter, setContribFilter] = useState<ContribFilter>('pending');
  const [selectedContribs, setSelectedContribs] = useState<Set<number>>(new Set());
  const [editingWord, setEditingWord] = useState<string | null>(null);
  const [editDescription, setEditDescription] = useState('');

  const [assignments, setAssignments] = useState<any[]>([]);
  const [members, setMembers] = useState<any[]>([]);
  const [assignDate, setAssignDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [datasetWords, setDatasetWords] = useState<string[]>([]);
  const [wordSearch, setWordSearch] = useState('');
  const [selectedWords, setSelectedWords] = useState<Set<string>>(new Set());
  const [showWordPicker, setShowWordPicker] = useState(false);
  const [previewId, setPreviewId] = useState<number | null>(null);
  const [previewContrib, setPreviewContrib] = useState<ContributionInfo | null>(null);

  const login = async () => {
    try {
      await api.login(password);
      setLogged(true);
      sessionStorage.setItem('admin_logged', 'true');
      setLoginError('');
      addToast('success', 'Connexion réussie');
    } catch { setLoginError('Mot de passe incorrect'); }
  };

  const refresh = useCallback(async () => {
    try {
      const [s, c, w, a, m] = await Promise.all([
        api.getStats(), api.getContributions(), api.getVocabulary(),
        api.getAssignments().catch(() => []),
        api.getMembers().catch(() => []),
      ]);
      setStats(s); setContributions(c); setWords(w);
      setAssignments(a); setMembers(m);
      if (datasetWords.length === 0)
        api.getDatasetWords().then(setDatasetWords).catch(() => {});
    } catch { addToast('error', 'Erreur de chargement'); }
  }, [addToast]);

  useEffect(() => { if (logged) refresh(); }, [logged, refresh]);

  const handleTrain = () => {
    if (building) return;
    setBuilding(true); setBuildProgress(0); setTrainLogs([]); setTrainDone(null);
    const es = api.trainStream(undefined, 200);
    es.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === 'epoch') {
        setTrainLogs(prev => {
          const next = [...prev, msg];
          setTimeout(() => trainLogRef.current?.scrollTo({ top: 99999, behavior: 'smooth' }), 30);
          return next;
        });
        setBuildProgress(Math.round((msg.epoch / msg.total) * 100));
      } else if (msg.type === 'done') {
        setBuildProgress(100); setTrainDone({ meta: msg.meta }); setBuilding(false);
        addToast('success', `Terminé — ${msg.meta.n_classes} classes, val_acc ${(msg.meta.best_val_accuracy * 100).toFixed(1)}%`);
        refresh(); es.close();
      } else if (msg.type === 'error') {
        addToast('error', `Erreur : ${msg.message}`); setBuilding(false); es.close();
      }
    };
    es.onerror = () => { addToast('error', 'Connexion SSE perdue'); setBuilding(false); es.close(); };
  };

  const handleReview = async (id: number, status: 'approved' | 'rejected') => {
    try {
      await api.reviewContribution(id, status);
      addToast('success', status === 'approved' ? 'Approuvée' : 'Rejetée');
      refresh();
    } catch { addToast('error', 'Erreur'); }
  };

  const handleBatchReview = async (status: 'approved' | 'rejected') => {
    const ids = Array.from(selectedContribs);
    if (!ids.length) return;
    try {
      await Promise.all(ids.map(id => api.reviewContribution(id, status)));
      addToast('success', `${ids.length} ${status === 'approved' ? 'approuvées' : 'rejetées'}`);
      setSelectedContribs(new Set()); refresh();
    } catch { addToast('error', 'Erreur batch'); }
  };

  const handleCreateWord = async () => {
    if (!newWordName.trim()) return;
    try {
      await api.createWord(newWordName.trim());
      addToast('success', `"${newWordName.trim()}" créé`);
      setNewWordName(''); refresh();
    } catch { addToast('error', 'Erreur'); }
  };

  const handleDeleteWord = async (name: string) => {
    if (!confirm(`Supprimer "${name}" et tous ses templates ?`)) return;
    try {
      await api.deleteWord(name);
      addToast('success', `"${name}" supprimé`); refresh();
    } catch { addToast('error', 'Erreur'); }
  };

  const handleSaveDescription = async (name: string) => {
    try {
      await api.updateWord(name, { description: editDescription });
      setEditingWord(null); addToast('success', 'Description mise à jour'); refresh();
    } catch { addToast('error', 'Erreur'); }
  };

  const handleToggleActive = async (word: WordInfo) => {
    try {
      await api.updateWord(word.name, { is_active: !word.is_active });
      addToast('info', `${word.name} ${!word.is_active ? 'activé' : 'désactivé'}`); refresh();
    } catch { addToast('error', 'Erreur'); }
  };

  const toggleContribSelection = (id: number) => {
    setSelectedContribs(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const filteredContribs = contributions.filter(c => contribFilter === 'all' || c.status === contribFilter);

  const handleExport = () => {
    const blob = new Blob([JSON.stringify({ words, contributions }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `signinterpreter-${new Date().toISOString().slice(0, 10)}.json`;
    a.click(); URL.revokeObjectURL(url);
    addToast('success', 'Exporté');
  };

  const openPreview = (c: ContributionInfo) => { setPreviewId(c.id); setPreviewContrib(c); };
  const closePreview = () => { setPreviewId(null); setPreviewContrib(null); };

  const inputCls = 'px-3 py-1.5 text-sm bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-700 rounded-md placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 text-zinc-900 dark:text-zinc-50';
  const btnPrimary = 'px-4 py-2 text-sm font-medium bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors cursor-pointer';
  const btnSecondary = 'px-3 py-1.5 text-sm border border-zinc-200 dark:border-zinc-700 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-zinc-800 rounded-md transition-colors cursor-pointer';
  const iconBtn = 'p-1.5 text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-md transition-colors cursor-pointer';

  /* ── Login ─────────────────────────────────────── */
  if (!logged) {
    return (
      <div className="h-full flex items-center justify-center p-6 bg-zinc-50 dark:bg-zinc-950">
        <div className="w-full max-w-xs animate-scale-in">
          <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-xl shadow-sm p-8">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-zinc-100 dark:bg-zinc-800 rounded-lg">
                <Lock className="w-5 h-5 text-zinc-600 dark:text-zinc-400" />
              </div>
              <span className="text-base font-semibold text-zinc-900 dark:text-zinc-50">Administration</span>
            </div>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && login()}
              placeholder="Mot de passe"
              className={`${inputCls} w-full mb-2`}
            />
            {loginError && (
              <p className="text-xs text-red-500 mb-3">{loginError}</p>
            )}
            <button onClick={login} className={`${btnPrimary} w-full justify-center mt-2`}>
              Connexion
            </button>
          </div>
        </div>
      </div>
    );
  }

  const TABS: { key: Tab; label: string; icon: React.ElementType }[] = [
    { key: 'dashboard',     label: 'Dashboard',    icon: BarChart3 },
    { key: 'team',          label: 'Équipe',        icon: Users     },
    { key: 'contributions', label: 'Contributions', icon: Eye       },
    { key: 'index',         label: 'Modèle',        icon: Cpu       },
    { key: 'words',         label: 'Mots',          icon: Database  },
  ];

  return (
    <div className="h-full flex flex-col bg-white dark:bg-zinc-950">

      {/* ── Tab bar ──────────────────────────────────── */}
      <div className="shrink-0 flex items-center border-b border-zinc-200 dark:border-zinc-800 px-4 h-12">
        <div className="flex items-center gap-0.5 flex-1">
          {TABS.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`flex items-center gap-1.5 px-3 h-12 text-sm transition-colors cursor-pointer border-b-2 -mb-px ${
                tab === key
                  ? 'border-blue-600 dark:border-blue-400 text-blue-600 dark:text-blue-400 font-medium'
                  : 'border-transparent text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-zinc-800/50'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">{label}</span>
            </button>
          ))}
        </div>
        <div className="flex items-center gap-0.5">
          <button onClick={handleExport} className={iconBtn} title="Exporter">
            <Download className="w-4 h-4" />
          </button>
          <button onClick={refresh} className={iconBtn} title="Rafraîchir">
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* ── Tab content ──────────────────────────────── */}
      <div className="flex-1 min-h-0 overflow-auto p-6">

        {/* Dashboard */}
        {tab === 'dashboard' && stats && (
          <div className="space-y-6 animate-fade-in max-w-3xl">
            {/* Stat grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                { label: 'Mots',       value: stats.words,                   icon: Database,    color: 'text-blue-600 dark:text-blue-400',    bg: 'bg-blue-50 dark:bg-blue-950/40'    },
                { label: 'Templates',  value: stats.templates,                icon: HardDrive,   color: 'text-violet-600 dark:text-violet-400', bg: 'bg-violet-50 dark:bg-violet-950/40' },
                { label: 'En attente', value: stats.contributions.pending,    icon: Clock,       color: 'text-amber-600 dark:text-amber-400',   bg: 'bg-amber-50 dark:bg-amber-950/40'   },
                { label: 'Approuvés',  value: stats.contributions.approved,   icon: CheckCircle, color: 'text-emerald-600 dark:text-emerald-400', bg: 'bg-emerald-50 dark:bg-emerald-950/40' },
              ].map(({ label, value, icon: Icon, color, bg }) => (
                <AnimatedStat key={label} label={label} value={value} icon={Icon} color={color} iconBg={bg} />
              ))}
            </div>

            {/* Engine status */}
            <div>
              <h3 className="text-xs uppercase tracking-widest font-medium text-zinc-400 mb-3">Moteur d'inférence</h3>
              <div className="grid grid-cols-3 gap-4">
                {[
                  {
                    label: 'Statut',
                    content: (
                      <div className="flex items-center gap-2 mt-1">
                        <span className={`w-2 h-2 rounded-full ${stats.engine_loaded ? 'bg-emerald-500' : 'bg-zinc-300 dark:bg-zinc-600'}`} />
                        <span className="text-sm font-medium">{stats.engine_loaded ? 'Chargé' : 'Non chargé'}</span>
                      </div>
                    ),
                    icon: Zap,
                  },
                  {
                    label: 'Mots actifs',
                    content: <p className="text-2xl font-bold font-mono tabular-nums mt-1">{stats.engine_words}</p>,
                    icon: Activity,
                  },
                  {
                    label: 'Dernier build',
                    content: stats.last_build.built_at
                      ? <p className="text-sm font-medium mt-1">{new Date(stats.last_build.built_at).toLocaleDateString('fr-FR')}</p>
                      : <p className="text-sm text-zinc-400 mt-1">Jamais</p>,
                    icon: RefreshCw,
                  },
                ].map(({ label, content, icon: Icon }) => (
                  <div key={label} className="bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg p-4">
                    <div className="flex items-center gap-1.5 text-zinc-400 mb-0.5">
                      <Icon className="w-3.5 h-3.5" />
                      <span className="text-[10px] uppercase tracking-widest font-medium">{label}</span>
                    </div>
                    {content}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Contributions */}
        {tab === 'contributions' && (
          <div className="space-y-4 animate-fade-in max-w-3xl">
            {/* Toolbar */}
            <div className="flex flex-wrap items-center gap-2">
              <div className="flex items-center gap-0.5 bg-zinc-100 dark:bg-zinc-800 rounded-md p-0.5">
                {(['all', 'pending', 'approved', 'rejected'] as ContribFilter[]).map(f => (
                  <button
                    key={f}
                    onClick={() => { setContribFilter(f); setSelectedContribs(new Set()); }}
                    className={`px-2.5 py-1 text-xs rounded transition-colors cursor-pointer ${
                      contribFilter === f
                        ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-zinc-50 shadow-sm font-medium'
                        : 'text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'
                    }`}
                  >
                    {f === 'all' ? 'Tous' : f === 'pending' ? 'Attente' : f === 'approved' ? 'OK' : 'Rejetés'}
                    {f !== 'all' && (
                      <span className="ml-1 opacity-50">({contributions.filter(c => c.status === f).length})</span>
                    )}
                  </button>
                ))}
              </div>

              <div className="flex-1" />

              {contributions.some(c => c.status === 'rejected') && (
                <button
                  onClick={async () => {
                    const rej = contributions.filter(c => c.status === 'rejected');
                    try {
                      await Promise.all(rej.map(c => api.deleteContribution(c.id)));
                      addToast('success', `${rej.length} supprimées`); refresh();
                    } catch { addToast('error', 'Erreur'); }
                  }}
                  className={btnSecondary}
                >
                  Purger rejetées
                </button>
              )}

              {filteredContribs.some(c => c.status === 'pending') && (
                <button
                  onClick={() => {
                    const ids = filteredContribs.filter(c => c.status === 'pending').map(c => c.id);
                    const allSel = ids.every(id => selectedContribs.has(id));
                    setSelectedContribs(allSel ? new Set() : new Set(ids));
                  }}
                  className={btnSecondary}
                >
                  {filteredContribs.filter(c => c.status === 'pending').every(c => selectedContribs.has(c.id)) && selectedContribs.size > 0
                    ? 'Désélectionner' : 'Tout sélectionner'}
                </button>
              )}

              {selectedContribs.size > 0 && (
                <>
                  <span className="text-xs font-mono text-zinc-400">{selectedContribs.size} sél.</span>
                  <button onClick={() => handleBatchReview('approved')} className={btnPrimary}>Approuver</button>
                  <button onClick={() => handleBatchReview('rejected')} className={btnSecondary}>Rejeter</button>
                </>
              )}
            </div>

            {/* List */}
            {filteredContribs.length === 0 ? (
              <p className="text-sm text-zinc-400 py-10 text-center">Aucune contribution</p>
            ) : (
              <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl overflow-hidden divide-y divide-zinc-100 dark:divide-zinc-800">
                {filteredContribs.map(c => (
                  <div
                    key={c.id}
                    className={`px-4 py-3 flex items-center gap-3 text-sm ${
                      selectedContribs.has(c.id) ? 'bg-blue-50 dark:bg-blue-950/20' : 'hover:bg-zinc-50 dark:hover:bg-zinc-800/40'
                    } transition-colors`}
                  >
                    {c.status === 'pending' && (
                      <button
                        onClick={() => toggleContribSelection(c.id)}
                        className={`w-4 h-4 border-2 rounded shrink-0 cursor-pointer flex items-center justify-center transition-colors ${
                          selectedContribs.has(c.id)
                            ? 'bg-blue-600 border-blue-600'
                            : 'border-zinc-300 dark:border-zinc-600 hover:border-blue-400'
                        }`}
                      >
                        {selectedContribs.has(c.id) && (
                          <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                          </svg>
                        )}
                      </button>
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-medium text-zinc-900 dark:text-zinc-50">{c.word}</span>
                        <span className="text-zinc-400 text-xs">par {c.contributor}</span>
                        <span className={`text-[10px] font-medium uppercase tracking-wide px-1.5 py-0.5 rounded ${
                          c.status === 'approved'
                            ? 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-400'
                            : c.status === 'pending'
                            ? 'bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-400'
                            : 'bg-red-50 dark:bg-red-950/40 text-red-600 dark:text-red-400'
                        }`}>
                          {c.status}
                        </span>
                      </div>
                      {c.recorded_at && (
                        <p className="text-xs font-mono text-zinc-400 mt-0.5">
                          {new Date(c.recorded_at).toLocaleString('fr-FR')}
                        </p>
                      )}
                    </div>
                    <div className="flex gap-0.5 shrink-0">
                      <button onClick={() => openPreview(c)} className={iconBtn} title="Visualiser">
                        <Play className="w-3.5 h-3.5" />
                      </button>
                      {c.status === 'pending' && (
                        <>
                          <button onClick={() => handleReview(c.id, 'approved')} className={iconBtn} title="Approuver">
                            <CheckCircle className="w-3.5 h-3.5 text-emerald-500" />
                          </button>
                          <button onClick={() => handleReview(c.id, 'rejected')} className={iconBtn} title="Rejeter">
                            <XCircle className="w-3.5 h-3.5 text-red-500" />
                          </button>
                        </>
                      )}
                      {c.status === 'rejected' && (
                        <button
                          onClick={async () => {
                            try { await api.deleteContribution(c.id); addToast('success', 'Supprimée'); refresh(); }
                            catch { addToast('error', 'Erreur'); }
                          }}
                          className={iconBtn}
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Modèle (training) */}
        {tab === 'index' && (
          <div className="max-w-xl animate-fade-in space-y-4">
            <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-5">
              <div className="flex items-center gap-2 mb-1">
                <Cpu className="w-4 h-4 text-zinc-400" />
                <span className="text-[10px] uppercase tracking-widest font-medium text-zinc-400">BiLSTM + attention temporelle</span>
              </div>
              <p className="text-sm text-zinc-500 mb-1 mt-2">
                Entraîne le réseau sur tous les templates disponibles (30 frames × 171 points).
                Le modèle est rechargé automatiquement à la fin.
              </p>
              {stats && (
                <p className="text-xs font-mono text-zinc-400">
                  {stats.templates} templates · {stats.words} classes
                </p>
              )}

              {(building || buildProgress === 100) && (
                <div className="mt-4">
                  <div className="flex justify-between text-xs font-mono text-zinc-500 mb-1.5">
                    <span>{building ? `Epoch ${trainLogs.length}…` : 'Terminé ✓'}</span>
                    <span className="font-medium text-blue-600 dark:text-blue-400">{buildProgress}%</span>
                  </div>
                  <div className="h-1.5 bg-zinc-100 dark:bg-zinc-800 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-500 transition-all duration-500 rounded-full" style={{ width: `${buildProgress}%` }} />
                  </div>
                </div>
              )}

              <button
                onClick={handleTrain}
                disabled={building}
                className={`mt-4 w-full py-2.5 text-sm font-medium flex items-center justify-center gap-2 rounded-md transition-colors cursor-pointer ${
                  building
                    ? 'bg-zinc-100 dark:bg-zinc-800 text-zinc-400 cursor-not-allowed'
                    : btnPrimary
                }`}
              >
                <RefreshCw className={`w-4 h-4 ${building ? 'animate-spin-slow' : ''}`} />
                {building ? 'Entraînement en cours…' : "Lancer l'entraînement"}
              </button>
            </div>

            {(trainLogs.length > 0 || trainDone) && (
              <div className="border border-zinc-200 dark:border-zinc-800 rounded-xl overflow-hidden">
                <div className="px-4 py-2.5 border-b border-zinc-200 dark:border-zinc-800 flex items-center gap-2 bg-zinc-50 dark:bg-zinc-900">
                  <Activity className="w-3.5 h-3.5 text-zinc-400" />
                  <span className="text-[10px] uppercase tracking-widest font-medium text-zinc-400">Logs</span>
                </div>
                <div
                  ref={trainLogRef}
                  className="h-56 overflow-y-auto p-4 font-mono text-xs bg-zinc-950 text-zinc-300 space-y-0.5"
                >
                  {trainLogs.map((log, i) => (
                    <div key={i} className="flex gap-3 leading-5">
                      <span className="text-zinc-600 w-16 shrink-0 tabular-nums">{String(log.epoch).padStart(3,' ')}/{log.total}</span>
                      <span className="text-emerald-400">acc {(log.acc*100).toFixed(1)}%</span>
                      <span className="text-blue-400">val {(log.val_acc*100).toFixed(1)}%</span>
                      <span className="text-zinc-600">loss {log.loss.toFixed(4)}</span>
                    </div>
                  ))}
                  {trainDone && (
                    <div className="pt-2 border-t border-zinc-800 mt-2 text-emerald-400 font-semibold">
                      ✓ {trainDone.meta.n_classes} classes · val_acc {(trainDone.meta.best_val_accuracy*100).toFixed(1)}%
                    </div>
                  )}
                </div>
              </div>
            )}

            {stats?.last_build.built_at && !building && trainLogs.length === 0 && (
              <div className="bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl px-4 py-3.5">
                <p className="text-[10px] uppercase tracking-widest font-medium text-zinc-400 mb-1">Dernier entraînement</p>
                <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">{new Date(stats.last_build.built_at).toLocaleString('fr-FR')}</p>
                <p className="text-xs font-mono text-zinc-400 mt-0.5">
                  {stats.last_build.n_templates} tpl · {(stats.last_build.duration_ms / 1000).toFixed(0)}s · {stats.last_build.status}
                </p>
              </div>
            )}
          </div>
        )}

        {/* Team */}
        {tab === 'team' && (
          <div className="space-y-6 animate-fade-in max-w-3xl">
            {/* Members */}
            <div>
              <h3 className="text-xs uppercase tracking-widest font-medium text-zinc-400 mb-3">Membres ({members.length})</h3>
              {members.length === 0 ? (
                <p className="text-sm text-zinc-400">Aucun membre — ils rejoignent via la page Équipe.</p>
              ) : (
                <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl overflow-hidden divide-y divide-zinc-100 dark:divide-zinc-800">
                  {members.map((m: any) => (
                    <div key={m.id} className="px-4 py-3 flex items-center justify-between text-sm">
                      <span className="font-medium text-zinc-900 dark:text-zinc-50">{m.name}</span>
                      <div className="flex items-center gap-3 text-xs font-mono text-zinc-400">
                        <span className="text-blue-600 dark:text-blue-400 font-medium">{m.tasks_pending} en attente</span>
                        <span>{m.tasks_done} faits</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Assign words */}
            <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-5">
              <h3 className="text-xs uppercase tracking-widest font-medium text-zinc-400 mb-4">Assigner des mots</h3>

              {selectedWords.size > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-4">
                  {Array.from(selectedWords).map(w => (
                    <span key={w} className="flex items-center gap-1 text-xs bg-blue-50 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800 px-2 py-0.5 rounded-md">
                      {w}
                      <button
                        onClick={() => setSelectedWords(prev => { const n = new Set(prev); n.delete(w); return n; })}
                        className="text-blue-400 hover:text-blue-600 cursor-pointer ml-0.5"
                      >
                        <X className="w-2.5 h-2.5" />
                      </button>
                    </span>
                  ))}
                  <button
                    onClick={() => setSelectedWords(new Set())}
                    className="text-xs text-zinc-400 hover:text-zinc-600 cursor-pointer"
                  >
                    Tout retirer
                  </button>
                </div>
              )}

              <div className="flex gap-2 mb-4">
                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={wordSearch}
                    onChange={e => { setWordSearch(e.target.value); setShowWordPicker(true); }}
                    onFocus={() => setShowWordPicker(true)}
                    placeholder="Rechercher dans le dataset SL…"
                    className={`${inputCls} w-full`}
                  />
                  {showWordPicker && (
                    <>
                      <div className="fixed inset-0 z-30" onClick={() => setShowWordPicker(false)} />
                      <div className="absolute left-0 right-0 top-full mt-1 z-40 max-h-56 overflow-auto bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-lg shadow-lg">
                        {wordSearch.trim() && !datasetWords.includes(wordSearch.trim()) && (
                          <button
                            onClick={() => { setSelectedWords(p => new Set(p).add(wordSearch.trim())); setWordSearch(''); setShowWordPicker(false); }}
                            className="w-full px-3 py-2.5 text-left text-sm hover:bg-zinc-50 dark:hover:bg-zinc-800 cursor-pointer border-b border-zinc-100 dark:border-zinc-800 flex items-center gap-2 text-blue-600 dark:text-blue-400"
                          >
                            <Plus className="w-3.5 h-3.5" />
                            Ajouter "<strong>{wordSearch.trim()}</strong>" (nouveau)
                          </button>
                        )}
                        {datasetWords
                          .filter(w => !wordSearch.trim() || w.toLowerCase().includes(wordSearch.toLowerCase()))
                          .map(w => {
                            const sel = selectedWords.has(w);
                            const assigned = assignments.some((a: any) => a.word === w && a.assigned_date === assignDate);
                            return (
                              <button
                                key={w}
                                onClick={() => { if (!sel && !assigned) setSelectedWords(p => new Set(p).add(w)); }}
                                disabled={sel || assigned}
                                className={`w-full px-3 py-2 text-left text-sm flex items-center justify-between transition-colors ${
                                  sel || assigned ? 'opacity-40 cursor-not-allowed' : 'hover:bg-zinc-50 dark:hover:bg-zinc-800 cursor-pointer'
                                }`}
                              >
                                <span className="text-zinc-900 dark:text-zinc-50">{w}</span>
                                {(sel || assigned) && <span className="text-xs text-zinc-400">{sel ? 'sélectionné' : 'assigné'}</span>}
                              </button>
                            );
                          })}
                      </div>
                    </>
                  )}
                </div>
                <input
                  type="date"
                  value={assignDate}
                  onChange={e => setAssignDate(e.target.value)}
                  className={inputCls}
                />
              </div>

              <button
                onClick={async () => {
                  if (!selectedWords.size) return;
                  try {
                    const r = await api.assignWords(Array.from(selectedWords), assignDate);
                    addToast('success', `${r.words_assigned} mots assignés`);
                    setSelectedWords(new Set()); setWordSearch(''); refresh();
                  } catch { addToast('error', 'Erreur'); }
                }}
                disabled={selectedWords.size === 0}
                className={`${btnPrimary} disabled:opacity-40 disabled:cursor-not-allowed`}
              >
                Assigner {selectedWords.size || ''} mot{selectedWords.size > 1 ? 's' : ''}
              </button>
            </div>

            {/* Assignments */}
            {assignments.length > 0 && (
              <div>
                <h3 className="text-xs uppercase tracking-widest font-medium text-zinc-400 mb-3">Assignations</h3>
                <div className="space-y-3">
                  {assignments.map((a: any) => (
                    <div key={a.id} className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl overflow-hidden">
                      <div className="px-4 py-3 flex items-center gap-3 border-b border-zinc-100 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/50">
                        <span className="font-semibold text-sm text-zinc-900 dark:text-zinc-50">{a.word}</span>
                        <span className="flex items-center gap-1 text-xs text-zinc-400">
                          <Calendar className="w-3 h-3" />{a.assigned_date}
                        </span>
                        <span className="text-xs font-mono text-zinc-400">{a.progress.done}/{a.progress.total_members}</span>
                        <div className="flex-1" />
                        <button
                          onClick={async () => {
                            try { await api.deleteAssignment(a.id); addToast('success', 'Supprimée'); refresh(); }
                            catch { addToast('error', 'Erreur'); }
                          }}
                          className={iconBtn}
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                      <div className="divide-y divide-zinc-100 dark:divide-zinc-800">
                        {a.members.map((m: any) => (
                          <div key={m.task_id} className="px-4 py-2.5 flex items-center gap-3 text-sm">
                            <span className="w-28 font-medium text-zinc-900 dark:text-zinc-50">{m.member_name}</span>
                            <span className={`text-[10px] font-medium uppercase tracking-wide px-1.5 py-0.5 rounded ${
                              m.status === 'done'
                                ? 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-400'
                                : m.status === 'rejected'
                                ? 'bg-red-50 dark:bg-red-950/40 text-red-600 dark:text-red-400'
                                : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500'
                            }`}>
                              {m.status === 'done' ? 'fait' : m.status === 'rejected' ? 'rejeté' : 'attente'}
                            </span>
                            <span className="text-xs font-mono text-zinc-400 tabular-nums">{m.templates_recorded}/{a.templates_required}</span>
                            <div className="flex-1" />
                            {m.status === 'done' && (
                              <button
                                onClick={async () => {
                                  try { await api.rejectTask(m.task_id, 'À refaire'); addToast('info', `Rejeté pour ${m.member_name}`); refresh(); }
                                  catch { addToast('error', 'Erreur'); }
                                }}
                                className="flex items-center gap-1 text-xs text-zinc-400 hover:text-red-500 transition-colors cursor-pointer"
                              >
                                <AlertTriangle className="w-3 h-3" /> Rejeter
                              </button>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Words */}
        {tab === 'words' && (
          <div className="space-y-4 animate-fade-in max-w-2xl">
            {/* Add word */}
            <div className="flex gap-2">
              <input
                type="text"
                value={newWordName}
                onChange={e => setNewWordName(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleCreateWord()}
                placeholder="Nouveau mot…"
                className={`${inputCls} flex-1`}
              />
              <button onClick={handleCreateWord} className={`${btnPrimary} flex items-center gap-1.5`}>
                <Plus className="w-4 h-4" /> Ajouter
              </button>
            </div>

            {/* Word list */}
            <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl overflow-hidden divide-y divide-zinc-100 dark:divide-zinc-800">
              {words.map(w => (
                <div key={w.name} className="px-4 py-3 flex items-center gap-3 hover:bg-zinc-50 dark:hover:bg-zinc-800/40 transition-colors">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-medium text-zinc-900 dark:text-zinc-50">{w.name}</span>
                      <span className="text-xs font-mono text-zinc-400 tabular-nums">{w.template_count} tpl</span>
                      {!w.is_active && (
                        <span className="text-[10px] uppercase tracking-wide bg-zinc-100 dark:bg-zinc-800 text-zinc-500 px-1.5 py-0.5 rounded">inactif</span>
                      )}
                    </div>
                    {editingWord === w.name ? (
                      <div className="flex items-center gap-1.5 mt-2">
                        <input
                          type="text"
                          value={editDescription}
                          onChange={e => setEditDescription(e.target.value)}
                          placeholder="Description…"
                          className="flex-1 px-2.5 py-1 text-xs border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-zinc-900 dark:text-zinc-50"
                          autoFocus
                        />
                        <button onClick={() => handleSaveDescription(w.name)} className={iconBtn}>
                          <Save className="w-3.5 h-3.5 text-emerald-500" />
                        </button>
                        <button onClick={() => setEditingWord(null)} className={iconBtn}>
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ) : w.description ? (
                      <p className="text-xs text-zinc-400 mt-0.5 truncate">{w.description}</p>
                    ) : null}
                  </div>

                  <div className="flex items-center gap-1.5 shrink-0">
                    {/* Toggle switch */}
                    <button
                      onClick={() => handleToggleActive(w)}
                      className={`w-9 h-5 flex items-center px-0.5 rounded-full transition-colors cursor-pointer ${
                        w.is_active ? 'bg-blue-600' : 'bg-zinc-200 dark:bg-zinc-700'
                      }`}
                    >
                      <div className={`w-4 h-4 bg-white rounded-full shadow-sm transition-transform ${w.is_active ? 'translate-x-4' : 'translate-x-0'}`} />
                    </button>
                    <button
                      onClick={() => { setEditingWord(w.name); setEditDescription(w.description || ''); }}
                      className={iconBtn}
                    >
                      <Edit3 className="w-3.5 h-3.5" />
                    </button>
                    <button onClick={() => handleDeleteWord(w.name)} className={iconBtn}>
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Skeleton preview modal */}
      {previewId && previewContrib && (
        <>
          <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm" onClick={closePreview} />
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none">
            <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-xl shadow-xl w-full max-w-md pointer-events-auto animate-scale-in">
              <div className="px-5 py-4 border-b border-zinc-100 dark:border-zinc-800 flex items-center justify-between">
                <div>
                  <span className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">{previewContrib.word}</span>
                  <span className="ml-2 text-xs text-zinc-400">par {previewContrib.contributor}</span>
                </div>
                <button onClick={closePreview} className={iconBtn}>
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="p-5">
                <SkeletonPlayer contributionId={previewId} />
              </div>
              {previewContrib.status === 'pending' && (
                <div className="px-5 pb-5 flex gap-2 justify-end">
                  <button
                    onClick={() => { closePreview(); handleReview(previewId, 'approved'); }}
                    className={`${btnPrimary} flex items-center gap-1.5`}
                  >
                    <CheckCircle className="w-3.5 h-3.5" /> Approuver
                  </button>
                  <button
                    onClick={() => { closePreview(); handleReview(previewId, 'rejected'); }}
                    className={`${btnSecondary} flex items-center gap-1.5`}
                  >
                    <XCircle className="w-3.5 h-3.5" /> Rejeter
                  </button>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function AnimatedStat({
  label, value, icon: Icon, color, iconBg,
}: { label: string; value: number; icon: React.ElementType; color: string; iconBg: string }) {
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    const dur = 700;
    const start = performance.now();
    const animate = (now: number) => {
      const t = Math.min((now - start) / dur, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplay(Math.round(eased * value));
      if (t < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }, [value]);

  return (
    <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-5">
      <div className={`inline-flex p-2 rounded-lg ${iconBg} mb-3`}>
        <Icon className={`w-4 h-4 ${color}`} />
      </div>
      <p className="text-3xl font-bold font-mono tabular-nums text-zinc-900 dark:text-zinc-50">{display}</p>
      <p className="text-xs text-zinc-400 mt-1 font-medium">{label}</p>
    </div>
  );
}
