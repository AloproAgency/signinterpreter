import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '../lib/api';
import { useApp } from '../lib/context';
import {
  Database, CheckCircle, XCircle, Clock,
  RefreshCw, Trash2, Plus, Lock, Activity,
  Download, Edit3, Save, X,
  Cpu, HardDrive, Zap, Eye, BarChart3, Play, Users, Calendar, AlertTriangle,
} from 'lucide-react';
import type { StatsInfo, ContributionInfo, WordInfo } from '../lib/types';
import { SkeletonPlayer } from '../components/SkeletonPlayer';

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
  const [newWordName, setNewWordName] = useState('');
  const [tab, setTab] = useState<'dashboard' | 'contributions' | 'index' | 'words' | 'team'>('dashboard');
  const [contribFilter, setContribFilter] = useState<'all' | 'pending' | 'approved' | 'rejected'>('pending');
  const [selectedContribs, setSelectedContribs] = useState<Set<number>>(new Set());
  const [editingWord, setEditingWord] = useState<string | null>(null);
  const [editDescription, setEditDescription] = useState('');

  // Team management
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
      addToast('success', 'Connexion reussie');
    } catch {
      setLoginError('Mot de passe incorrect');
    }
  };

  const refresh = useCallback(async () => {
    try {
      const [s, c, w, a, m] = await Promise.all([
        api.getStats(), api.getContributions(), api.getVocabulary(),
        api.getAssignments().catch(() => []),
        api.getMembers().catch(() => []),
      ]);
      setStats(s);
      setContributions(c);
      setWords(w);
      setAssignments(a);
      setMembers(m);
      if (datasetWords.length === 0) {
        api.getDatasetWords().then(setDatasetWords).catch(() => {});
      }
    } catch {
      addToast('error', 'Erreur de chargement');
    }
  }, [addToast]);

  useEffect(() => { if (logged) refresh(); }, [logged, refresh]);

  const handleBuildIndex = async () => {
    setBuilding(true);
    setBuildProgress(0);
    const interval = setInterval(() => {
      setBuildProgress(p => Math.min(p + Math.random() * 15, 90));
    }, 300);
    try {
      const result = await api.buildIndex();
      clearInterval(interval);
      setBuildProgress(100);
      addToast('success', `Index construit: ${result.n_templates} templates, ${result.n_words} mots`);
      refresh();
    } catch {
      clearInterval(interval);
      addToast('error', 'Erreur lors de la construction');
    }
    setTimeout(() => { setBuilding(false); setBuildProgress(0); }, 1000);
  };

  const handleReview = async (id: number, status: 'approved' | 'rejected') => {
    try {
      await api.reviewContribution(id, status);
      addToast('success', status === 'approved' ? 'Contribution approuvee' : 'Contribution rejetee');
      refresh();
    } catch {
      addToast('error', 'Erreur');
    }
  };

  const handleBatchReview = async (status: 'approved' | 'rejected') => {
    const ids = Array.from(selectedContribs);
    if (ids.length === 0) return;
    try {
      await Promise.all(ids.map(id => api.reviewContribution(id, status)));
      addToast('success', `${ids.length} contributions ${status === 'approved' ? 'approuvees' : 'rejetees'}`);
      setSelectedContribs(new Set());
      refresh();
    } catch {
      addToast('error', 'Erreur batch');
    }
  };

  const handleCreateWord = async () => {
    if (!newWordName.trim()) return;
    try {
      await api.createWord(newWordName.trim());
      addToast('success', `Mot "${newWordName.trim()}" cree`);
      setNewWordName('');
      refresh();
    } catch {
      addToast('error', 'Erreur de creation');
    }
  };

  const handleDeleteWord = async (name: string) => {
    if (!confirm(`Supprimer "${name}" et tous ses templates ?`)) return;
    try {
      await api.deleteWord(name);
      addToast('success', `"${name}" supprime`);
      refresh();
    } catch {
      addToast('error', 'Erreur de suppression');
    }
  };

  const handleSaveDescription = async (name: string) => {
    try {
      await api.updateWord(name, { description: editDescription });
      setEditingWord(null);
      addToast('success', 'Description mise a jour');
      refresh();
    } catch {
      addToast('error', 'Erreur');
    }
  };

  const handleToggleActive = async (word: WordInfo) => {
    try {
      await api.updateWord(word.name, { is_active: !word.is_active });
      addToast('info', `${word.name} ${!word.is_active ? 'active' : 'desactive'}`);
      refresh();
    } catch {
      addToast('error', 'Erreur');
    }
  };

  const toggleContribSelection = (id: number) => {
    setSelectedContribs(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const filteredContribs = contributions.filter(
    c => contribFilter === 'all' || c.status === contribFilter
  );

  const handleExport = () => {
    const data = JSON.stringify({ words, contributions }, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `signinterpreter-export-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    addToast('success', 'Donnees exportees');
  };

  // Preview contribution video
  const openPreview = (contrib: ContributionInfo) => {
    setPreviewId(contrib.id);
    setPreviewContrib(contrib);
  };

  const closePreview = () => {
    setPreviewId(null);
    setPreviewContrib(null);
  };

  // Login screen
  if (!logged) {
    return (
      <div className="h-full flex items-center justify-center p-4 bg-white dark:bg-[#0d1117]">
        <div className="w-full max-w-xs animate-scale-in">
          <div className="rounded-lg border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22] p-6 shadow-sm dark:shadow-none">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-[rgba(19,150,186,0.1)] border border-[rgba(19,150,186,0.15)]">
                <Lock className="w-5 h-5 text-[#1396ba]" />
              </div>
              <div>
                <h2 className="text-base font-bold tracking-tight text-[#1f2328] dark:text-[#e6edf3]">
                  Administration
                </h2>
                <p className="text-sm text-[#8b949e] dark:text-[#484f58]">Acces restreint</p>
              </div>
            </div>

            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && login()}
              placeholder="Mot de passe"
              className="w-full rounded-md px-3 py-2 text-sm mb-3 bg-white dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] text-[#1f2328] dark:text-[#e6edf3] placeholder-[#8b949e] dark:placeholder-[#484f58] focus:outline-none focus:border-[#1396ba] focus:ring-1 focus:ring-[#1396ba]/30"
            />
            {loginError && (
              <p className="text-sm mb-3 flex items-center gap-1.5 text-[#ef4444]">
                <XCircle className="w-3.5 h-3.5" />{loginError}
              </p>
            )}
            <button
              onClick={login}
              className="w-full py-2 rounded-md font-medium text-sm bg-[#1396ba] hover:bg-[#17b8e3] text-white cursor-pointer transition-colors"
            >
              Connexion
            </button>
          </div>
        </div>
      </div>
    );
  }

  const tabs = [
    { key: 'dashboard' as const, label: 'Dashboard', icon: BarChart3 },
    { key: 'team' as const, label: 'Equipe', icon: Users },
    { key: 'contributions' as const, label: 'Contributions', icon: Eye },
    { key: 'index' as const, label: 'Index', icon: Database },
    { key: 'words' as const, label: 'Mots', icon: BookIcon },
  ];

  return (
    <div className="h-full p-4 md:p-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6 max-w-5xl">
        <div>
          <h1 className="text-3xl font-bold tracking-tight mb-1 text-[#1f2328] dark:text-[#e6edf3]">
            Administration
          </h1>
          <p className="text-sm text-[#656d76] dark:text-[#8b949e]">Gestion du systeme</p>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={handleExport}
            className="p-2 rounded-md transition-colors text-[#656d76] dark:text-[#8b949e] hover:text-[#1396ba] hover:bg-[rgba(19,150,186,0.1)]"
            title="Exporter"
          >
            <Download className="w-4 h-4" />
          </button>
          <button
            onClick={refresh}
            className="p-2 rounded-md transition-colors text-[#656d76] dark:text-[#8b949e] hover:text-[#1396ba] hover:bg-[rgba(19,150,186,0.1)]"
            title="Rafraichir"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Tabs - underline style */}
      <div className="flex gap-0 mb-6 border-b border-[#d0d7de] dark:border-[#30363d] max-w-5xl">
        {tabs.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition-colors relative cursor-pointer ${
              tab === key
                ? 'text-[#1396ba]'
                : 'text-[#656d76] dark:text-[#8b949e] hover:text-[#1f2328] dark:hover:text-[#e6edf3]'
            }`}
          >
            <Icon className="w-4 h-4" />
            <span className="hidden sm:inline">{label}</span>
            {tab === key && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[#1396ba] rounded-full" />
            )}
          </button>
        ))}
      </div>

      {/* Dashboard Tab */}
      {tab === 'dashboard' && stats && (
        <div className="space-y-6 animate-fade-in max-w-5xl">
          {/* Stat cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <StatCard label="Mots" value={stats.words} icon={<Database className="w-4 h-4" />} color="#1396ba" />
            <StatCard label="Templates" value={stats.templates} icon={<HardDrive className="w-4 h-4" />} color="#17b8e3" />
            <StatCard label="En attente" value={stats.contributions.pending} icon={<Clock className="w-4 h-4" />} color="#f59e0b" />
            <StatCard label="Approuves" value={stats.contributions.approved} icon={<CheckCircle className="w-4 h-4" />} color="#10b981" />
          </div>

          {/* Engine status */}
          <div>
            <h3 className="text-sm font-semibold mb-3 uppercase tracking-wider flex items-center gap-2 text-[#656d76] dark:text-[#8b949e]">
              <Cpu className="w-3.5 h-3.5 text-[#1396ba]" />
              Moteur d'inference
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="rounded-lg p-4 border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22]">
                <div className="flex items-center gap-2 mb-2">
                  <div className={`w-2 h-2 rounded-full ${
                    stats.engine_loaded ? 'bg-[#10b981]' : 'bg-[#8b949e]'
                  }`} />
                  <span className="text-sm uppercase tracking-wider font-medium text-[#656d76] dark:text-[#8b949e]">
                    Statut
                  </span>
                </div>
                <p className={`text-sm font-bold ${
                  stats.engine_loaded ? 'text-[#10b981]' : 'text-[#8b949e]'
                }`}>
                  {stats.engine_loaded ? 'Charge' : 'Non charge'}
                </p>
              </div>

              <div className="rounded-lg p-4 border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22]">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="w-3 h-3 text-[#1396ba]" />
                  <span className="text-sm uppercase tracking-wider font-medium text-[#656d76] dark:text-[#8b949e]">
                    Mots actifs
                  </span>
                </div>
                <p className="text-sm font-bold font-mono text-[#1396ba]">
                  {stats.engine_words}
                </p>
              </div>

              <div className="rounded-lg p-4 border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22]">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-3 h-3 text-[#1396ba]" />
                  <span className="text-sm uppercase tracking-wider font-medium text-[#656d76] dark:text-[#8b949e]">
                    Dernier build
                  </span>
                </div>
                {stats.last_build.built_at ? (
                  <div>
                    <p className="text-sm font-bold text-[#1f2328] dark:text-[#e6edf3]">
                      {new Date(stats.last_build.built_at).toLocaleDateString('fr-FR')}
                    </p>
                    <p className="text-sm font-mono mt-0.5 text-[#8b949e] dark:text-[#484f58]">
                      {stats.last_build.n_templates} tpl | {stats.last_build.duration_ms}ms
                    </p>
                  </div>
                ) : (
                  <p className="text-sm text-[#8b949e]">Jamais</p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Contributions Tab */}
      {tab === 'contributions' && (
        <div className="space-y-4 animate-fade-in max-w-4xl">
          {/* Filter & batch actions */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-center gap-1">
              {(['all', 'pending', 'approved', 'rejected'] as const).map(f => (
                <button
                  key={f}
                  onClick={() => { setContribFilter(f); setSelectedContribs(new Set()); }}
                  className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                    contribFilter === f
                      ? 'bg-[#1396ba] text-white'
                      : 'text-[#656d76] dark:text-[#8b949e] hover:bg-[#f6f8fa] dark:hover:bg-[#161b22]'
                  }`}
                >
                  {f === 'all' ? 'Tous' : f === 'pending' ? 'Attente' : f === 'approved' ? 'OK' : 'Rejetes'}
                  {f !== 'all' && (
                    <span className="ml-1 opacity-60">
                      ({contributions.filter(c => c.status === f).length})
                    </span>
                  )}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-2">
              {/* Purge rejected */}
              {contributions.some(c => c.status === 'rejected') && (
                <button
                  onClick={async () => {
                    const rejected = contributions.filter(c => c.status === 'rejected');
                    try {
                      await Promise.all(rejected.map(c => api.deleteContribution(c.id)));
                      addToast('success', `${rejected.length} contributions rejetees supprimees`);
                      refresh();
                    } catch { addToast('error', 'Erreur'); }
                  }}
                  className="px-3 py-1.5 rounded-md text-sm font-medium border border-[#ef4444]/30 text-[#ef4444] hover:bg-[#ef4444]/10 cursor-pointer"
                >
                  Purger rejetees ({contributions.filter(c => c.status === 'rejected').length})
                </button>
              )}

              {/* Select all / deselect */}
              {filteredContribs.some(c => c.status === 'pending') && (
                <>
                  <button
                    onClick={() => {
                      const pendingIds = filteredContribs.filter(c => c.status === 'pending').map(c => c.id);
                      setSelectedContribs(prev => {
                        const allSelected = pendingIds.every(id => prev.has(id));
                        return allSelected ? new Set() : new Set(pendingIds);
                      });
                    }}
                    className="px-3 py-1.5 rounded-md text-sm font-medium border border-[#d0d7de] dark:border-[#30363d] text-[#656d76] dark:text-[#8b949e] hover:bg-[#f6f8fa] dark:hover:bg-[#161b22] cursor-pointer"
                  >
                    {filteredContribs.filter(c => c.status === 'pending').every(c => selectedContribs.has(c.id)) && selectedContribs.size > 0
                      ? 'Deselectionner tout'
                      : 'Tout selectionner'}
                  </button>
                </>
              )}

              {selectedContribs.size > 0 && (
                <>
                  <span className="text-sm font-mono font-bold text-[#1396ba]">
                    {selectedContribs.size} sel.
                  </span>
                  <button
                    onClick={() => handleBatchReview('approved')}
                    className="px-3 py-1.5 rounded-md text-sm font-medium transition-colors bg-[#10b981]/10 text-[#10b981] hover:bg-[#10b981]/20 cursor-pointer"
                  >
                    Approuver
                  </button>
                  <button
                    onClick={() => handleBatchReview('rejected')}
                    className="px-3 py-1.5 rounded-md text-sm font-medium transition-colors bg-[#ef4444]/10 text-[#ef4444] hover:bg-[#ef4444]/20 cursor-pointer"
                  >
                    Rejeter
                  </button>
                </>
              )}
            </div>
          </div>

          {/* List */}
          {filteredContribs.length === 0 ? (
            <div className="text-center py-16">
              <div className="w-14 h-14 rounded-lg flex items-center justify-center mx-auto mb-3 bg-[rgba(19,150,186,0.1)] border border-[rgba(19,150,186,0.15)]">
                <Eye className="w-6 h-6 text-[#1396ba]/50" />
              </div>
              <p className="text-sm text-[#8b949e] dark:text-[#484f58]">Aucune contribution</p>
            </div>
          ) : (
            <div className="border border-[#d0d7de] dark:border-[#30363d] rounded-lg overflow-hidden">
              {filteredContribs.map((c, i) => (
                <div
                  key={c.id}
                  className={`px-4 py-3 flex items-center gap-3 ${
                    i !== filteredContribs.length - 1 ? 'border-b border-[#d0d7de] dark:border-[#30363d]' : ''
                  } ${
                    selectedContribs.has(c.id)
                      ? 'bg-[rgba(19,150,186,0.05)]'
                      : 'bg-[#f6f8fa] dark:bg-[#161b22]'
                  }`}
                >
                  {c.status === 'pending' && (
                    <button
                      onClick={() => toggleContribSelection(c.id)}
                      className={`w-4 h-4 rounded border-2 transition-colors flex items-center justify-center shrink-0 cursor-pointer ${
                        selectedContribs.has(c.id)
                          ? 'border-[#1396ba] bg-[#1396ba]'
                          : 'border-[#d0d7de] dark:border-[#30363d] hover:border-[#1396ba]'
                      }`}
                    >
                      {selectedContribs.has(c.id) && (
                        <CheckCircle className="w-3 h-3 text-white" />
                      )}
                    </button>
                  )}

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-semibold text-sm text-[#1f2328] dark:text-[#e6edf3]">{c.word}</span>
                      <span className="text-sm text-[#656d76] dark:text-[#8b949e]">par {c.contributor}</span>
                      <span className={`text-sm font-bold uppercase tracking-wider px-2 py-0.5 rounded-md ${
                        c.status === 'pending'
                          ? 'bg-[#f59e0b]/10 text-[#f59e0b]'
                          : c.status === 'approved'
                          ? 'bg-[#10b981]/10 text-[#10b981]'
                          : 'bg-[#ef4444]/10 text-[#ef4444]'
                      }`}>
                        {c.status}
                      </span>
                    </div>
                    {c.recorded_at && (
                      <p className="text-sm mt-0.5 font-mono text-[#8b949e] dark:text-[#484f58]">
                        {new Date(c.recorded_at).toLocaleString('fr-FR')}
                      </p>
                    )}
                  </div>

                  <div className="flex gap-1 shrink-0">
                    <button
                      onClick={() => openPreview(c)}
                      className="p-1.5 rounded-md transition-colors bg-[rgba(19,150,186,0.1)] hover:bg-[rgba(19,150,186,0.2)] text-[#1396ba] cursor-pointer"
                      title="Visualiser"
                    >
                      <Play className="w-3.5 h-3.5" />
                    </button>
                    {c.status === 'pending' && (
                      <>
                        <button
                          onClick={() => handleReview(c.id, 'approved')}
                          className="p-1.5 rounded-md transition-colors bg-[#10b981]/10 hover:bg-[#10b981]/20 text-[#10b981] cursor-pointer"
                          title="Approuver"
                        >
                          <CheckCircle className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => handleReview(c.id, 'rejected')}
                          className="p-1.5 rounded-md transition-colors bg-[#ef4444]/10 hover:bg-[#ef4444]/20 text-[#ef4444] cursor-pointer"
                          title="Rejeter"
                        >
                          <XCircle className="w-3.5 h-3.5" />
                        </button>
                      </>
                    )}
                    {c.status === 'rejected' && (
                      <button
                        onClick={async () => {
                          try {
                            await api.deleteContribution(c.id);
                            addToast('success', 'Contribution supprimee');
                            refresh();
                          } catch { addToast('error', 'Erreur'); }
                        }}
                        className="p-1.5 rounded-md transition-colors bg-[#ef4444]/10 hover:bg-[#ef4444]/20 text-[#ef4444] cursor-pointer"
                        title="Supprimer"
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

      {/* Video Preview Modal */}
      {previewId && previewContrib && (
        <>
          <div className="fixed inset-0 z-50 bg-black/50" onClick={closePreview} />
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none">
            <div className="bg-white dark:bg-[#161b22] rounded-lg border border-[#d0d7de] dark:border-[#30363d] shadow-xl w-full max-w-md pointer-events-auto animate-scale-in">
              <div className="px-4 py-3 border-b border-[#d0d7de] dark:border-[#30363d] flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-[#1f2328] dark:text-[#e6edf3]">{previewContrib.word}</h3>
                  <p className="text-sm text-[#8b949e]">par {previewContrib.contributor}</p>
                </div>
                <button onClick={closePreview} className="p-1.5 rounded-md text-[#8b949e] hover:text-[#1f2328] dark:hover:text-white hover:bg-[#f6f8fa] dark:hover:bg-[#0d1117] cursor-pointer">
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Skeleton replay */}
              <div className="p-4">
                <SkeletonPlayer contributionId={previewId} />
              </div>

              {/* Actions */}
              {previewContrib.status === 'pending' && (
                <div className="px-4 pb-4 flex justify-end gap-2">
                  <button
                    onClick={() => { closePreview(); handleReview(previewId, 'approved'); }}
                    className="px-4 py-2 rounded-md text-sm font-medium bg-[#10b981] text-white hover:bg-[#059669] cursor-pointer flex items-center gap-1.5"
                  >
                    <CheckCircle className="w-4 h-4" />
                    Approuver
                  </button>
                  <button
                    onClick={() => { closePreview(); handleReview(previewId, 'rejected'); }}
                    className="px-4 py-2 rounded-md text-sm font-medium bg-[#ef4444] text-white hover:bg-[#dc2626] cursor-pointer flex items-center gap-1.5"
                  >
                    <XCircle className="w-4 h-4" />
                    Rejeter
                  </button>
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {/* Index Tab */}
      {tab === 'index' && (
        <div className="max-w-md animate-fade-in">
          <div className="rounded-lg p-5 border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22]">
            <h3 className="text-sm font-semibold mb-1 uppercase tracking-wider flex items-center gap-2 text-[#656d76] dark:text-[#8b949e]">
              <Database className="w-3.5 h-3.5 text-[#1396ba]" />
              Index FAISS
            </h3>
            <p className="text-sm mb-5 text-[#8b949e] dark:text-[#484f58]">
              Reconstruisez l'index apres avoir approuve de nouvelles contributions.
            </p>

            {building && (
              <div className="mb-5">
                <div className="flex justify-between text-sm mb-1.5">
                  <span className="font-medium text-[#656d76] dark:text-[#8b949e]">Construction...</span>
                  <span className="font-mono font-bold text-[#1396ba]">{Math.round(buildProgress)}%</span>
                </div>
                <div className="h-1.5 rounded-full overflow-hidden bg-[#d0d7de] dark:bg-[#30363d]">
                  <div
                    className="h-full rounded-full bg-[#1396ba] transition-all duration-300"
                    style={{ width: `${buildProgress}%` }}
                  />
                </div>
              </div>
            )}

            <button
              onClick={handleBuildIndex}
              disabled={building}
              className={`w-full py-2.5 rounded-md font-medium text-sm flex items-center justify-center gap-2 transition-colors cursor-pointer ${
                building
                  ? 'bg-[#f6f8fa] dark:bg-[#0d1117] text-[#8b949e] border border-[#d0d7de] dark:border-[#30363d]'
                  : 'bg-[#1396ba] hover:bg-[#17b8e3] text-white'
              }`}
            >
              <RefreshCw className={`w-4 h-4 ${building ? 'animate-spin-slow' : ''}`} />
              {building ? 'Construction...' : 'Reconstruire l\'index'}
            </button>

            {stats && stats.last_build.built_at && (
              <div className="mt-4 rounded-md p-3 border border-[#d0d7de] dark:border-[#30363d] bg-white dark:bg-[#0d1117]">
                <p className="text-sm uppercase tracking-wider mb-1 font-medium text-[#656d76] dark:text-[#8b949e]">
                  Dernier build
                </p>
                <p className="text-sm font-bold text-[#1f2328] dark:text-[#e6edf3]">
                  {new Date(stats.last_build.built_at).toLocaleString('fr-FR')}
                </p>
                <p className="text-sm mt-0.5 font-mono text-[#8b949e] dark:text-[#484f58]">
                  {stats.last_build.n_templates} templates | {stats.last_build.duration_ms}ms | {stats.last_build.status}
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Team Tab */}
      {tab === 'team' && (
        <div className="space-y-6 animate-fade-in max-w-5xl">
          {/* Members */}
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider mb-3 text-[#656d76] dark:text-[#8b949e]">
              Membres ({members.length})
            </h3>
            {members.length === 0 ? (
              <p className="text-sm text-[#8b949e]">Aucun membre. Les membres rejoignent via la page "Mon equipe".</p>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {members.map((m: any) => (
                  <div key={m.id} className="rounded-lg p-3 border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22]">
                    <p className="text-sm font-bold text-[#1f2328] dark:text-[#e6edf3]">{m.name}</p>
                    <p className="text-sm text-[#8b949e]">{m.tasks_done} faits, {m.tasks_pending} en attente</p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Assign words */}
          <div className="rounded-lg p-5 border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22]">
            <h3 className="text-sm font-semibold uppercase tracking-wider mb-3 text-[#656d76] dark:text-[#8b949e]">
              Assigner des mots du jour
            </h3>

            {/* Selected words pills */}
            {selectedWords.size > 0 && (
              <div className="flex flex-wrap gap-2 mb-3">
                {Array.from(selectedWords).map(w => (
                  <span key={w} className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-sm bg-[rgba(19,150,186,0.1)] text-[#1396ba] border border-[rgba(19,150,186,0.2)]">
                    {w}
                    <button
                      onClick={() => setSelectedWords(prev => { const n = new Set(prev); n.delete(w); return n; })}
                      className="hover:text-[#ef4444] cursor-pointer ml-1"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </span>
                ))}
                <button
                  onClick={() => setSelectedWords(new Set())}
                  className="text-sm text-[#8b949e] hover:text-[#ef4444] cursor-pointer"
                >
                  Tout retirer
                </button>
              </div>
            )}

            {/* Word picker */}
            <div className="flex flex-col sm:flex-row gap-3 mb-3">
              <div className="flex-1 relative">
                <input
                  type="text"
                  value={wordSearch}
                  onChange={e => { setWordSearch(e.target.value); setShowWordPicker(true); }}
                  onFocus={() => setShowWordPicker(true)}
                  placeholder="Rechercher un mot dans le dataset SL..."
                  className="w-full rounded-md px-3 py-2 text-sm bg-white dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] text-[#1f2328] dark:text-[#e6edf3] placeholder-[#8b949e] focus:outline-none focus:border-[#1396ba] focus:ring-1 focus:ring-[#1396ba]/30"
                />

                {/* Dropdown */}
                {showWordPicker && (
                  <>
                    <div className="fixed inset-0 z-30" onClick={() => setShowWordPicker(false)} />
                    <div className="absolute left-0 right-0 top-full mt-1 z-40 max-h-64 overflow-auto rounded-lg border border-[#d0d7de] dark:border-[#30363d] bg-white dark:bg-[#161b22] shadow-lg">
                      {/* Quick add custom word */}
                      {wordSearch.trim() && !datasetWords.includes(wordSearch.trim()) && (
                        <button
                          onClick={() => {
                            setSelectedWords(prev => new Set(prev).add(wordSearch.trim()));
                            setWordSearch('');
                            setShowWordPicker(false);
                          }}
                          className="w-full px-3 py-2 text-left text-sm hover:bg-[#f6f8fa] dark:hover:bg-[#0d1117] cursor-pointer border-b border-[#d0d7de] dark:border-[#30363d] flex items-center gap-2"
                        >
                          <Plus className="w-3.5 h-3.5 text-[#1396ba]" />
                          <span>Ajouter "<strong>{wordSearch.trim()}</strong>" (nouveau mot)</span>
                        </button>
                      )}

                      {/* Dataset words filtered */}
                      {datasetWords
                        .filter(w => !wordSearch.trim() || w.toLowerCase().includes(wordSearch.toLowerCase()))
                        .map(w => {
                          const alreadySelected = selectedWords.has(w);
                          const alreadyAssigned = assignments.some(
                            (a: any) => a.word === w && a.assigned_date === assignDate
                          );
                          return (
                            <button
                              key={w}
                              onClick={() => {
                                if (!alreadySelected && !alreadyAssigned) {
                                  setSelectedWords(prev => new Set(prev).add(w));
                                }
                              }}
                              disabled={alreadySelected || alreadyAssigned}
                              className={`w-full px-3 py-2 text-left text-sm flex items-center justify-between cursor-pointer ${
                                alreadySelected || alreadyAssigned
                                  ? 'opacity-40 cursor-not-allowed'
                                  : 'hover:bg-[#f6f8fa] dark:hover:bg-[#0d1117]'
                              }`}
                            >
                              <span className="text-[#1f2328] dark:text-[#e6edf3]">{w}</span>
                              {alreadySelected && <span className="text-sm text-[#1396ba]">selectionne</span>}
                              {alreadyAssigned && <span className="text-sm text-[#8b949e]">deja assigne</span>}
                            </button>
                          );
                        })}

                      {datasetWords.filter(w => !wordSearch.trim() || w.toLowerCase().includes(wordSearch.toLowerCase())).length === 0 && (
                        <p className="px-3 py-4 text-sm text-[#8b949e] text-center">Aucun mot trouve</p>
                      )}
                    </div>
                  </>
                )}
              </div>

              <input
                type="date"
                value={assignDate}
                onChange={e => setAssignDate(e.target.value)}
                className="rounded-md px-3 py-2 text-sm bg-white dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] text-[#1f2328] dark:text-[#e6edf3]"
              />
            </div>

            <button
              onClick={async () => {
                if (selectedWords.size === 0) return;
                try {
                  const result = await api.assignWords(Array.from(selectedWords), assignDate);
                  addToast('success', `${result.words_assigned} mots assignes pour le ${assignDate}`);
                  setSelectedWords(new Set());
                  setWordSearch('');
                  refresh();
                } catch { addToast('error', 'Erreur'); }
              }}
              disabled={selectedWords.size === 0}
              className="px-4 py-2 rounded-md text-sm font-medium bg-[#1396ba] hover:bg-[#17b8e3] text-white disabled:opacity-40 cursor-pointer"
            >
              Assigner {selectedWords.size} mot{selectedWords.size > 1 ? 's' : ''}
            </button>
          </div>

          {/* Assignments list */}
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider mb-3 text-[#656d76] dark:text-[#8b949e]">
              Assignations
            </h3>
            {assignments.length === 0 ? (
              <p className="text-sm text-[#8b949e]">Aucune assignation</p>
            ) : (
              <div className="space-y-4">
                {assignments.map((a: any) => (
                  <div key={a.id} className="rounded-lg border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22] overflow-hidden">
                    <div className="px-4 py-3 flex items-center justify-between border-b border-[#d0d7de] dark:border-[#30363d]">
                      <div className="flex items-center gap-3">
                        <span className="text-base font-bold text-[#1f2328] dark:text-[#e6edf3]">{a.word}</span>
                        <span className="text-sm text-[#8b949e] flex items-center gap-1">
                          <Calendar className="w-3 h-3" />
                          {a.assigned_date}
                        </span>
                        <span className="text-sm font-mono text-[#1396ba]">
                          {a.progress.done}/{a.progress.total_members}
                        </span>
                      </div>
                      <button
                        onClick={async () => {
                          try {
                            await api.deleteAssignment(a.id);
                            addToast('success', 'Assignation supprimee');
                            refresh();
                          } catch { addToast('error', 'Erreur'); }
                        }}
                        className="p-1.5 rounded-md text-[#8b949e] hover:text-[#ef4444] hover:bg-[#ef4444]/10 cursor-pointer"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                    {/* Member progress */}
                    <div className="divide-y divide-[#d0d7de] dark:divide-[#30363d]">
                      {a.members.map((m: any) => (
                        <div key={m.task_id} className="px-4 py-2 flex items-center gap-3">
                          <span className="text-sm text-[#1f2328] dark:text-[#e6edf3] w-28">{m.member_name}</span>
                          <span className={`text-sm font-bold uppercase tracking-wider px-2 py-0.5 rounded-md ${
                            m.status === 'done' ? 'bg-[#10b981]/10 text-[#10b981]'
                            : m.status === 'rejected' ? 'bg-[#ef4444]/10 text-[#ef4444]'
                            : 'bg-[#f59e0b]/10 text-[#f59e0b]'
                          }`}>
                            {m.status === 'done' ? 'Fait' : m.status === 'rejected' ? 'Rejete' : 'En attente'}
                          </span>
                          <span className="text-sm text-[#8b949e] font-mono">{m.templates_recorded}/{a.templates_required}</span>
                          <div className="flex-1" />
                          {m.status === 'done' && (
                            <button
                              onClick={async () => {
                                try {
                                  await api.rejectTask(m.task_id, 'A refaire');
                                  addToast('info', `Tache rejetee pour ${m.member_name}`);
                                  refresh();
                                } catch { addToast('error', 'Erreur'); }
                              }}
                              className="px-2 py-1 rounded-md text-sm text-[#ef4444] hover:bg-[#ef4444]/10 cursor-pointer flex items-center gap-1"
                            >
                              <AlertTriangle className="w-3 h-3" />
                              Rejeter
                            </button>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Words Tab */}
      {tab === 'words' && (
        <div className="space-y-4 animate-fade-in max-w-3xl">
          {/* Add word */}
          <div className="flex gap-2 max-w-sm">
            <input
              type="text"
              value={newWordName}
              onChange={e => setNewWordName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleCreateWord()}
              placeholder="Nouveau mot..."
              className="flex-1 rounded-md px-3 py-2 text-sm bg-white dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] text-[#1f2328] dark:text-[#e6edf3] placeholder-[#8b949e] dark:placeholder-[#484f58] focus:outline-none focus:border-[#1396ba] focus:ring-1 focus:ring-[#1396ba]/30"
            />
            <button
              onClick={handleCreateWord}
              className="px-3 py-2 rounded-md bg-[#1396ba] hover:bg-[#17b8e3] text-white transition-colors cursor-pointer"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>

          {/* Word list */}
          <div className="border border-[#d0d7de] dark:border-[#30363d] rounded-lg overflow-hidden">
            {words.map((w, i) => (
              <div key={w.name} className={`px-4 py-3 flex items-center gap-3 ${
                i !== words.length - 1 ? 'border-b border-[#d0d7de] dark:border-[#30363d]' : ''
              } bg-[#f6f8fa] dark:bg-[#161b22]`}>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-sm text-[#1f2328] dark:text-[#e6edf3]">{w.name}</span>
                    <span className="text-sm font-mono font-bold text-[#1396ba]">
                      {w.template_count} tpl
                    </span>
                    {!w.is_active && (
                      <span className="text-sm font-bold uppercase tracking-wider px-2 py-0.5 rounded-md bg-[#484f58]/15 text-[#8b949e]">
                        Inactif
                      </span>
                    )}
                  </div>

                  {editingWord === w.name ? (
                    <div className="flex items-center gap-1.5 mt-2">
                      <input
                        type="text"
                        value={editDescription}
                        onChange={e => setEditDescription(e.target.value)}
                        placeholder="Description..."
                        className="flex-1 rounded-md px-3 py-1.5 text-sm bg-white dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] text-[#1f2328] dark:text-[#e6edf3] focus:outline-none focus:border-[#1396ba] focus:ring-1 focus:ring-[#1396ba]/30"
                        autoFocus
                      />
                      <button
                        onClick={() => handleSaveDescription(w.name)}
                        className="p-1.5 rounded-md bg-[#10b981]/10 text-[#10b981] hover:bg-[#10b981]/20 cursor-pointer transition-colors"
                      >
                        <Save className="w-3 h-3" />
                      </button>
                      <button
                        onClick={() => setEditingWord(null)}
                        className="p-1.5 rounded-md text-[#8b949e] hover:bg-[#f6f8fa] dark:hover:bg-[#0d1117] cursor-pointer transition-colors"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  ) : (
                    w.description && (
                      <p className="text-sm mt-0.5 text-[#8b949e] dark:text-[#484f58]">{w.description}</p>
                    )
                  )}
                </div>

                <div className="flex items-center gap-1.5 shrink-0">
                  {/* Toggle active */}
                  <button
                    onClick={() => handleToggleActive(w)}
                    className={`w-9 h-5 rounded-full transition-colors flex items-center px-0.5 cursor-pointer ${
                      w.is_active ? 'bg-[#1396ba]' : 'bg-[#d0d7de] dark:bg-[#30363d]'
                    }`}
                  >
                    <div className={`w-4 h-4 rounded-full bg-white shadow-sm transition-transform ${
                      w.is_active ? 'translate-x-4' : 'translate-x-0'
                    }`} />
                  </button>

                  {/* Edit */}
                  <button
                    onClick={() => {
                      setEditingWord(w.name);
                      setEditDescription(w.description || '');
                    }}
                    className="p-1.5 rounded-md transition-colors text-[#656d76] dark:text-[#8b949e] hover:text-[#1396ba] hover:bg-[rgba(19,150,186,0.1)] cursor-pointer"
                  >
                    <Edit3 className="w-3 h-3" />
                  </button>

                  {/* Delete */}
                  <button
                    onClick={() => handleDeleteWord(w.name)}
                    className="p-1.5 rounded-md transition-colors text-[#656d76] dark:text-[#8b949e] hover:text-[#ef4444] hover:bg-[#ef4444]/10 cursor-pointer"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function BookIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
    </svg>
  );
}

function StatCard({
  label, value, icon, color,
}: {
  label: string;
  value: number;
  icon: React.ReactNode;
  color: string;
}) {
  const [displayValue, setDisplayValue] = useState(0);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const duration = 600;
    const startTime = performance.now();

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayValue(Math.round(eased * value));
      if (progress < 1) requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  }, [value]);

  return (
    <div
      ref={ref}
      className="rounded-lg p-4 border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22] shadow-sm dark:shadow-none"
    >
      <div className="flex items-center gap-2 mb-3">
        <div
          className="w-8 h-8 rounded-md flex items-center justify-center"
          style={{
            background: `${color}18`,
            color: color,
          }}
        >
          {icon}
        </div>
        <span className="text-sm font-medium text-[#656d76] dark:text-[#8b949e]">
          {label}
        </span>
      </div>
      <p
        className="text-3xl font-bold tracking-tight font-mono"
        style={{ color }}
      >
        {displayValue}
      </p>
    </div>
  );
}
