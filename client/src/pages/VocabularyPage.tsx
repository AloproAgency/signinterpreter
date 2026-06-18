import { useState, useEffect, useMemo } from 'react';
import { api } from '../lib/api';
import { Search, X, SortAsc, SortDesc } from 'lucide-react';
import type { WordInfo, ContributionInfo } from '../lib/types';

type SortField = 'name' | 'template_count';
type SortDir   = 'asc' | 'desc';
type Filter    = 'all' | 'needs_more' | 'active' | 'inactive';

export default function VocabularyPage() {
  const [words, setWords]       = useState<WordInfo[]>([]);
  const [search, setSearch]     = useState('');
  const [loading, setLoading]   = useState(true);
  const [sortField, setSortField] = useState<SortField>('name');
  const [sortDir, setSortDir]   = useState<SortDir>('asc');
  const [filter, setFilter]     = useState<Filter>('all');
  const [selected, setSelected] = useState<WordInfo | null>(null);
  const [detail, setDetail]     = useState<ContributionInfo[]>([]);
  const [loadingDetail, setLoadingDetail] = useState(false);

  useEffect(() => {
    api.getVocabulary()
      .then(w => { setWords(w); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    let r = words.filter(w => w.name.toLowerCase().includes(search.toLowerCase()));
    if (filter === 'needs_more') r = r.filter(w => w.template_count < 10);
    else if (filter === 'active')   r = r.filter(w => w.is_active);
    else if (filter === 'inactive') r = r.filter(w => !w.is_active);
    r.sort((a, b) => {
      const cmp = sortField === 'name'
        ? a.name.localeCompare(b.name)
        : a.template_count - b.template_count;
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return r;
  }, [words, search, filter, sortField, sortDir]);

  const stats = useMemo(() => ({
    total:     words.length,
    needsMore: words.filter(w => w.template_count < 10).length,
    templates: words.reduce((s, w) => s + w.template_count, 0),
  }), [words]);

  const openDetail = async (w: WordInfo) => {
    setSelected(w); setLoadingDetail(true);
    try {
      const c = await api.getContributions();
      setDetail(c.filter((x: ContributionInfo) => x.word === w.name));
    } catch { setDetail([]); }
    setLoadingDetail(false);
  };

  const toggleSort = (f: SortField) => {
    if (sortField === f) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortField(f); setSortDir('asc'); }
  };

  const uniqueContributors = useMemo(
    () => new Set(detail.map(c => c.contributor)).size,
    [detail],
  );

  const FILTERS: [Filter, string][] = [
    ['all', 'Tous'], ['needs_more', '< 10'], ['active', 'Actifs'], ['inactive', 'Inactifs'],
  ];

  return (
    <div className="h-full flex flex-col bg-white dark:bg-zinc-950">

      {/* ── Controls bar ─────────────────────────────── */}
      <div className="shrink-0 h-12 px-4 flex items-center gap-3 border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950">
        <span className="text-sm font-semibold shrink-0">Vocabulaire</span>

        {/* Stats */}
        <div className="text-xs font-mono text-zinc-400 shrink-0 hidden sm:flex items-center gap-2">
          <span>{stats.total} mots</span>
          <span className="text-zinc-300 dark:text-zinc-700">·</span>
          <span>{stats.templates} templates</span>
          {stats.needsMore > 0 && (
            <>
              <span className="text-zinc-300 dark:text-zinc-700">·</span>
              <span className="text-amber-600 dark:text-amber-400 font-medium">{stats.needsMore} incomplets</span>
            </>
          )}
        </div>

        <div className="w-px h-5 bg-zinc-200 dark:bg-zinc-800" />

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-zinc-400" />
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Rechercher…"
            className="pl-8 pr-7 py-1.5 text-sm bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-700 rounded-md placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 w-40"
          />
          {search && (
            <button onClick={() => setSearch('')} className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300 cursor-pointer">
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Filters */}
        <div className="flex items-center gap-0.5 bg-zinc-100 dark:bg-zinc-800 rounded-md p-0.5">
          {FILTERS.map(([f, label]) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-2.5 py-1 text-xs rounded transition-colors cursor-pointer ${
                filter === f
                  ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-zinc-50 shadow-sm font-medium'
                  : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="w-px h-5 bg-zinc-200 dark:bg-zinc-800" />

        {/* Sort */}
        <div className="flex items-center gap-0.5">
          {([['name', 'A–Z'], ['template_count', 'Count']] as [SortField, string][]).map(([f, label]) => (
            <button
              key={f}
              onClick={() => toggleSort(f)}
              className={`flex items-center gap-1 px-2.5 py-1 text-xs rounded-md transition-colors cursor-pointer ${
                sortField === f
                  ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/40 font-medium'
                  : 'text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800'
              }`}
            >
              {label}
              {sortField === f && (sortDir === 'asc' ? <SortAsc className="w-3 h-3" /> : <SortDesc className="w-3 h-3" />)}
            </button>
          ))}
        </div>
      </div>

      {/* ── Word table ───────────────────────────────── */}
      <div className="flex-1 min-h-0 overflow-auto">
        {loading ? (
          <div className="p-6 space-y-2">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="h-10 animate-pulse bg-zinc-100 dark:bg-zinc-800/50 rounded-md" />
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-sm text-zinc-400">Aucun mot trouvé</p>
          </div>
        ) : (
          <table className="w-full text-sm border-collapse">
            <thead className="sticky top-0 bg-zinc-50 dark:bg-zinc-900 border-b border-zinc-200 dark:border-zinc-800">
              <tr>
                <th className="text-left px-4 py-2.5 text-[10px] uppercase tracking-widest font-medium text-zinc-400">Mot</th>
                <th className="text-left px-4 py-2.5 text-[10px] uppercase tracking-widest font-medium text-zinc-400 hidden sm:table-cell">Description</th>
                <th className="text-right px-4 py-2.5 text-[10px] uppercase tracking-widest font-medium text-zinc-400">Templates</th>
                <th className="text-center px-4 py-2.5 text-[10px] uppercase tracking-widest font-medium text-zinc-400 hidden md:table-cell">Couverture</th>
                <th className="text-center px-4 py-2.5 text-[10px] uppercase tracking-widest font-medium text-zinc-400">Statut</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60">
              {filtered.map(word => (
                <tr
                  key={word.name}
                  onClick={() => openDetail(word)}
                  className="hover:bg-zinc-50 dark:hover:bg-zinc-900/40 cursor-pointer transition-colors"
                >
                  <td className="px-4 py-3 font-medium text-zinc-900 dark:text-zinc-50">{word.name}</td>
                  <td className="px-4 py-3 text-zinc-400 hidden sm:table-cell truncate max-w-xs text-sm">
                    {word.description || <span className="text-zinc-300 dark:text-zinc-700">—</span>}
                  </td>
                  <td className="px-4 py-3 text-right font-mono tabular-nums">
                    <span className={word.template_count < 10 ? 'text-amber-600 dark:text-amber-400 font-medium' : 'text-zinc-700 dark:text-zinc-300'}>
                      {word.template_count}
                    </span>
                  </td>
                  <td className="px-4 py-3 hidden md:table-cell">
                    <div className="w-24 mx-auto h-1.5 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${word.template_count < 10 ? 'bg-amber-400' : 'bg-blue-500'}`}
                        style={{ width: `${Math.min(100, (word.template_count / 20) * 100)}%` }}
                      />
                    </div>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium uppercase tracking-wide ${
                      word.is_active
                        ? 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-400'
                        : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500'
                    }`}>
                      {word.is_active ? 'actif' : 'inactif'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* ── Detail modal ─────────────────────────────── */}
      {selected && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm"
          onClick={() => setSelected(null)}
        >
          <div
            className="w-full max-w-md bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-xl shadow-xl animate-scale-in max-h-[80vh] overflow-auto"
            onClick={e => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-start justify-between px-5 py-4 border-b border-zinc-100 dark:border-zinc-800">
              <div>
                <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">{selected.name}</h2>
                {selected.description && (
                  <p className="text-sm text-zinc-500 mt-0.5">{selected.description}</p>
                )}
              </div>
              <button
                onClick={() => setSelected(null)}
                className="p-1.5 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-md transition-colors cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 divide-x divide-zinc-100 dark:divide-zinc-800 border-b border-zinc-100 dark:border-zinc-800">
              {[
                { label: 'Templates', value: selected.template_count },
                { label: 'Contributeurs', value: uniqueContributors },
                { label: 'Statut', value: selected.is_active ? 'Actif' : 'Inactif' },
              ].map(({ label, value }) => (
                <div key={label} className="px-4 py-4 text-center">
                  <p className="text-2xl font-bold font-mono tabular-nums text-zinc-900 dark:text-zinc-50">{value}</p>
                  <p className="text-[10px] uppercase tracking-widest text-zinc-400 mt-1">{label}</p>
                </div>
              ))}
            </div>

            {/* Coverage */}
            <div className="px-5 py-4 border-b border-zinc-100 dark:border-zinc-800">
              <div className="flex justify-between text-xs font-mono text-zinc-500 mb-2">
                <span>Couverture</span>
                <span className="font-medium">{selected.template_count}/20</span>
              </div>
              <div className="h-2 bg-zinc-100 dark:bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full transition-all"
                  style={{ width: `${Math.min(100, (selected.template_count / 20) * 100)}%` }}
                />
              </div>
            </div>

            {/* Contributions */}
            <div>
              <div className="px-5 py-3 border-b border-zinc-100 dark:border-zinc-800">
                <span className="text-[10px] uppercase tracking-widest font-medium text-zinc-400">Contributions</span>
              </div>
              {loadingDetail ? (
                <div className="p-4 space-y-2">
                  {[1,2,3].map(i => <div key={i} className="h-9 animate-pulse bg-zinc-100 dark:bg-zinc-800 rounded-md" />)}
                </div>
              ) : detail.length === 0 ? (
                <p className="px-5 py-5 text-sm text-zinc-400">Aucune contribution</p>
              ) : (
                <div className="divide-y divide-zinc-100 dark:divide-zinc-800 max-h-48 overflow-auto">
                  {detail.map(c => (
                    <div key={c.id} className="px-5 py-2.5 flex items-center justify-between">
                      <div>
                        <span className="text-sm font-medium text-zinc-900 dark:text-zinc-50">{c.contributor}</span>
                        {c.recorded_at && (
                          <span className="ml-2 text-xs font-mono text-zinc-400">
                            {new Date(c.recorded_at).toLocaleDateString('fr-FR')}
                          </span>
                        )}
                      </div>
                      <span className={`text-[10px] font-medium uppercase tracking-wide px-2 py-0.5 rounded ${
                        c.status === 'approved'
                          ? 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-400'
                          : c.status === 'pending'
                          ? 'bg-blue-50 dark:bg-blue-950/40 text-blue-600 dark:text-blue-400'
                          : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500'
                      }`}>
                        {c.status}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
