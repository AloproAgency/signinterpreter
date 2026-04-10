import { useState, useEffect, useMemo } from 'react';
import { api } from '../lib/api';
import {
  Search, X, AlertTriangle,
  SortAsc, SortDesc, Users, Hash,
} from 'lucide-react';
import type { WordInfo, ContributionInfo } from '../lib/types';

type SortField = 'name' | 'template_count';
type SortDir = 'asc' | 'desc';
type FilterMode = 'all' | 'needs_more' | 'active' | 'inactive';

export default function VocabularyPage() {
  const [words, setWords] = useState<WordInfo[]>([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [sortField, setSortField] = useState<SortField>('name');
  const [sortDir, setSortDir] = useState<SortDir>('asc');
  const [filterMode, setFilterMode] = useState<FilterMode>('all');
  const [selectedWord, setSelectedWord] = useState<WordInfo | null>(null);
  const [wordContributions, setWordContributions] = useState<ContributionInfo[]>([]);
  const [loadingDetail, setLoadingDetail] = useState(false);

  useEffect(() => {
    api.getVocabulary()
      .then(w => { setWords(w); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    let result = words.filter(w =>
      w.name.toLowerCase().includes(search.toLowerCase())
    );

    if (filterMode === 'needs_more') result = result.filter(w => w.template_count < 10);
    else if (filterMode === 'active') result = result.filter(w => w.is_active);
    else if (filterMode === 'inactive') result = result.filter(w => !w.is_active);

    result.sort((a, b) => {
      let cmp = 0;
      if (sortField === 'name') cmp = a.name.localeCompare(b.name);
      else cmp = a.template_count - b.template_count;
      return sortDir === 'asc' ? cmp : -cmp;
    });

    return result;
  }, [words, search, filterMode, sortField, sortDir]);

  const openDetail = async (word: WordInfo) => {
    setSelectedWord(word);
    setLoadingDetail(true);
    try {
      const contributions = await api.getContributions();
      setWordContributions(
        contributions.filter((c: ContributionInfo) => c.word === word.name)
      );
    } catch {
      setWordContributions([]);
    }
    setLoadingDetail(false);
  };

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDir('asc');
    }
  };

  const stats = useMemo(() => ({
    total: words.length,
    needsMore: words.filter(w => w.template_count < 10).length,
    totalTemplates: words.reduce((s, w) => s + w.template_count, 0),
  }), [words]);

  const uniqueContributors = useMemo(() => {
    const set = new Set(wordContributions.map(c => c.contributor));
    return set.size;
  }, [wordContributions]);

  return (
    <div className="h-full p-4 md:p-6 overflow-auto">
      {/* Header */}
      <div className="mb-6 max-w-6xl">
        <h1 className="text-3xl font-bold tracking-tight mb-1 text-[#1f2328] dark:text-[#e6edf3]">
          Vocabulaire
        </h1>
        <div className="flex items-center gap-3 flex-wrap">
          <p className="text-sm text-[#656d76] dark:text-[#8b949e]">
            <span className="font-mono font-bold text-[#1396ba]">{stats.total}</span> mots,{' '}
            <span className="font-mono font-bold text-[#1396ba]">{stats.totalTemplates}</span> templates
          </p>
          {stats.needsMore > 0 && (
            <span className="text-sm font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-[#f59e0b]/10 text-[#f59e0b]">
              {stats.needsMore} ont besoin de plus
            </span>
          )}
        </div>
      </div>

      {/* Controls bar */}
      <div className="flex flex-col sm:flex-row gap-3 mb-6 max-w-6xl">
        {/* Search */}
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[#8b949e] dark:text-[#484f58]" />
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Rechercher un mot..."
            className="w-full rounded-md pl-9 pr-3 py-2 text-sm bg-white dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] text-[#1f2328] dark:text-[#e6edf3] placeholder-[#8b949e] dark:placeholder-[#484f58] focus:outline-none focus:border-[#1396ba] focus:ring-1 focus:ring-[#1396ba]/30"
          />
          {search && (
            <button onClick={() => setSearch('')} className="absolute right-3 top-1/2 -translate-y-1/2 text-[#8b949e] hover:text-[#1396ba] transition-colors">
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Filter pills */}
        <div className="flex items-center gap-1">
          {([
            ['all', 'Tous'],
            ['needs_more', '< 10'],
            ['active', 'Actifs'],
            ['inactive', 'Inactifs'],
          ] as const).map(([mode, label]) => (
            <button
              key={mode}
              onClick={() => setFilterMode(mode)}
              className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                filterMode === mode
                  ? 'bg-[#1396ba] text-white'
                  : 'text-[#656d76] dark:text-[#8b949e] hover:bg-[#f6f8fa] dark:hover:bg-[#161b22]'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Sort */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => toggleSort('name')}
            className={`flex items-center gap-1 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
              sortField === 'name'
                ? 'text-[#1396ba]'
                : 'text-[#8b949e] dark:text-[#484f58] hover:text-[#656d76] dark:hover:text-[#8b949e]'
            }`}
          >
            A-Z {sortField === 'name' && (sortDir === 'asc' ? <SortAsc className="w-3 h-3" /> : <SortDesc className="w-3 h-3" />)}
          </button>
          <button
            onClick={() => toggleSort('template_count')}
            className={`flex items-center gap-1 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
              sortField === 'template_count'
                ? 'text-[#1396ba]'
                : 'text-[#8b949e] dark:text-[#484f58] hover:text-[#656d76] dark:hover:text-[#8b949e]'
            }`}
          >
            Count {sortField === 'template_count' && (sortDir === 'asc' ? <SortAsc className="w-3 h-3" /> : <SortDesc className="w-3 h-3" />)}
          </button>
        </div>
      </div>

      {/* Grid */}
      {loading ? (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
          {Array.from({ length: 12 }).map((_, i) => (
            <div key={i} className="rounded-lg p-4 h-24 animate-pulse border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22]" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="text-center py-20">
          <div className="w-14 h-14 rounded-lg flex items-center justify-center mx-auto mb-3 bg-[rgba(19,150,186,0.1)] border border-[rgba(19,150,186,0.15)]">
            <Search className="w-6 h-6 text-[#1396ba]/50" />
          </div>
          <p className="text-sm text-[#8b949e] dark:text-[#484f58]">Aucun mot trouve</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3 max-w-6xl">
          {filtered.map(word => (
            <button
              key={word.name}
              onClick={() => openDetail(word)}
              className="rounded-lg p-4 text-left border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22] hover:bg-[#f0f2f5] dark:hover:bg-[#1c2333] hover:border-[rgba(19,150,186,0.15)] transition-colors cursor-pointer"
            >
              <div className="flex items-start justify-between mb-2">
                <p className="text-sm font-semibold truncate flex-1 text-[#1f2328] dark:text-[#e6edf3]">
                  {word.name}
                </p>
                {!word.is_active && (
                  <span className="text-sm font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-md shrink-0 ml-2 bg-[#484f58]/15 text-[#8b949e]">
                    Off
                  </span>
                )}
              </div>

              {word.description && (
                <p className="text-sm mb-2 line-clamp-1 text-[#8b949e] dark:text-[#484f58]">
                  {word.description}
                </p>
              )}

              <div className="flex items-center justify-between mt-auto">
                <div className="flex items-center gap-1.5">
                  <Hash className="w-2.5 h-2.5 text-[#1396ba]/50" />
                  <span className={`text-sm font-mono font-bold ${
                    word.template_count < 10 ? 'text-[#f59e0b]' : 'text-[#1396ba]'
                  }`}>
                    {word.template_count}
                  </span>
                </div>

                {word.template_count < 10 && (
                  <AlertTriangle className="w-3 h-3 text-[#f59e0b]" />
                )}

                {/* Coverage bar */}
                <div className="w-16 h-1.5 rounded-full overflow-hidden bg-[#d0d7de] dark:bg-[#30363d]">
                  <div
                    className="h-full rounded-full bg-[#1396ba]"
                    style={{ width: `${Math.min(100, (word.template_count / 20) * 100)}%` }}
                  />
                </div>
              </div>
            </button>
          ))}
        </div>
      )}

      {/* Detail Modal */}
      {selectedWord && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50"
          onClick={() => setSelectedWord(null)}
        >
          <div
            className="w-full max-w-md rounded-lg p-6 shadow-xl animate-scale-in max-h-[80vh] overflow-auto border border-[#d0d7de] dark:border-[#30363d] bg-white dark:bg-[#161b22]"
            onClick={e => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-start justify-between mb-5">
              <div>
                <h2 className="text-2xl font-bold tracking-tight text-[#1f2328] dark:text-[#e6edf3]">
                  {selectedWord.name}
                </h2>
                {selectedWord.description && (
                  <p className="text-sm mt-0.5 text-[#656d76] dark:text-[#8b949e]">
                    {selectedWord.description}
                  </p>
                )}
              </div>
              <button
                onClick={() => setSelectedWord(null)}
                className="p-1.5 rounded-md transition-colors text-[#8b949e] dark:text-[#484f58] hover:bg-[#f6f8fa] dark:hover:bg-[#0d1117] hover:text-[#656d76] dark:hover:text-[#8b949e]"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Stats row */}
            <div className="grid grid-cols-3 gap-3 mb-5">
              <div className="rounded-lg p-3 text-center bg-[rgba(19,150,186,0.1)] border border-[rgba(19,150,186,0.15)]">
                <p className="text-2xl font-bold font-mono text-[#1396ba]">
                  {selectedWord.template_count}
                </p>
                <p className="text-sm uppercase tracking-wider mt-0.5 font-medium text-[#656d76] dark:text-[#8b949e]">
                  Templates
                </p>
              </div>
              <div className="rounded-lg p-3 text-center border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#0d1117]">
                <p className="text-2xl font-bold font-mono text-[#1f2328] dark:text-[#e6edf3]">
                  {uniqueContributors}
                </p>
                <p className="text-sm uppercase tracking-wider mt-0.5 font-medium text-[#656d76] dark:text-[#8b949e]">
                  Contributeurs
                </p>
              </div>
              <div className="rounded-lg p-3 text-center border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#0d1117]">
                <div className={`text-2xl font-bold ${
                  selectedWord.is_active ? 'text-[#10b981]' : 'text-[#8b949e]'
                }`}>
                  {selectedWord.is_active ? 'On' : 'Off'}
                </div>
                <p className="text-sm uppercase tracking-wider mt-0.5 font-medium text-[#656d76] dark:text-[#8b949e]">
                  Statut
                </p>
              </div>
            </div>

            {/* Template coverage bar */}
            <div className="mb-5">
              <div className="flex justify-between text-sm mb-1.5">
                <span className="font-medium text-[#656d76] dark:text-[#8b949e]">Couverture</span>
                <span className="font-mono font-bold text-[#1396ba]">
                  {selectedWord.template_count}/20
                </span>
              </div>
              <div className="h-2 rounded-full overflow-hidden bg-[#d0d7de] dark:bg-[#30363d]">
                <div
                  className="h-full rounded-full bg-[#1396ba] transition-all"
                  style={{ width: `${Math.min(100, (selectedWord.template_count / 20) * 100)}%` }}
                />
              </div>
            </div>

            {/* Contributions list */}
            <div>
              <h3 className="text-sm font-semibold mb-3 uppercase tracking-wider flex items-center gap-2 text-[#656d76] dark:text-[#8b949e]">
                <Users className="w-3.5 h-3.5 text-[#1396ba]" />
                Contributions
              </h3>
              {loadingDetail ? (
                <div className="space-y-2">
                  {[1, 2, 3].map(i => (
                    <div key={i} className="rounded-md p-3 h-12 animate-pulse bg-[#f6f8fa] dark:bg-[#0d1117]" />
                  ))}
                </div>
              ) : wordContributions.length === 0 ? (
                <p className="text-sm text-[#8b949e] dark:text-[#484f58]">Aucune contribution</p>
              ) : (
                <div className="space-y-1 max-h-44 overflow-auto">
                  {wordContributions.map(c => (
                    <div key={c.id} className="rounded-md p-3 flex items-center justify-between border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#0d1117]">
                      <div className="flex items-center gap-2.5">
                        <div className="w-6 h-6 rounded-md flex items-center justify-center bg-[rgba(19,150,186,0.1)]">
                          <span className="text-sm font-bold text-[#1396ba]">
                            {c.contributor[0]?.toUpperCase()}
                          </span>
                        </div>
                        <div>
                          <span className="text-sm font-medium text-[#1f2328] dark:text-[#e6edf3]">{c.contributor}</span>
                          {c.recorded_at && (
                            <p className="text-sm font-mono text-[#8b949e] dark:text-[#484f58]">
                              {new Date(c.recorded_at).toLocaleDateString('fr-FR')}
                            </p>
                          )}
                        </div>
                      </div>
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
