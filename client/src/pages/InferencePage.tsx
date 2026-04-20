import { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import { useWebcam } from '../hooks/useWebcam';
import { useInference } from '../hooks/useInference';
import { useSpeech } from '../hooks/useSpeech';
import { useMediaPipe } from '../hooks/useMediaPipe';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';
import { useApp } from '../lib/context';
import {
  Video, Wifi, WifiOff, Trash2, Activity,
  Copy, Settings2, Maximize, Minimize,
  Zap, Send, Type, Loader2, Volume2, VolumeX,
} from 'lucide-react';
import { api } from '../lib/api';

export default function InferencePage() {
  const { videoRef, canvasRef, active, start } = useWebcam();
  const { init: initMediaPipe, ready: mediaPipeReady, loading: mediaPipeLoading, processFrame } = useMediaPipe();
  const {
    connected, status, prediction, sentence, translated, translatedScore,
    phrases, phraseSigns, phraseScores, lastAddedIndex,
    connect, disconnect, clearSentence, setThreshold, finalizeSentence,
  } = useInference(processFrame, videoRef, active, mediaPipeReady);

  // Only display the French translation when we're confident enough.
  // Otherwise fall back to the raw sign sequence so the user can still read it.
  const TRANSLATION_MIN_SCORE = 0.7;
  const { addToast } = useApp();
  const speech = useSpeech();
  const lastSpokenRef = useRef<number>(0);

  const [showSettings, setShowSettings] = useState(false);
  const [threshold, setThresholdLocal] = useState(0.8);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [manualInput, setManualInput] = useState('');
  const [manualResult, setManualResult] = useState('');
  const [showManual, setShowManual] = useState(false);
  const transcriptRef = useRef<HTMLDivElement>(null);

  useEffect(() => { connect(); return () => disconnect(); }, [connect, disconnect]);
  useEffect(() => { initMediaPipe(); }, [initMediaPipe]);

  // Auto-scroll transcript to bottom when new phrase arrives
  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: 0,
      behavior: 'smooth',
    });
  }, [phrases.length, translated]);

  // Speak each newly finalized phrase (only high-confidence ones).
  useEffect(() => {
    if (phrases.length <= lastSpokenRef.current) {
      lastSpokenRef.current = Math.min(lastSpokenRef.current, phrases.length);
      return;
    }
    const idx = phrases.length - 1;
    const text = phrases[idx];
    const score = phraseScores[idx] ?? 0;
    if (text && score >= TRANSLATION_MIN_SCORE) {
      speech.speak(text);
    }
    lastSpokenRef.current = phrases.length;
  }, [phrases, phraseScores, speech]);

  const handleStart = useCallback(async () => {
    await initMediaPipe();
    await start();
  }, [initMediaPipe, start]);

  const handleManualTranslate = async () => {
    if (!manualInput.trim()) return;
    try {
      const result = await api.translate(manualInput.trim());
      setManualResult(result.sentence);
    } catch {
      setManualResult('Erreur de traduction');
    }
  };

  const copyAll = useCallback(() => {
    const fullText = [...phrases, translated].filter(Boolean).join('\n');
    if (!fullText) return;
    navigator.clipboard.writeText(fullText);
    addToast('success', 'Copié');
  }, [phrases, translated, addToast]);

  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  const handleClear = useCallback(() => {
    speech.cancel();
    lastSpokenRef.current = 0;
    clearSentence();
  }, [clearSentence, speech]);

  const shortcuts = useMemo(() => ({
    ' ': handleClear,
    'f': toggleFullscreen,
    'Enter': finalizeSentence,
  }), [handleClear, toggleFullscreen, finalizeSentence]);
  useKeyboardShortcuts(shortcuts);

  const handleThresholdChange = (value: number) => {
    setThresholdLocal(value);
    setThreshold(value);
  };

  const isEmpty = phrases.length === 0 && sentence.length === 0 && !translated;

  return (
    <div className="h-full flex flex-col relative bg-white dark:bg-[#0d1117]">

      {/* ============================ HEADER ============================ */}
      <header className="shrink-0 px-4 md:px-6 py-3 border-b border-[#d0d7de] dark:border-[#30363d] bg-white dark:bg-[#0d1117]">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-[rgba(19,150,186,0.1)] border border-[rgba(19,150,186,0.2)] shrink-0">
              <Activity className="w-4 h-4 text-[#1396ba]" />
            </div>
            <div className="min-w-0">
              <h1 className="text-sm font-semibold text-[#1f2328] dark:text-[#e6edf3] truncate">Transcription en direct</h1>
              <div className="flex items-center gap-1.5 text-sm text-[#8b949e] dark:text-[#484f58]">
                <span className={`inline-flex items-center gap-1 ${connected ? 'text-[#10b981]' : ''}`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-[#10b981] animate-pulse' : 'bg-[#8b949e]'}`} />
                  {connected ? 'Connecté' : 'Reconnexion…'}
                </span>
                {mediaPipeReady && <span className="text-[#1396ba]">· MediaPipe prêt</span>}
                {prediction && <span className="font-mono">· {prediction.inference_ms}ms</span>}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-1 shrink-0">
            {speech.available && (
              <button
                onClick={() => speech.setEnabled(!speech.enabled)}
                className={`p-2 rounded-md transition-colors ${speech.enabled
                  ? 'text-[#1396ba] bg-[rgba(19,150,186,0.1)]'
                  : 'text-[#656d76] dark:text-[#8b949e] hover:bg-[#f6f8fa] dark:hover:bg-[#1c2333]'
                }`}
                title={speech.enabled ? 'Couper la lecture vocale' : 'Activer la lecture vocale'}
              >
                {speech.enabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
              </button>
            )}
            <button
              onClick={copyAll}
              disabled={isEmpty}
              className="p-2 rounded-md text-[#656d76] dark:text-[#8b949e] hover:text-[#1396ba] hover:bg-[rgba(19,150,186,0.1)] disabled:opacity-30 disabled:cursor-not-allowed"
              title="Copier la transcription"
            >
              <Copy className="w-4 h-4" />
            </button>
            <button
              onClick={handleClear}
              disabled={isEmpty}
              className="p-2 rounded-md text-[#656d76] dark:text-[#8b949e] hover:text-[#ef4444] hover:bg-[#ef4444]/10 disabled:opacity-30 disabled:cursor-not-allowed"
              title="Effacer (Espace)"
            >
              <Trash2 className="w-4 h-4" />
            </button>
            <div className="w-px h-5 mx-1 bg-[#d0d7de] dark:bg-[#30363d]" />
            <button
              onClick={() => setShowManual(v => !v)}
              className={`p-2 rounded-md transition-colors ${showManual
                ? 'text-[#1396ba] bg-[rgba(19,150,186,0.1)]'
                : 'text-[#656d76] dark:text-[#8b949e] hover:bg-[#f6f8fa] dark:hover:bg-[#1c2333]'
              }`}
              title="Tester la traduction"
            >
              <Type className="w-4 h-4" />
            </button>
            <button
              onClick={() => setShowSettings(v => !v)}
              className={`p-2 rounded-md transition-colors ${showSettings
                ? 'text-[#1396ba] bg-[rgba(19,150,186,0.1)]'
                : 'text-[#656d76] dark:text-[#8b949e] hover:bg-[#f6f8fa] dark:hover:bg-[#1c2333]'
              }`}
              title="Paramètres"
            >
              <Settings2 className="w-4 h-4" />
            </button>
            <button
              onClick={toggleFullscreen}
              className="p-2 rounded-md hidden md:block text-[#656d76] dark:text-[#8b949e] hover:bg-[#f6f8fa] dark:hover:bg-[#1c2333]"
              title="Plein écran (F)"
            >
              {isFullscreen ? <Minimize className="w-4 h-4" /> : <Maximize className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </header>

      {/* Settings popover */}
      {showSettings && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setShowSettings(false)} />
          <div className="absolute right-6 top-16 z-50 w-80 rounded-lg border border-[#d0d7de] dark:border-[#30363d] bg-white dark:bg-[#161b22] shadow-2xl animate-scale-in">
            <div className="p-4">
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-medium text-[#1f2328] dark:text-[#e6edf3]">Seuil de confiance</label>
                <span className="text-base font-mono font-bold text-[#1396ba]">{threshold.toFixed(2)}</span>
              </div>
              <input
                type="range" min={0.1} max={3.0} step={0.05}
                value={threshold}
                onChange={e => handleThresholdChange(Number(e.target.value))}
                className="w-full h-2 rounded-full appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, #1396ba 0%, #1396ba ${(threshold - 0.1) / 2.9 * 100}%, ${
                    document.documentElement.classList.contains('dark') ? '#30363d' : '#d0d7de'
                  } ${(threshold - 0.1) / 2.9 * 100}%, ${
                    document.documentElement.classList.contains('dark') ? '#30363d' : '#d0d7de'
                  } 100%)`
                }}
              />
              <div className="flex justify-between mt-1.5 text-sm text-[#8b949e]">
                <span>Strict</span><span>Permissif</span>
              </div>
            </div>
            <div className="border-t border-[#d0d7de] dark:border-[#30363d] px-4 py-2.5 flex gap-4 text-sm text-[#8b949e]">
              <span><kbd className="px-1.5 py-0.5 rounded bg-[#f6f8fa] dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] font-mono">↵</kbd> Fin de phrase</span>
              <span><kbd className="px-1.5 py-0.5 rounded bg-[#f6f8fa] dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] font-mono">⎵</kbd> Effacer</span>
              <span><kbd className="px-1.5 py-0.5 rounded bg-[#f6f8fa] dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] font-mono">F</kbd> Plein écran</span>
            </div>
          </div>
        </>
      )}

      {/* Manual translation */}
      {showManual && (
        <div className="shrink-0 px-4 md:px-6 py-3 border-b border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22] animate-slide-up">
          <div className="flex gap-2 items-center max-w-2xl">
            <input
              type="text"
              value={manualInput}
              onChange={e => setManualInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleManualTranslate()}
              placeholder="Tapez des signes : moi manger pomme"
              className="flex-1 rounded-md px-3 py-2 text-sm bg-white dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] text-[#1f2328] dark:text-[#e6edf3] placeholder-[#8b949e] focus:outline-none focus:border-[#1396ba] focus:ring-1 focus:ring-[#1396ba]/30"
            />
            <button
              onClick={handleManualTranslate}
              className="px-4 py-2 rounded-md text-sm font-medium bg-[#1396ba] hover:bg-[#17b8e3] text-white cursor-pointer flex items-center gap-1.5"
            >
              <Send className="w-3.5 h-3.5" />
              Traduire
            </button>
          </div>
          {manualResult && (
            <p className="mt-2 text-base font-medium text-[#1f2328] dark:text-[#e6edf3]">→ {manualResult}</p>
          )}
        </div>
      )}

      {/* ============================ MAIN ============================ */}
      <main className="flex-1 min-h-0 flex flex-col lg:flex-row">

        {/* Transcript — primary focus */}
        <section className="flex-1 min-h-0 flex flex-col border-b lg:border-b-0 lg:border-r border-[#d0d7de] dark:border-[#30363d]">
          <div
            ref={transcriptRef}
            className="flex-1 min-h-0 overflow-y-auto px-6 md:px-10 py-8"
          >
            {isEmpty ? (
              <div className="h-full flex flex-col items-center justify-center text-center">
                <div className="w-16 h-16 rounded-2xl flex items-center justify-center bg-[rgba(19,150,186,0.08)] border border-[rgba(19,150,186,0.15)] mb-5">
                  <Activity className="w-7 h-7 text-[#1396ba]" />
                </div>
                <h2 className="text-lg font-semibold text-[#1f2328] dark:text-[#e6edf3] mb-1.5">
                  Prêt à transcrire
                </h2>
                <p className="text-sm text-[#8b949e] dark:text-[#484f58] max-w-sm">
                  {active
                    ? 'Signez devant la caméra pour voir la transcription apparaître ici.'
                    : 'Activez la caméra pour commencer.'}
                </p>
              </div>
            ) : (
              <div className="max-w-2xl mx-auto flex flex-col divide-y divide-[#d0d7de]/60 dark:divide-[#30363d]/60">
                {/* Current (in-progress) phrase — always on top */}
                {(sentence.length > 0 || translated) && (
                  <article className="animate-fade-in py-4 first:pt-0">
                    <div className="flex items-baseline gap-3">
                      <span className="text-sm font-mono text-[#1396ba] tabular-nums shrink-0">
                        {String(phrases.length + 1).padStart(2, '0')}
                      </span>
                      <div className="flex-1 min-w-0">
                        {translated && translatedScore >= TRANSLATION_MIN_SCORE ? (
                          // Traduction confiante → phrase française en gros + pills en dessous
                          <>
                            <p className="text-xl md:text-2xl leading-relaxed text-[#1396ba] font-semibold flex items-baseline gap-2">
                              <span className="min-w-0">{translated}</span>
                              <span className="inline-block w-0.5 h-5 bg-[#1396ba] animate-pulse align-[-0.1em]" />
                              <span
                                className="text-sm font-mono font-semibold tabular-nums px-1.5 py-0.5 rounded shrink-0 text-[#10b981] bg-[#10b981]/10"
                                title="Confiance de la traduction"
                              >
                                {Math.round(translatedScore * 100)}%
                              </span>
                            </p>
                            {sentence.length > 0 && (
                              <div className="flex gap-1.5 mt-2 flex-wrap">
                                {sentence.map((word, i) => (
                                  <span
                                    key={`${i}-${word}`}
                                    className={`px-2 py-0.5 rounded text-sm text-[#656d76] dark:text-[#8b949e] bg-[#f6f8fa] dark:bg-[#1c2333] border border-[#d0d7de] dark:border-[#30363d] ${
                                      i === lastAddedIndex ? 'animate-word-pop' : ''
                                    }`}
                                  >
                                    {word}
                                  </span>
                                ))}
                              </div>
                            )}
                          </>
                        ) : (
                          // Traduction absente ou peu fiable → on montre la séquence de signes.
                          sentence.length > 0 && (
                            <p className="text-xl md:text-2xl leading-relaxed text-[#1396ba] font-semibold flex gap-2 flex-wrap items-baseline">
                              {sentence.map((word, i) => (
                                <span
                                  key={`${i}-${word}`}
                                  className={i === lastAddedIndex ? 'animate-word-pop' : ''}
                                >
                                  {word}
                                </span>
                              ))}
                              <span className="inline-block w-0.5 h-5 bg-[#1396ba] animate-pulse align-[-0.1em]" />
                              {translated && translatedScore > 0 && (
                                <span
                                  className={`text-sm font-mono font-semibold tabular-nums px-1.5 py-0.5 rounded shrink-0 ${
                                    translatedScore >= 0.4 ? 'text-[#d97706] bg-[#d97706]/10' : 'text-[#ef4444] bg-[#ef4444]/10'
                                  }`}
                                  title="Traduction masquée — confiance trop basse"
                                >
                                  {Math.round(translatedScore * 100)}%
                                </span>
                              )}
                            </p>
                          )
                        )}
                      </div>
                    </div>
                  </article>
                )}

                {/* Completed phrases — newest first */}
                {[...phrases].map((_, reverseIdx, arr) => {
                  const i = arr.length - 1 - reverseIdx;
                  const phrase = phrases[i];
                  const s = phraseScores[i] ?? 0;
                  const signs = phraseSigns[i] ?? [];
                  const isConfident = s >= TRANSLATION_MIN_SCORE;
                  const pct = Math.round(s * 100);
                  const tone = s >= 0.7 ? 'text-[#10b981] bg-[#10b981]/10'
                    : s >= 0.4 ? 'text-[#d97706] bg-[#d97706]/10'
                    : 'text-[#ef4444] bg-[#ef4444]/10';
                  return (
                    <article
                      key={`phrase-${i}`}
                      className="group animate-fade-in py-4 first:pt-0"
                    >
                      <div className="flex items-baseline gap-3">
                        <span className="text-sm font-mono text-[#8b949e] dark:text-[#484f58] tabular-nums shrink-0">
                          {String(i + 1).padStart(2, '0')}
                        </span>
                        {isConfident ? (
                          <p className="text-xl md:text-2xl leading-relaxed text-[#1f2328] dark:text-[#e6edf3] font-medium flex-1 wrap-break-word">
                            {phrase}
                          </p>
                        ) : (
                          <p className="text-xl md:text-2xl leading-relaxed text-[#656d76] dark:text-[#8b949e] font-medium flex-1 italic wrap-break-word">
                            {signs.length > 0 ? signs.join(' · ') : phrase}
                          </p>
                        )}
                        <span
                          className={`text-sm font-mono font-semibold tabular-nums px-1.5 py-0.5 rounded shrink-0 ${tone}`}
                          title={isConfident ? 'Confiance de la traduction' : 'Confiance trop basse — séquence de signes affichée'}
                        >
                          {pct}%
                        </span>
                      </div>
                    </article>
                  );
                })}
              </div>
            )}
          </div>

          {/* Live prediction strip — hidden when hand is not visible */}
          {active && prediction && status.hand_visible && (
            <div className="shrink-0 px-6 md:px-10 py-3 border-t border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22]">
              <div className="max-w-2xl mx-auto flex items-center gap-3">
                <span className="text-sm uppercase tracking-wider font-semibold text-[#8b949e]">
                  En cours
                </span>
                <span className={`text-base font-semibold ${
                  prediction.is_final ? 'text-[#1396ba]' : 'text-[#1f2328] dark:text-[#e6edf3]'
                }`}>
                  {prediction.word || '—'}
                </span>
                <div className="flex-1 h-1 rounded-full overflow-hidden bg-[#d0d7de] dark:bg-[#30363d] max-w-[160px]">
                  <div
                    className="h-full rounded-full bg-[#1396ba] transition-all duration-300"
                    style={{ width: `${Math.max(0, Math.min(100, prediction.confidence * 100))}%` }}
                  />
                </div>
                <span className="text-sm font-mono font-bold text-[#8b949e] tabular-nums min-w-[3ch] text-right">
                  {Math.round(prediction.confidence * 100)}%
                </span>
              </div>
              {/* Ambiguity strip: if top-1 is uncertain (<75%) AND top-2
                  is close (dist gap small), show the next 2 candidates
                  for the user to know the model is hesitating. */}
              {(() => {
                const tk = prediction.top_k ?? [];
                if (tk.length < 2) return null;
                const top1 = Math.exp(-tk[0].distance);
                const top2 = Math.exp(-tk[1].distance);
                const ambiguous = top1 < 0.75 && (top1 - top2) < 0.15;
                if (!ambiguous) return null;
                return (
                  <div className="max-w-2xl mx-auto mt-2 flex items-center gap-2 text-xs">
                    <span className="text-[#8b949e] uppercase tracking-wider">Aussi possible</span>
                    {tk.slice(1, 4).map((alt) => {
                      const p = Math.exp(-alt.distance);
                      return (
                        <span
                          key={alt.word}
                          className="px-2 py-0.5 rounded-full bg-[#d0d7de]/40 dark:bg-[#30363d]/60 text-[#1f2328] dark:text-[#e6edf3]"
                        >
                          {alt.word} <span className="text-[#8b949e] font-mono">{Math.round(p * 100)}%</span>
                        </span>
                      );
                    })}
                  </div>
                );
              })()}
            </div>
          )}
        </section>

        {/* Video — secondary, compact */}
        <aside className="shrink-0 w-full lg:w-72 xl:w-80 p-4 lg:p-5 flex flex-col gap-3 bg-[#f6f8fa] dark:bg-[#161b22]">
          <div className={`relative rounded-xl overflow-hidden aspect-square border-2 transition-colors ${
            active && status.is_signing
              ? 'border-[#1396ba] shadow-[0_0_0_3px_rgba(19,150,186,0.15)]'
              : 'border-[#d0d7de] dark:border-[#30363d]'
          }`}>
            <video
              ref={videoRef}
              autoPlay playsInline muted
              className="w-full h-full object-cover"
            />
            <canvas ref={canvasRef} className="hidden" />

            {active && (
              <>
                {/* Top-left : hand status */}
                <div className="absolute top-2 left-2">
                  <div className="flex items-center gap-1.5 bg-black/60 backdrop-blur-sm px-2 py-1 rounded-md text-sm font-medium text-white/90">
                    <div className={`w-1.5 h-1.5 rounded-full ${
                      status.hand_visible ? 'bg-[#10b981] animate-pulse' : 'bg-[#484f58]'
                    }`} />
                    {status.hand_visible ? 'Main' : 'Pas de main'}
                  </div>
                </div>

                {/* Top-right : REC */}
                {status.is_signing && (
                  <div className="absolute top-2 right-2 flex items-center gap-1.5 bg-[#ef4444]/90 backdrop-blur-sm px-2 py-1 rounded-md">
                    <div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                    <span className="text-white text-sm font-bold tracking-wider">REC</span>
                  </div>
                )}

                {/* Bottom : motion bar */}
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 via-black/40 to-transparent px-3 pt-5 pb-2">
                  <div className="flex items-center gap-2 mb-1">
                    <Zap className={`w-3 h-3 ${
                      status.motion_energy > 0.05 ? 'text-[#1396ba]' : 'text-white/40'
                    }`} />
                    <div className="flex-1 h-1 rounded-full overflow-hidden bg-white/15">
                      <div
                        className="h-full rounded-full bg-[#1396ba] transition-all duration-150 ease-out"
                        style={{ width: `${Math.min(100, status.motion_energy * 1000)}%` }}
                      />
                    </div>
                  </div>
                </div>
              </>
            )}

            {/* Start overlay */}
            {!active && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/75 backdrop-blur-sm">
                <button
                  onClick={handleStart}
                  disabled={mediaPipeLoading}
                  className="bg-[#1396ba] hover:bg-[#17b8e3] disabled:opacity-60 disabled:cursor-wait text-white px-4 py-2.5 rounded-lg font-medium text-sm flex items-center gap-2 transition-colors cursor-pointer shadow-lg"
                >
                  {mediaPipeLoading ? (
                    <><Loader2 className="w-4 h-4 animate-spin" />Chargement…</>
                  ) : (
                    <><Video className="w-4 h-4" />Activer la caméra</>
                  )}
                </button>
              </div>
            )}
          </div>

          {/* Compact status line below video */}
          <div className="flex items-center justify-between text-sm">
            <div className={`flex items-center gap-1.5 px-2 py-1 rounded-md ${
              connected
                ? 'text-[#10b981] bg-[#10b981]/10'
                : 'text-[#8b949e] bg-[#8b949e]/10'
            }`}>
              {connected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
              <span>{connected ? 'Serveur' : 'Hors-ligne'}</span>
            </div>
            <div className={`flex items-center gap-1.5 px-2 py-1 rounded-md ${
              mediaPipeReady
                ? 'text-[#1396ba] bg-[rgba(19,150,186,0.1)]'
                : 'text-[#8b949e] bg-[#8b949e]/10'
            }`}>
              {mediaPipeReady
                ? <div className="w-1.5 h-1.5 rounded-full bg-[#1396ba]" />
                : <Loader2 className="w-3 h-3 animate-spin" />}
              <span>MediaPipe</span>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
