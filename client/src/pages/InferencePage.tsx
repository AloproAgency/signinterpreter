import { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import { useWebcam } from '../hooks/useWebcam';
import { useInference } from '../hooks/useInference';
import { useSpeech } from '../hooks/useSpeech';
import { useMediaPipe } from '../hooks/useMediaPipe';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';
import { useApp } from '../lib/context';
import {
  Video, Trash2, Copy, Settings2, Loader2,
  Volume2, VolumeX, Send, Type, Square,
} from 'lucide-react';
import { api } from '../lib/api';

export default function InferencePage() {
  const { videoRef, canvasRef, active, start, stop } = useWebcam();
  const { init: initMediaPipe, ready: mediaPipeReady, loading: mediaPipeLoading, processFrame } = useMediaPipe();
  const {
    connected, status, prediction, sentence, translated, translatedScore,
    phrases, phraseSigns, phraseScores,
    connect, disconnect, clearSentence, setThreshold, finalizeSentence,
    ctcActive,
  } = useInference(processFrame, videoRef, active, mediaPipeReady);

  const TRANSLATION_MIN_SCORE = 0.7;
  const { addToast } = useApp();
  const speech = useSpeech();
  const lastSpokenRef = useRef<number>(0);

  const [showSettings, setShowSettings] = useState(false);
  const [threshold, setThresholdLocal] = useState(0.8);
  const [manualInput, setManualInput] = useState('');
  const [manualResult, setManualResult] = useState('');
  const [showManual, setShowManual] = useState(false);

  useEffect(() => { connect(); return () => disconnect(); }, [connect, disconnect]);
  useEffect(() => { initMediaPipe(); }, [initMediaPipe]);

  useEffect(() => {
    if (phrases.length <= lastSpokenRef.current) {
      lastSpokenRef.current = Math.min(lastSpokenRef.current, phrases.length);
      return;
    }
    const idx = phrases.length - 1;
    const score = phraseScores[idx] ?? 0;
    if (phrases[idx] && score >= TRANSLATION_MIN_SCORE) speech.speak(phrases[idx]);
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
    } catch { setManualResult('Erreur'); }
  };

  const copyAll = useCallback(() => {
    const text = [...phrases, translated].filter(Boolean).join('\n');
    if (!text) return;
    navigator.clipboard.writeText(text);
    addToast('success', 'Copié');
  }, [phrases, translated, addToast]);

  const handleClear = useCallback(() => {
    speech.cancel();
    lastSpokenRef.current = 0;
    clearSentence();
  }, [clearSentence, speech]);

  const shortcuts = useMemo(() => ({
    ' ': handleClear,
    'Enter': finalizeSentence,
  }), [handleClear, finalizeSentence]);
  useKeyboardShortcuts(shortcuts);

  const handleThresholdChange = (v: number) => { setThresholdLocal(v); setThreshold(v); };

  const isEmpty = phrases.length === 0 && sentence.length === 0 && !translated;
  const topK = prediction?.top_k ?? [];

  /* Build the displayed text — all completed phrases + current */
  const allText = useMemo(() => {
    const parts: string[] = [];
    phrases.forEach((phrase, i) => {
      const score = phraseScores[i] ?? 0;
      const signs = phraseSigns[i] ?? [];
      parts.push(score >= TRANSLATION_MIN_SCORE ? phrase : signs.join(' · '));
    });
    return parts;
  }, [phrases, phraseScores, phraseSigns]);

  return (
    <div className="h-full relative bg-zinc-950 overflow-hidden">

      {/* ── Camera — full background ──────────────── */}
      <video
        ref={videoRef}
        autoPlay playsInline muted
        className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-700 ${active ? 'opacity-100' : 'opacity-0'}`}
        style={{ filter: 'brightness(0.45) saturate(0.7)' }}
      />
      <canvas ref={canvasRef} className="hidden" />

      {/* ── Overlays ─────────────────────────────── */}
      {/* Uniform dark base over the whole frame */}
      <div className="absolute inset-0 bg-black/20 pointer-events-none" />
      {/* Top fade */}
      <div className="absolute inset-x-0 top-0 h-40 bg-linear-to-b from-black/70 to-transparent pointer-events-none" />
      {/* Bottom gradient — covers ~65% */}
      <div className="absolute inset-x-0 bottom-0 h-[65%] bg-linear-to-t from-black/95 via-black/60 to-transparent pointer-events-none" />

      {/* ── Idle — start button ───────────────────── */}
      {!active && (
        <div className="absolute inset-0 z-40 flex flex-col items-center justify-center gap-4">
          <button
            onClick={handleStart}
            disabled={mediaPipeLoading}
            className="group cursor-pointer disabled:cursor-wait flex flex-col items-center gap-3"
          >
            <div className="w-24 h-24 rounded-full border-2 border-white/20 bg-white/8 backdrop-blur-sm flex items-center justify-center group-hover:bg-white/15 group-hover:border-white/35 transition-all duration-300">
              {mediaPipeLoading
                ? <Loader2 className="w-9 h-9 text-white animate-spin" />
                : <Video className="w-9 h-9 text-white ml-1" />
              }
            </div>
            <span className="text-sm text-white/40 group-hover:text-white/70 transition-colors tracking-wide">
              {mediaPipeLoading ? 'Chargement…' : 'Activer la caméra'}
            </span>
          </button>
        </div>
      )}

      {/* ── Top bar ──────────────────────────────── */}
      <div className="absolute top-0 left-0 right-0 z-30 flex items-center gap-2 px-5 pt-5">

        {/* Status pill */}
        <div className="flex items-center gap-2 bg-black/30 backdrop-blur-md rounded-full px-3 py-1.5 border border-white/10">
          <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${connected ? 'bg-emerald-400 animate-pulse' : 'bg-white/20'}`} />
          <span className="text-[11px] font-mono text-white/60 tracking-wide">
            {connected ? (ctcActive ? 'CTC' : 'SW') : 'offline'}
          </span>
          {prediction && (
            <span className="text-[10px] font-mono text-white/30 border-l border-white/15 pl-2">
              {prediction.inference_ms}ms
            </span>
          )}
        </div>

        <div className="flex-1" />

        {/* Icon buttons — pill group */}
        <div className="flex items-center gap-1">
          {speech.available && (
            <button
              onClick={() => speech.setEnabled(!speech.enabled)}
              className={`w-9 h-9 rounded-full backdrop-blur-md flex items-center justify-center transition-all cursor-pointer border ${
                speech.enabled
                  ? 'bg-white/20 border-white/20 text-white'
                  : 'bg-black/25 border-white/10 text-white/40 hover:bg-white/15 hover:text-white/80'
              }`}
            >
              {speech.enabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
            </button>
          )}
          <button
            onClick={() => setShowManual(v => !v)}
            className={`w-9 h-9 rounded-full backdrop-blur-md flex items-center justify-center transition-all cursor-pointer border ${
              showManual
                ? 'bg-white/20 border-white/20 text-white'
                : 'bg-black/25 border-white/10 text-white/40 hover:bg-white/15 hover:text-white/80'
            }`}
          >
            <Type className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowSettings(v => !v)}
            className={`w-9 h-9 rounded-full backdrop-blur-md flex items-center justify-center transition-all cursor-pointer border ${
              showSettings
                ? 'bg-white/20 border-white/20 text-white'
                : 'bg-black/25 border-white/10 text-white/40 hover:bg-white/15 hover:text-white/80'
            }`}
          >
            <Settings2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* ── Top-K predictions (top-right, under bar) ── */}
      {active && topK.length > 0 && (
        <div className="absolute right-5 top-18 z-20 space-y-0.5">
          {topK.slice(0, 4).map((alt, i) => {
            const p = Math.round(Math.exp(-alt.distance) * 100);
            return (
              <div key={alt.word} className={`flex items-center gap-2 px-2.5 py-1 rounded-lg ${i === 0 ? 'bg-white/12 backdrop-blur-sm' : ''}`}>
                <span className={`text-xs font-mono ${i === 0 ? 'text-white font-semibold' : 'text-white/25'}`}>{alt.word}</span>
                <span className={`text-[10px] font-mono ml-auto ${i === 0 ? 'text-white/50' : 'text-white/15'}`}>{p}%</span>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Manual translate ──────────────────────── */}
      {showManual && (
        <div className="absolute left-5 right-5 top-18 z-30 animate-slide-up">
          <div className="flex gap-2">
            <input
              type="text"
              value={manualInput}
              onChange={e => setManualInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleManualTranslate()}
              placeholder="moi manger pomme…"
              className="flex-1 px-3 py-2 text-sm bg-black/40 backdrop-blur-md border border-white/15 rounded-xl text-white placeholder-white/25 focus:outline-none focus:ring-1 focus:ring-white/30"
            />
            <button
              onClick={handleManualTranslate}
              className="px-4 py-2 bg-white/12 hover:bg-white/20 backdrop-blur-md border border-white/15 text-white rounded-xl cursor-pointer transition-all"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          {manualResult && (
            <p className="mt-2 text-sm text-white/50 pl-1">→ <span className="text-white font-medium">{manualResult}</span></p>
          )}
        </div>
      )}

      {/* ── Transcript — teleprompter text block ─────── */}
      <div className="absolute inset-x-0 z-20 flex flex-col items-center justify-end text-center px-8 overflow-hidden" style={{ top: '10%', bottom: '148px' }}>

        {!isEmpty ? (
          <div className="w-full space-y-4">

            {/* Show only last 2 completed phrases to avoid overflow */}
            {allText.slice(-2).map((text, i) => (
              <p
                key={`phrase-${i}`}
                className="text-white text-5xl md:text-6xl font-bold leading-tight tracking-tight animate-fade-in"
              >
                {text}
              </p>
            ))}

            {/* Current in-progress phrase */}
            {(sentence.length > 0 || translated) && (
              <div className="animate-fade-in">
                {/* Translation or signs — same size/color */}
                {translated && translatedScore >= TRANSLATION_MIN_SCORE ? (
                  <p className="text-blue-400 text-5xl md:text-6xl font-bold leading-tight tracking-tight">
                    {translated}
                    <span className="inline-block w-0.5 h-11 bg-blue-400 ml-2 animate-pulse align-middle rounded-full" />
                  </p>
                ) : sentence.length > 0 ? (
                  <p className="text-blue-400 text-5xl md:text-6xl font-bold leading-tight tracking-tight">
                    {sentence.join(' · ')}
                    <span className="inline-block w-0.5 h-11 bg-blue-400 ml-2 animate-pulse align-middle rounded-full" />
                  </p>
                ) : null}
              </div>
            )}

          </div>
        ) : active ? (
          <div className="flex-1 flex items-center justify-center">
            <p className="text-white/20 text-3xl font-medium tracking-wide">Signez pour transcrire…</p>
          </div>
        ) : null}
      </div>

      {/* ── Live prediction + dots (centered, above buttons) ── */}
      <div className="absolute left-0 right-0 z-30 flex flex-col items-center gap-3" style={{ bottom: '72px' }}>

        {/* Recording indicator */}
        {active && prediction && status.hand_visible && (
          <div className="flex items-center gap-2.5 bg-black/30 backdrop-blur-md rounded-full px-4 py-1.5 border border-white/10">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse shrink-0" />
            <span className="text-sm font-mono text-white/65 tabular-nums tracking-wide">{prediction.word || '—'}</span>
            <div className="w-16 h-0.5 bg-white/15 rounded-full overflow-hidden">
              <div
                className="h-full bg-white/50 rounded-full transition-all duration-150"
                style={{ width: `${Math.round(prediction.confidence * 100)}%` }}
              />
            </div>
          </div>
        )}

        {/* Mediapipe dots */}
        {active && (
          <div className="flex items-center gap-1.5">
            <span className={`w-1.5 h-1.5 rounded-full ${mediaPipeReady ? 'bg-blue-400' : 'bg-white/15'}`} />
            <span className={`w-1.5 h-1.5 rounded-full ${status.hand_visible ? 'bg-emerald-400 animate-pulse' : 'bg-white/15'}`} />
          </div>
        )}
      </div>

      {/* ── Bottom 3 action buttons ───────────────── */}
      <div className="absolute bottom-0 left-0 right-0 z-30 flex items-center justify-center gap-8 pb-9">

        {/* Left — Copy (small) */}
        <button
          onClick={copyAll}
          disabled={isEmpty}
          className="w-12 h-12 rounded-full bg-white/10 backdrop-blur-md border border-white/15 flex items-center justify-center text-white/60 hover:bg-white/20 hover:text-white transition-all cursor-pointer disabled:opacity-20"
        >
          <Copy className="w-5 h-5" />
        </button>

        {/* Center — Start/Stop dynamique */}
        <button
          onClick={active ? stop : handleStart}
          disabled={mediaPipeLoading}
          className={`w-18 h-18 rounded-full border-2 flex items-center justify-center active:scale-95 transition-all duration-300 cursor-pointer disabled:opacity-40 shadow-2xl ${
            !active
              ? 'bg-white/15 backdrop-blur-md border-white/30 text-white hover:bg-white/25'
              : status.hand_visible
              ? 'bg-red-500 border-red-400 text-white shadow-red-500/40'
              : 'bg-white/15 backdrop-blur-md border-white/30 text-white hover:bg-white/25'
          }`}
        >
          {mediaPipeLoading ? (
            <Loader2 className="w-7 h-7 animate-spin" />
          ) : !active ? (
            <Video className="w-7 h-7" />
          ) : status.hand_visible ? (
            /* Recording indicator — pulsing red dot */
            <span className="w-5 h-5 rounded-full bg-white animate-pulse" />
          ) : (
            <Square className="w-6 h-6 fill-white" />
          )}
        </button>

        {/* Right — Clear */}
        <button
          onClick={handleClear}
          disabled={isEmpty}
          className="w-12 h-12 rounded-full bg-white/10 backdrop-blur-md border border-white/15 flex items-center justify-center text-white/60 hover:bg-white/20 hover:text-white transition-all cursor-pointer disabled:opacity-20"
        >
          <Trash2 className="w-5 h-5" />
        </button>
      </div>

      {/* ── Settings popover ─────────────────────── */}
      {showSettings && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setShowSettings(false)} />
          <div className="fixed right-4 top-16 z-50 w-72 bg-zinc-950/95 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl animate-scale-in overflow-hidden">
            <div className="px-5 py-4 border-b border-white/8">
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm font-semibold text-white">Seuil de confiance</span>
                <span className="text-sm font-mono font-bold text-blue-400">{threshold.toFixed(2)}</span>
              </div>
              <input
                type="range" min={0.1} max={3.0} step={0.05}
                value={threshold}
                onChange={e => handleThresholdChange(Number(e.target.value))}
                className="w-full cursor-pointer"
              />
              <div className="flex justify-between mt-2 text-xs text-white/30">
                <span>strict</span><span>permissif</span>
              </div>
            </div>
            <div className="px-5 py-4 grid grid-cols-2 gap-2 text-xs text-white/35 text-center">
              <div><kbd className="block font-mono bg-white/8 px-2 py-1 rounded-lg border border-white/10 mb-1">↵</kbd>phrase</div>
              <div><kbd className="block font-mono bg-white/8 px-2 py-1 rounded-lg border border-white/10 mb-1">⎵</kbd>effacer</div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
