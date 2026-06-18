import { useState, useEffect, useRef, useCallback } from 'react';
import { useWebcam } from '../hooks/useWebcam';
import { useApp } from '../lib/context';
import { api } from '../lib/api';
import { Video, Square, RotateCcw, Play } from 'lucide-react';
import type { WordInfo } from '../lib/types';

interface Landmarks {
  pose: { x: number; y: number; v: number }[];
  left_hand: { x: number; y: number }[];
  right_hand: { x: number; y: number }[];
}

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17],
];
const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],
];

function drawLandmarks(
  ctx: CanvasRenderingContext2D,
  lm: Landmarks,
  cw: number, ch: number,
  fw: number, fh: number,
) {
  const sx = cw / fw, sy = ch / fh;
  ctx.strokeStyle = 'rgba(255,255,255,0.3)';
  ctx.lineWidth = 1.5;
  for (const [a, b] of POSE_CONNECTIONS) {
    if (lm.pose[a] && lm.pose[b]) {
      ctx.beginPath();
      ctx.moveTo(lm.pose[a].x * sx, lm.pose[a].y * sy);
      ctx.lineTo(lm.pose[b].x * sx, lm.pose[b].y * sy);
      ctx.stroke();
    }
  }
  const drawHand = (pts: { x: number; y: number }[]) => {
    if (!pts.length) return;
    ctx.strokeStyle = 'rgba(255,255,255,0.7)';
    ctx.lineWidth = 1.5;
    for (const [a, b] of HAND_CONNECTIONS) {
      if (pts[a] && pts[b]) {
        ctx.beginPath();
        ctx.moveTo(pts[a].x * sx, pts[a].y * sy);
        ctx.lineTo(pts[b].x * sx, pts[b].y * sy);
        ctx.stroke();
      }
    }
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x * sx, p.y * sy, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  };
  drawHand(lm.left_hand);
  drawHand(lm.right_hand);
}

const SEQUENCE_LENGTH = 30;

export default function ContributionPage() {
  const { videoRef, canvasRef, active, start, captureFrame } = useWebcam();
  const { addToast } = useApp();

  const [words, setWords] = useState<WordInfo[]>([]);
  const [selectedWord, setSelectedWord] = useState(() => localStorage.getItem('prefill_word') || '');
  const [newWord, setNewWord] = useState('');
  const [contributor, setContributor] = useState(() => localStorage.getItem('contributor') || '');
  const [teamTaskId] = useState(() => localStorage.getItem('prefill_task_id') || '');
  const [totalSequences, setTotalSequences] = useState(() => {
    const n = localStorage.getItem('prefill_count');
    localStorage.removeItem('prefill_word');
    localStorage.removeItem('prefill_task_id');
    localStorage.removeItem('prefill_count');
    return n ? parseInt(n) : 10;
  });

  const [recording, setRecording] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [handVisible, setHandVisible] = useState(false);
  const [error, setError] = useState('');
  const [countdown, setCountdown] = useState<number | null>(null);
  const [waitingForHand, setWaitingForHand] = useState(false);
  const [currentSequence, setCurrentSequence] = useState(0);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchComplete, setBatchComplete] = useState(false);
  const batchAbortRef = useRef(false);
  const [hasReference, setHasReference] = useState(false);
  const [referenceUrl, setReferenceUrl] = useState('');
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [landmarks, setLandmarks] = useState<Landmarks | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => { api.getVocabulary().then(setWords).catch(() => {}); }, []);
  useEffect(() => { if (contributor) localStorage.setItem('contributor', contributor); }, [contributor]);

  const currentWord = newWord.trim() || selectedWord;

  useEffect(() => {
    if (!currentWord) { setHasReference(false); setReferenceUrl(''); return; }
    api.getReference(currentWord).then(d => {
      setHasReference(d.exists);
      if (d.exists) setReferenceUrl(api.getReferenceVideoUrl(currentWord));
    }).catch(() => setHasReference(false));
  }, [currentWord]);

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/collect`);
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === 'preview' || data.type === 'recording') {
        setHandVisible(data.hand_visible);
        if (data.landmarks) setLandmarks(data.landmarks);
        if (data.type === 'recording') setFrameCount(data.frame_count);
      } else if (data.type === 'saved') {
        setRecording(false);
        setCurrentSequence(p => p + 1);
      } else if (data.type === 'error') {
        setError(data.message);
        setRecording(false);
      } else if (data.type === 'buffer_full') {
        ws.send(JSON.stringify({ action: 'stop' }));
      }
    };
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      ws.close();
    };
  }, [addToast]);

  useEffect(() => {
    if (!active || !wsRef.current) return;
    const id = window.setInterval(() => {
      const blob = captureFrame();
      if (blob && wsRef.current?.readyState === WebSocket.OPEN)
        blob.arrayBuffer().then(buf => wsRef.current?.send(buf));
    }, 1000 / 12);
    intervalRef.current = id;
    return () => clearInterval(id);
  }, [active, captureFrame]);

  useEffect(() => {
    if (!overlayRef.current || !landmarks || !active) return;
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!video) return;
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawLandmarks(ctx, landmarks, canvas.width, canvas.height, 480, 480);
  }, [landmarks, active, videoRef]);

  const startSingleRecording = useCallback(() => {
    setError(''); setFrameCount(0);
    setCountdown(3);
    setTimeout(() => setCountdown(2), 1000);
    setTimeout(() => setCountdown(1), 2000);
    setTimeout(() => { setCountdown(null); setWaitingForHand(true); }, 3000);
  }, []);

  useEffect(() => {
    if (!waitingForHand || !handVisible) return;
    setWaitingForHand(false);
    setRecording(true);
    wsRef.current?.send(JSON.stringify({ action: 'start', word: currentWord, contributor: contributor.trim() }));
  }, [waitingForHand, handVisible, currentWord, contributor]);

  const startBatch = () => {
    if (!currentWord) { setError('Choisissez un mot'); return; }
    if (!contributor.trim()) { setError('Entrez votre nom'); return; }
    setError(''); setBatchComplete(false); setBatchRunning(true);
    batchAbortRef.current = false; setCurrentSequence(0);
  };

  useEffect(() => {
    if (!batchRunning || batchAbortRef.current) return;
    if (currentSequence >= totalSequences) {
      setBatchRunning(false); setBatchComplete(true);
      addToast('success', `${totalSequences} templates pour "${currentWord}"`);
      api.getVocabulary().then(setWords).catch(() => {});
      if (teamTaskId) api.completeTask(parseInt(teamTaskId)).catch(() => {});
      return;
    }
    const delay = currentSequence === 0 ? 0 : 500;
    const t = setTimeout(() => { if (!batchAbortRef.current) startSingleRecording(); }, delay);
    return () => clearTimeout(t);
  }, [currentSequence, batchRunning, totalSequences, startSingleRecording, currentWord, addToast]);

  const cancelBatch = () => {
    batchAbortRef.current = true;
    wsRef.current?.send(JSON.stringify({ action: 'cancel' }));
    setRecording(false); setBatchRunning(false); setWaitingForHand(false);
    setFrameCount(0); setCountdown(null);
    addToast('info', `Arrêté après ${currentSequence}/${totalSequences}`);
    api.getVocabulary().then(setWords).catch(() => {});
  };

  const existingCount = words.find(w => w.name === currentWord)?.template_count ?? 0;

  const inputCls = 'text-sm px-2.5 py-1.5 bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-700 rounded-md placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 dark:focus:ring-offset-zinc-950 text-zinc-900 dark:text-zinc-50';

  return (
    <div className="h-full flex flex-col bg-white dark:bg-zinc-950">

      {/* ── Page header ──────────────────────────────── */}
      <div className="shrink-0 px-4 py-2.5 flex items-center gap-3 border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 flex-wrap">
        <span className="text-sm font-semibold shrink-0">Contribuer</span>
        <div className="w-px h-4 bg-zinc-200 dark:bg-zinc-800 shrink-0" />

        {/* Mot */}
        <select
          value={selectedWord}
          onChange={e => { setSelectedWord(e.target.value); setNewWord(''); }}
          className={`${inputCls} max-w-48`}
        >
          <option value="">— mot existant —</option>
          {words.map(w => (
            <option key={w.name} value={w.name}>{w.name} ({w.template_count})</option>
          ))}
        </select>
        <span className="text-zinc-400 text-sm shrink-0">ou</span>
        <input
          type="text"
          value={newWord}
          onChange={e => { setNewWord(e.target.value); setSelectedWord(''); }}
          placeholder="nouveau mot…"
          className={`${inputCls} w-32`}
        />
        {currentWord && (
          <span className="text-xs font-mono text-zinc-400 shrink-0 bg-zinc-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded">
            {existingCount} tpl
          </span>
        )}

        <div className="w-px h-4 bg-zinc-200 dark:bg-zinc-800 shrink-0" />

        {/* Contributeur */}
        <span className="text-xs uppercase tracking-wider text-zinc-400 shrink-0">par</span>
        <input
          type="text"
          value={contributor}
          onChange={e => setContributor(e.target.value)}
          placeholder="votre nom"
          className={`${inputCls} w-28`}
        />

        <div className="w-px h-4 bg-zinc-200 dark:bg-zinc-800 shrink-0" />

        {/* Nb séquences */}
        <input
          type="number" min={1} max={50}
          value={totalSequences}
          onChange={e => setTotalSequences(Math.max(1, Math.min(50, Number(e.target.value))))}
          disabled={batchRunning}
          className={`${inputCls} w-16 text-center disabled:opacity-40`}
        />
        <span className="text-xs text-zinc-400 shrink-0">séq.</span>

        {/* Action */}
        {!batchRunning ? (
          <button
            onClick={startBatch}
            disabled={!active || !currentWord}
            className="ml-auto px-4 py-1.5 text-sm font-medium bg-blue-600 hover:bg-blue-700 text-white rounded-md disabled:opacity-40 disabled:cursor-not-allowed transition-colors cursor-pointer shrink-0"
          >
            Enregistrer
          </button>
        ) : (
          <button
            onClick={cancelBatch}
            className="ml-auto px-4 py-1.5 text-sm font-medium bg-red-50 hover:bg-red-100 dark:bg-red-950/40 dark:hover:bg-red-950/60 text-red-600 dark:text-red-400 border border-red-200 dark:border-red-800 rounded-md transition-colors cursor-pointer shrink-0 flex items-center gap-1.5"
          >
            <Square className="w-3 h-3 fill-current" /> Arrêter
          </button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="shrink-0 px-4 py-2 border-b border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/30">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </div>
      )}

      {/* ── Main area ────────────────────────────────── */}
      <div className="flex-1 min-h-0 flex">

        {/* Batch progress sidebar */}
        {batchRunning && (
          <div className="shrink-0 w-52 border-r border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/50 p-5 flex flex-col gap-5">
            <div>
              <div className="flex justify-between text-xs font-mono mb-2">
                <span className="text-zinc-500 uppercase tracking-wider">Séquences</span>
                <span className="font-semibold text-zinc-900 dark:text-zinc-50">{currentSequence}/{totalSequences}</span>
              </div>
              <div className="h-1.5 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-300 rounded-full"
                  style={{ width: `${(currentSequence / totalSequences) * 100}%` }}
                />
              </div>
            </div>

            {recording && (
              <div>
                <div className="flex justify-between text-xs font-mono mb-2">
                  <span className="text-zinc-500 uppercase tracking-wider">Frames</span>
                  <span className="font-semibold text-zinc-900 dark:text-zinc-50">{frameCount}/{SEQUENCE_LENGTH}</span>
                </div>
                <div className="h-1 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-red-500 transition-all duration-100 rounded-full"
                    style={{ width: `${(frameCount / SEQUENCE_LENGTH) * 100}%` }}
                  />
                </div>
              </div>
            )}

            <div className="text-sm font-medium text-zinc-900 dark:text-zinc-50">
              {countdown !== null
                ? <span className="text-2xl font-bold text-blue-600 dark:text-blue-400">{countdown}</span>
                : waitingForHand ? 'Levez la main…'
                : recording ? <span className="flex items-center gap-1.5"><span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" /> Signez !</span>
                : <span className="text-zinc-400">Sauvegarde…</span>}
            </div>
          </div>
        )}

        {/* Camera (center, dominant) */}
        <div className="flex-1 min-h-0 flex items-center justify-center p-6 overflow-auto bg-zinc-50 dark:bg-zinc-900/20">
          <div className="relative max-w-lg w-full aspect-square rounded-xl overflow-hidden bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-sm">
            <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
            <canvas ref={overlayRef} className="absolute inset-0 w-full h-full pointer-events-none" />
            <canvas ref={canvasRef} className="hidden" />

            {/* Hand status */}
            {active && (
              <div className="absolute top-3 left-3 flex items-center gap-1.5 bg-black/60 backdrop-blur-sm px-2.5 py-1 rounded-md text-[11px] text-white font-mono">
                <span className={`w-1.5 h-1.5 rounded-full ${handVisible ? 'bg-emerald-400 animate-pulse' : 'bg-zinc-500'}`} />
                {handVisible ? 'main détectée' : 'pas de main'}
              </div>
            )}

            {/* REC indicator */}
            {recording && (
              <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-red-500/90 backdrop-blur-sm px-2.5 py-1 rounded-md text-[11px] text-white font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                REC
              </div>
            )}

            {/* Countdown overlay */}
            {countdown !== null && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm">
                <span className="text-8xl font-bold text-white font-mono animate-countdown">{countdown}</span>
              </div>
            )}

            {/* Waiting for hand */}
            {waitingForHand && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/40 backdrop-blur-sm">
                <p className="text-white text-xl font-semibold">Levez la main</p>
                <p className="text-white/60 text-sm mt-1">L'enregistrement démarre automatiquement</p>
              </div>
            )}

            {/* Start camera */}
            {!active && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/70 backdrop-blur-sm">
                <button
                  onClick={start}
                  className="flex items-center gap-2 px-5 py-2.5 bg-white text-zinc-900 text-sm font-semibold rounded-lg hover:bg-zinc-100 transition-colors cursor-pointer shadow-sm"
                >
                  <Video className="w-4 h-4" /> Activer la caméra
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Reference video sidebar */}
        <div className="shrink-0 w-56 border-l border-zinc-200 dark:border-zinc-800 flex flex-col bg-zinc-50 dark:bg-zinc-900/50">
          <div className="px-4 py-3 border-b border-zinc-200 dark:border-zinc-800">
            <span className="text-[10px] uppercase tracking-widest font-medium text-zinc-400">Référence</span>
          </div>

          {hasReference && referenceUrl ? (
            <div className="p-3 flex flex-col gap-3">
              <video
                key={referenceUrl}
                src={referenceUrl}
                autoPlay loop muted playsInline
                className="w-full aspect-square object-cover rounded-lg border border-zinc-200 dark:border-zinc-800"
              />
              <div className="flex items-center gap-1.5 text-xs text-zinc-500">
                <Play className="w-3 h-3" />
                <span className="font-mono font-medium text-zinc-700 dark:text-zinc-300">{currentWord}</span>
              </div>
              <p className="text-xs text-zinc-400">Reproduisez ce signe</p>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center p-4">
              <p className="text-xs text-zinc-400 text-center">
                {currentWord ? `Pas de vidéo pour "${currentWord}"` : 'Sélectionnez un mot'}
              </p>
            </div>
          )}

          {batchComplete && (
            <button
              onClick={() => { setBatchComplete(false); startBatch(); }}
              className="mx-3 mb-3 mt-auto px-3 py-2 text-xs font-medium border border-zinc-200 dark:border-zinc-700 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-md transition-colors cursor-pointer flex items-center justify-center gap-1.5"
            >
              <RotateCcw className="w-3 h-3" /> Encore {totalSequences}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
