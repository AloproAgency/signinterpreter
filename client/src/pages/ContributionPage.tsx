import { useState, useEffect, useRef, useCallback } from 'react';
import { useWebcam } from '../hooks/useWebcam';
import { useApp } from '../lib/context';
import { api } from '../lib/api';
import {
  Video, Circle, Square, CheckCircle, AlertCircle,
  Play, RotateCcw, ChevronRight,
} from 'lucide-react';
import type { WordInfo } from '../lib/types';

interface Landmarks {
  pose: { x: number; y: number; v: number }[];
  left_hand: { x: number; y: number }[];
  right_hand: { x: number; y: number }[];
}

// Hand connections for drawing
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
  landmarks: Landmarks,
  canvasW: number,
  canvasH: number,
  frameW: number,
  frameH: number,
) {
  const scaleX = canvasW / frameW;
  const scaleY = canvasH / frameH;

  // Draw pose connections
  ctx.strokeStyle = 'rgba(80, 44, 121, 0.6)';
  ctx.lineWidth = 2;
  for (const [a, b] of POSE_CONNECTIONS) {
    if (landmarks.pose[a] && landmarks.pose[b]) {
      ctx.beginPath();
      ctx.moveTo(landmarks.pose[a].x * scaleX, landmarks.pose[a].y * scaleY);
      ctx.lineTo(landmarks.pose[b].x * scaleX, landmarks.pose[b].y * scaleY);
      ctx.stroke();
    }
  }

  // Draw pose points
  ctx.fillStyle = 'rgba(80, 22, 10, 0.8)';
  for (const p of landmarks.pose) {
    ctx.beginPath();
    ctx.arc(p.x * scaleX, p.y * scaleY, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  // Draw hand connections + points
  const drawHand = (points: { x: number; y: number }[], color: string) => {
    if (points.length === 0) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    for (const [a, b] of HAND_CONNECTIONS) {
      if (points[a] && points[b]) {
        ctx.beginPath();
        ctx.moveTo(points[a].x * scaleX, points[a].y * scaleY);
        ctx.lineTo(points[b].x * scaleX, points[b].y * scaleY);
        ctx.stroke();
      }
    }
    ctx.fillStyle = color;
    for (const p of points) {
      ctx.beginPath();
      ctx.arc(p.x * scaleX, p.y * scaleY, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  drawHand(landmarks.left_hand, 'rgba(121, 44, 250, 0.9)');
  drawHand(landmarks.right_hand, 'rgba(245, 66, 230, 0.9)');
}

export default function ContributionPage() {
  const { videoRef, canvasRef, active, start, captureFrame } = useWebcam();
  const { addToast } = useApp();

  const [words, setWords] = useState<WordInfo[]>([]);
  const [selectedWord, setSelectedWord] = useState(() => localStorage.getItem('prefill_word') || '');
  const [newWord, setNewWord] = useState('');
  const [contributor, setContributor] = useState(() => localStorage.getItem('contributor') || '');
  const [teamTaskId] = useState(() => localStorage.getItem('prefill_task_id') || '');

  // Read prefill count and clean up localStorage
  const [totalSequences, setTotalSequences] = useState(() => {
    const prefill = localStorage.getItem('prefill_count');
    // Clean up prefill values
    localStorage.removeItem('prefill_word');
    localStorage.removeItem('prefill_task_id');
    localStorage.removeItem('prefill_count');
    return prefill ? parseInt(prefill) : 10;
  });

  const [recording, setRecording] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [handVisible, setHandVisible] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState('');
  const [countdown, setCountdown] = useState<number | null>(null);
  const [waitingForHand, setWaitingForHand] = useState(false);

  // Batch recording
  const [currentSequence, setCurrentSequence] = useState(0);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchComplete, setBatchComplete] = useState(false);
  const batchAbortRef = useRef(false);

  // Reference video
  const [hasReference, setHasReference] = useState(false);
  const [referenceUrl, setReferenceUrl] = useState('');
  const videoRefElement = useRef<HTMLVideoElement>(null);

  // Landmarks overlay
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [landmarks, setLandmarks] = useState<Landmarks | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<number | null>(null);

  // Load word list
  useEffect(() => {
    api.getVocabulary().then(setWords).catch(() => {});
  }, []);

  // Save contributor name
  useEffect(() => {
    if (contributor) localStorage.setItem('contributor', contributor);
  }, [contributor]);

  // Check for reference video when word changes
  const currentWord = newWord.trim() || selectedWord;
  useEffect(() => {
    if (!currentWord) {
      setHasReference(false);
      setReferenceUrl('');
      return;
    }
    api.getReference(currentWord).then(data => {
      setHasReference(data.exists);
      if (data.exists) {
        setReferenceUrl(api.getReferenceVideoUrl(currentWord));
      }
    }).catch(() => setHasReference(false));
  }, [currentWord]);

  // Connect WebSocket
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/collect`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'preview' || data.type === 'recording') {
        setHandVisible(data.hand_visible);
        if (data.landmarks) setLandmarks(data.landmarks);
        if (data.type === 'recording') {
          setFrameCount(data.frame_count);
        }
      } else if (data.type === 'saved') {
        setRecording(false);
        // In batch mode, trigger next sequence
        setCurrentSequence(prev => {
          const next = prev + 1;
          return next;
        });
      } else if (data.type === 'error') {
        setError(data.message);
        setRecording(false);
      } else if (data.type === 'buffer_full') {
        ws.send(JSON.stringify({ action: 'stop' }));
      }
    };

    ws.onclose = () => {
      // Reconnect
      setTimeout(() => {}, 2000);
    };

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      ws.close();
    };
  }, [addToast]);

  // Send frames continuously when webcam is active (for landmark preview)
  useEffect(() => {
    if (!active || !wsRef.current) return;
    const interval = window.setInterval(() => {
      const blob = captureFrame();
      if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
        blob.arrayBuffer().then(buf => wsRef.current?.send(buf));
      }
    }, 1000 / 12); // 12 FPS for preview
    intervalRef.current = interval;
    return () => clearInterval(interval);
  }, [active, captureFrame]);

  // Draw landmarks on overlay canvas
  useEffect(() => {
    if (!overlayRef.current || !landmarks || !active) return;
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!video) return;

    // Match canvas size to video display size
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // The frame sent to server is 480x480 (square crop)
    drawLandmarks(ctx, landmarks, canvas.width, canvas.height, 480, 480);
  }, [landmarks, active, videoRef]);

  // Start a single recording: countdown → wait for hand → record
  const startSingleRecording = useCallback(() => {
    setError('');
    setFrameCount(0);

    setCountdown(3);
    setTimeout(() => setCountdown(2), 1000);
    setTimeout(() => setCountdown(1), 2000);
    setTimeout(() => {
      setCountdown(null);
      // After countdown, wait for hand to be visible
      setWaitingForHand(true);
    }, 3000);
  }, []);

  // When waitingForHand is true and hand becomes visible, start recording
  useEffect(() => {
    if (!waitingForHand || !handVisible) return;
    setWaitingForHand(false);
    setRecording(true);
    wsRef.current?.send(JSON.stringify({
      action: 'start',
      word: currentWord,
      contributor: contributor.trim(),
    }));
  }, [waitingForHand, handVisible, currentWord, contributor]);

  // Start batch recording
  const startBatch = () => {
    if (!currentWord) { setError('Choisissez un mot'); return; }
    if (!contributor.trim()) { setError('Entrez votre nom'); return; }
    if (totalSequences < 1) { setError('Au moins 1 sequence'); return; }

    setError('');
    setSaved(false);
    setBatchComplete(false);
    setBatchRunning(true);
    batchAbortRef.current = false;
    setCurrentSequence(0);
  };

  // Auto-chain: when currentSequence changes and batch is running, start next recording
  useEffect(() => {
    if (!batchRunning || batchAbortRef.current) return;

    if (currentSequence >= totalSequences) {
      // Batch complete
      setBatchRunning(false);
      setBatchComplete(true);
      setSaved(true);
      addToast('success', `${totalSequences} templates enregistres pour "${currentWord}"`);
      api.getVocabulary().then(setWords).catch(() => {});
      // Mark team task as done if coming from team page
      if (teamTaskId) {
        api.completeTask(parseInt(teamTaskId)).catch(() => {});
      }
      return;
    }

    // Small delay between sequences
    const delay = currentSequence === 0 ? 0 : 500;
    const timer = setTimeout(() => {
      if (!batchAbortRef.current) {
        startSingleRecording();
      }
    }, delay);
    return () => clearTimeout(timer);
  }, [currentSequence, batchRunning, totalSequences, startSingleRecording, currentWord, addToast]);

  const stopRecording = () => {
    wsRef.current?.send(JSON.stringify({ action: 'stop' }));
  };

  const cancelBatch = () => {
    batchAbortRef.current = true;
    wsRef.current?.send(JSON.stringify({ action: 'cancel' }));
    setRecording(false);
    setBatchRunning(false);
    setWaitingForHand(false);
    setFrameCount(0);
    setCountdown(null);
    addToast('info', `Arrete apres ${currentSequence}/${totalSequences} sequences`);
    api.getVocabulary().then(setWords).catch(() => {});
  };

  const existingCount = words.find(w => w.name === currentWord)?.template_count || 0;

  return (
    <div className="h-full p-4 md:p-6 overflow-auto bg-white dark:bg-[#0d1117]">
      <h1 className="text-2xl font-bold tracking-tight mb-6 text-[#1f2328] dark:text-[#e6edf3]">
        Contribuer au vocabulaire
      </h1>

      <div className="flex flex-col lg:flex-row gap-6">

        {/* Left column: controls */}
        <div className="w-full lg:w-72 space-y-4 shrink-0">

          {/* Word selection */}
          <div className="rounded-lg border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22] p-4">
            <label className="text-sm font-medium mb-2 block text-[#1f2328] dark:text-[#e6edf3]">Mot</label>
            <select
              value={selectedWord}
              onChange={e => { setSelectedWord(e.target.value); setNewWord(''); }}
              className="w-full rounded-md px-3 py-2 text-sm mb-3 bg-white dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] text-[#1f2328] dark:text-[#e6edf3]"
            >
              <option value="">-- Existant --</option>
              {words.map(w => (
                <option key={w.name} value={w.name}>
                  {w.name} ({w.template_count})
                </option>
              ))}
            </select>

            <div className="flex items-center gap-2 my-2">
              <div className="h-px flex-1 bg-[#d0d7de] dark:bg-[#30363d]" />
              <span className="text-sm text-[#8b949e]">ou</span>
              <div className="h-px flex-1 bg-[#d0d7de] dark:bg-[#30363d]" />
            </div>

            <input
              type="text"
              value={newWord}
              onChange={e => { setNewWord(e.target.value); setSelectedWord(''); }}
              placeholder="Nouveau mot..."
              className="w-full rounded-md px-3 py-2 text-sm bg-white dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] text-[#1f2328] dark:text-[#e6edf3] placeholder-[#8b949e]"
            />

            {currentWord && (
              <p className="mt-2 text-sm text-[#8b949e]">
                <span className="font-mono font-bold text-[#1396ba]">{existingCount}</span> templates existants
              </p>
            )}
          </div>

          {/* Contributor */}
          <div className="rounded-lg border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22] p-4">
            <label className="text-sm font-medium mb-2 block text-[#1f2328] dark:text-[#e6edf3]">Contributeur</label>
            <input
              type="text"
              value={contributor}
              onChange={e => setContributor(e.target.value)}
              placeholder="Votre nom"
              className="w-full rounded-md px-3 py-2 text-sm bg-white dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] text-[#1f2328] dark:text-[#e6edf3] placeholder-[#8b949e]"
            />
          </div>

          {/* Number of sequences */}
          <div className="rounded-lg border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22] p-4">
            <label className="text-sm font-medium mb-2 block text-[#1f2328] dark:text-[#e6edf3]">Nombre de sequences</label>
            <input
              type="number"
              min={1}
              max={50}
              value={totalSequences}
              onChange={e => setTotalSequences(Math.max(1, Math.min(50, Number(e.target.value))))}
              disabled={batchRunning}
              className="w-full rounded-md px-3 py-2 text-sm bg-white dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] text-[#1f2328] dark:text-[#e6edf3] disabled:opacity-50"
            />
          </div>

          {/* Actions */}
          <div className="space-y-2">
            {!batchRunning ? (
              <button
                onClick={startBatch}
                disabled={!active || !currentWord}
                className="w-full py-2.5 rounded-md text-sm font-medium bg-[#1396ba] hover:bg-[#17b8e3] text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors cursor-pointer flex items-center justify-center gap-2"
              >
                <Circle className="w-4 h-4" />
                Enregistrer {totalSequences} sequence{totalSequences > 1 ? 's' : ''}
              </button>
            ) : (
              <button
                onClick={cancelBatch}
                className="w-full py-2.5 rounded-md text-sm font-medium border border-[#ef4444]/30 text-[#ef4444] hover:bg-[#ef4444]/10 transition-colors cursor-pointer flex items-center justify-center gap-2"
              >
                <Square className="w-4 h-4" />
                Arreter le batch
              </button>
            )}
          </div>

          {/* Batch + frame progress */}
          {batchRunning && (
            <div className="rounded-lg border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22] p-4 space-y-3">
              {/* Global batch progress */}
              <div>
                <div className="flex justify-between text-sm mb-1.5">
                  <span className="text-[#656d76] dark:text-[#8b949e]">Sequences</span>
                  <span className="font-mono font-bold text-[#1396ba]">{currentSequence}/{totalSequences}</span>
                </div>
                <div className="h-3 rounded-full bg-[#d0d7de] dark:bg-[#30363d] overflow-hidden">
                  <div
                    className="h-full rounded-full bg-[#1396ba] transition-all duration-300"
                    style={{ width: `${(currentSequence / totalSequences) * 100}%` }}
                  />
                </div>
              </div>

              {/* Current frame progress */}
              {recording && (
                <div>
                  <div className="flex justify-between text-sm mb-1.5">
                    <span className="text-[#656d76] dark:text-[#8b949e]">Frames</span>
                    <span className="font-mono text-sm text-[#8b949e]">{frameCount}/{SEQUENCE_LENGTH}</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-[#d0d7de] dark:bg-[#30363d] overflow-hidden">
                    <div
                      className="h-full rounded-full bg-[#1396ba]/60 transition-all duration-100"
                      style={{ width: `${(frameCount / SEQUENCE_LENGTH) * 100}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Status text */}
              <p className="text-sm text-[#8b949e]">
                {countdown !== null
                  ? `Preparez-vous... ${countdown}`
                  : waitingForHand
                  ? 'Levez la main pour commencer...'
                  : recording
                  ? 'Signez maintenant !'
                  : 'Sauvegarde en cours...'}
              </p>
            </div>
          )}

          {/* Status messages */}
          {saved && (
            <div className="flex items-center gap-2 p-3 rounded-lg border border-[#10b981]/20 bg-[#10b981]/5 text-[#10b981] text-sm">
              <CheckCircle className="w-4 h-4 shrink-0" />
              Template sauvegarde !
            </div>
          )}
          {error && (
            <div className="flex items-center gap-2 p-3 rounded-lg border border-[#ef4444]/20 bg-[#ef4444]/5 text-[#ef4444] text-sm">
              <AlertCircle className="w-4 h-4 shrink-0" />
              {error}
            </div>
          )}
        </div>

        {/* Center: webcam with landmarks */}
        <div className="flex-1 flex flex-col items-center">
          <div className="relative rounded-lg overflow-hidden max-w-lg w-full border border-[#d0d7de] dark:border-[#30363d]">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full aspect-square object-cover"
            />
            {/* Landmark overlay canvas */}
            <canvas
              ref={overlayRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />
            <canvas ref={canvasRef} className="hidden" />

            {/* Hand detection indicator */}
            {active && (
              <div className={`absolute top-3 left-3 flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium ${
                handVisible
                  ? 'bg-[#10b981]/80 text-white'
                  : 'bg-black/50 text-white/70'
              }`}>
                <div className={`w-2 h-2 rounded-full ${handVisible ? 'bg-white' : 'bg-white/40'}`} />
                {handVisible ? 'Main detectee' : 'Pas de main'}
              </div>
            )}

            {/* Recording indicator */}
            {recording && (
              <div className="absolute top-3 right-3 flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#ef4444]/80 text-white text-sm font-bold">
                <div className="w-2 h-2 rounded-full bg-white animate-pulse" />
                REC
              </div>
            )}

            {/* Countdown overlay */}
            {countdown !== null && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                <span className="text-8xl font-bold text-white drop-shadow-lg">{countdown}</span>
              </div>
            )}

            {/* Waiting for hand overlay */}
            {waitingForHand && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/40">
                <span className="text-2xl font-bold text-white mb-2">Levez la main</span>
                <span className="text-base text-white/60">L'enregistrement demarre automatiquement</span>
              </div>
            )}

            {/* Activate camera */}
            {!active && (
              <div className="absolute inset-0 flex items-center justify-center bg-[#0d1117]/80">
                <button
                  onClick={start}
                  className="px-6 py-3 rounded-lg bg-[#1396ba] hover:bg-[#17b8e3] text-white font-medium flex items-center gap-2 transition-colors cursor-pointer"
                >
                  <Video className="w-5 h-5" />
                  Activer la camera
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Right: reference video */}
        <div className="w-full lg:w-72 shrink-0">
          <div className="rounded-lg border border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22] overflow-hidden">
            <div className="px-4 py-3 border-b border-[#d0d7de] dark:border-[#30363d]">
              <h3 className="text-sm font-medium text-[#1f2328] dark:text-[#e6edf3] flex items-center gap-2">
                <Play className="w-4 h-4 text-[#1396ba]" />
                Exemple de reference
              </h3>
            </div>

            {hasReference && referenceUrl ? (
              <div className="p-2">
                <video
                  key={referenceUrl}
                  src={referenceUrl}
                  autoPlay
                  loop
                  muted
                  playsInline
                  className="w-full aspect-square object-cover rounded-md"
                />
                <p className="text-sm text-center mt-2 text-[#8b949e]">
                  Reproduisez ce signe
                </p>
              </div>
            ) : currentWord ? (
              <div className="p-8 text-center">
                <p className="text-sm text-[#8b949e]">
                  Pas de video de reference pour "{currentWord}"
                </p>
              </div>
            ) : (
              <div className="p-8 text-center">
                <p className="text-sm text-[#8b949e]">
                  Selectionnez un mot pour voir l'exemple
                </p>
              </div>
            )}
          </div>

          {/* Quick record again */}
          {batchComplete && (
            <button
              onClick={() => { setSaved(false); setBatchComplete(false); startBatch(); }}
              className="w-full mt-3 py-2.5 rounded-md text-sm font-medium border border-[#1396ba]/30 text-[#1396ba] hover:bg-[#1396ba]/10 transition-colors cursor-pointer flex items-center justify-center gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              Encore {totalSequences} sequences
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

const SEQUENCE_LENGTH = 30;
