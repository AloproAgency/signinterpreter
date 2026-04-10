import { useEffect, useState, useCallback, useMemo } from 'react';
import { useWebcam } from '../hooks/useWebcam';
import { useInference } from '../hooks/useInference';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';
import { useApp } from '../lib/context';
import {
  Video, Wifi, WifiOff, Trash2, Hand, Activity,
  Copy, Settings2, Maximize, Minimize,
  ChevronRight, Zap,
} from 'lucide-react';

export default function InferencePage() {
  const { videoRef, canvasRef, active, start, captureFrame } = useWebcam();
  const {
    connected, status, prediction, sentence, lastAddedIndex,
    connect, disconnect, clearSentence, setThreshold,
  } = useInference(captureFrame, active);
  const { addToast } = useApp();

  const [showSettings, setShowSettings] = useState(false);
  const [threshold, setThresholdLocal] = useState(50);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  const copySentence = useCallback(() => {
    if (sentence.length === 0) return;
    navigator.clipboard.writeText(sentence.join(' '));
    addToast('success', 'Phrase copiee dans le presse-papier');
  }, [sentence, addToast]);

  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  const shortcuts = useMemo(() => ({
    ' ': clearSentence,
    'f': toggleFullscreen,
  }), [clearSentence, toggleFullscreen]);

  useKeyboardShortcuts(shortcuts);

  const handleThresholdChange = (value: number) => {
    setThresholdLocal(value);
    setThreshold(value);
  };

  return (
    <div className="h-full flex flex-col relative">
      {/* Sentence bar */}
      <div className="border-b border-[#d0d7de] dark:border-[#30363d] px-4 md:px-6 py-3 flex items-center gap-3 bg-[#f6f8fa] dark:bg-[#161b22]">
        <div className="flex-1 min-h-[36px] flex items-center">
          {sentence.length > 0 ? (
            <div className="flex gap-2 flex-wrap">
              {sentence.map((word, i) => (
                <span
                  key={`${i}-${word}`}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium border border-[rgba(19,150,186,0.15)] bg-[rgba(19,150,186,0.1)] text-[#1396ba] ${
                    i === lastAddedIndex ? 'animate-word-pop' : ''
                  }`}
                >
                  {word}
                </span>
              ))}
            </div>
          ) : (
            <p className="text-sm flex items-center gap-2 text-[#8b949e] dark:text-[#484f58]">
              <ChevronRight className="w-3.5 h-3.5 text-[#1396ba]/50" />
              <span>Signez pour commencer...</span>
            </p>
          )}
        </div>

        <div className="flex items-center gap-1 shrink-0">
          {sentence.length > 0 && (
            <button
              onClick={copySentence}
              className="p-2 rounded-md transition-colors text-[#656d76] dark:text-[#8b949e] hover:text-[#1396ba] hover:bg-[rgba(19,150,186,0.1)]"
              title="Copier"
            >
              <Copy className="w-3.5 h-3.5" />
            </button>
          )}
          <button
            onClick={clearSentence}
            className="p-2 rounded-md transition-colors text-[#656d76] dark:text-[#8b949e] hover:bg-[#f6f8fa] dark:hover:bg-[#1c2333] hover:text-[#1f2328] dark:hover:text-[#e6edf3]"
            title="Effacer (Espace)"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`p-2 rounded-md transition-colors ${
              showSettings
                ? 'text-[#1396ba] bg-[rgba(19,150,186,0.1)]'
                : 'text-[#656d76] dark:text-[#8b949e] hover:bg-[#f6f8fa] dark:hover:bg-[#1c2333] hover:text-[#1f2328] dark:hover:text-[#e6edf3]'
            }`}
            title="Parametres"
          >
            <Settings2 className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={toggleFullscreen}
            className="p-2 rounded-md transition-colors hidden md:block text-[#656d76] dark:text-[#8b949e] hover:bg-[#f6f8fa] dark:hover:bg-[#1c2333] hover:text-[#1f2328] dark:hover:text-[#e6edf3]"
            title="Plein ecran (F)"
          >
            {isFullscreen ? <Minimize className="w-3.5 h-3.5" /> : <Maximize className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Settings popup (floating, positioned under the settings button) */}
      {showSettings && (
        <>
          {/* Backdrop to close on click outside */}
          <div className="fixed inset-0 z-40" onClick={() => setShowSettings(false)} />
          <div className="absolute right-16 top-12 z-50 w-80 rounded-lg border border-[#d0d7de] dark:border-[#30363d] bg-white dark:bg-[#1c2333] shadow-xl animate-scale-in">
            <div className="p-4">
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-medium text-[#1f2328] dark:text-[#e6edf3]">Seuil de confiance</label>
                <span className="text-base font-mono font-bold text-[#1396ba]">{threshold}</span>
              </div>
              <input
                type="range"
                min={10}
                max={300}
                value={threshold}
                onChange={e => handleThresholdChange(Number(e.target.value))}
                className="w-full h-2 rounded-full appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, #1396ba 0%, #1396ba ${(threshold - 10) / 290 * 100}%, ${
                    document.documentElement.classList.contains('dark') ? '#30363d' : '#d0d7de'
                  } ${(threshold - 10) / 290 * 100}%, ${
                    document.documentElement.classList.contains('dark') ? '#30363d' : '#d0d7de'
                  } 100%)`
                }}
              />
              <div className="flex justify-between mt-2 text-sm text-[#8b949e]">
                <span>10</span>
                <span>300</span>
              </div>
            </div>
            <div className="border-t border-[#d0d7de] dark:border-[#30363d] px-4 py-2.5 flex gap-4 text-sm text-[#8b949e]">
              <span><kbd className="px-1.5 py-0.5 rounded bg-[#f6f8fa] dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] text-sm font-mono">⎵</kbd> Effacer</span>
              <span><kbd className="px-1.5 py-0.5 rounded bg-[#f6f8fa] dark:bg-[#0d1117] border border-[#d0d7de] dark:border-[#30363d] text-sm font-mono">F</kbd> Plein ecran</span>
            </div>
          </div>
        </>
      )}

      {/* Main content */}
      <div className="flex-1 flex flex-col lg:flex-row min-h-0">
        {/* Webcam panel */}
        <div className="flex-1 p-4 md:p-6 flex flex-col items-center justify-center">
          <div className={`relative rounded-lg overflow-hidden max-w-xl w-full border ${
            active && status.is_signing
              ? 'border-[#1396ba] border-2'
              : 'border-[#d0d7de] dark:border-[#30363d]'
          }`}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full aspect-square object-cover"
            />
            <canvas ref={canvasRef} className="hidden" />

            {/* Overlays when active */}
            {active && (
              <>
                {/* Hand detection pill */}
                <div className="absolute top-3 left-3">
                  <div className="flex items-center gap-1.5 bg-black/60 px-2.5 py-1 rounded-md text-sm font-medium text-white/90">
                    <div className={`w-1.5 h-1.5 rounded-full ${
                      status.hand_visible ? 'bg-[#1396ba]' : 'bg-[#484f58]'
                    }`} />
                    {status.hand_visible ? 'Main detectee' : 'Pas de main'}
                  </div>
                </div>

                {/* Signing state badge */}
                {status.is_signing && (
                  <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-black/60 px-2.5 py-1 rounded-md">
                    <div className="w-1.5 h-1.5 rounded-full bg-[#1396ba] animate-pulse" />
                    <span className="text-[#17b8e3] text-sm font-bold tracking-wider">REC</span>
                  </div>
                )}

                {/* Motion bar at bottom */}
                <div className="absolute bottom-0 left-0 right-0 bg-black/60 px-3 py-2">
                  <div className="flex items-center gap-2 mb-1">
                    <Activity className="w-3 h-3 text-[#8b949e]" />
                    <span className="text-sm text-[#8b949e] uppercase tracking-wider font-medium">Mouvement</span>
                    <Zap className={`w-3 h-3 ml-auto ${
                      status.motion_energy > 0.05 ? 'text-[#1396ba]' : 'text-[#484f58]'
                    }`} />
                  </div>
                  <div className="h-1 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-[#1396ba] transition-all duration-150 ease-out"
                      style={{ width: `${Math.min(100, status.motion_energy * 1000)}%` }}
                    />
                  </div>
                </div>
              </>
            )}

            {/* Start button overlay */}
            {!active && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/60">
                <button
                  onClick={start}
                  className="bg-[#1396ba] hover:bg-[#17b8e3] text-white px-6 py-3 rounded-md font-medium text-sm flex items-center gap-2.5 transition-colors cursor-pointer"
                >
                  <Video className="w-5 h-5" />
                  Activer la camera
                </button>
              </div>
            )}
          </div>

          {/* Connection status */}
          <div className="mt-4 flex items-center gap-3">
            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-sm font-medium border ${
              connected
                ? 'text-[#1396ba] border-[rgba(19,150,186,0.15)] bg-[rgba(19,150,186,0.1)]'
                : 'text-[#8b949e] dark:text-[#484f58] border-[#d0d7de] dark:border-[#30363d] bg-[#f6f8fa] dark:bg-[#161b22]'
            }`}>
              {connected ? (
                <>
                  <Wifi className="w-3 h-3" />
                  <div className="w-1.5 h-1.5 rounded-full bg-[#10b981]" />
                  Connecte
                </>
              ) : (
                <><WifiOff className="w-3 h-3" />Reconnexion...</>
              )}
            </div>
            {prediction && (
              <span className="text-sm font-mono font-bold text-[#8b949e] dark:text-[#484f58]">
                {prediction.inference_ms}ms
              </span>
            )}
          </div>
        </div>

        {/* Prediction panel */}
        <div className="w-full lg:w-80 xl:w-96 border-t lg:border-t-0 lg:border-l border-[#d0d7de] dark:border-[#30363d] p-4 md:p-5 flex flex-col overflow-auto bg-[#f6f8fa] dark:bg-[#161b22]">
          <h2 className="text-sm font-semibold mb-4 flex items-center gap-2 uppercase tracking-wider text-[#656d76] dark:text-[#8b949e]">
            <Hand className="w-3.5 h-3.5 text-[#1396ba]" />
            Predictions
          </h2>

          {prediction ? (
            <div className="space-y-5 flex-1 animate-fade-in">
              {/* Best prediction */}
              <div className={`p-4 rounded-lg border ${
                prediction.is_final && prediction.confidence > 0
                  ? 'bg-[rgba(19,150,186,0.1)] border-[rgba(19,150,186,0.15)]'
                  : 'bg-white dark:bg-[#0d1117] border-[#d0d7de] dark:border-[#30363d]'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-3xl font-bold tracking-tight text-[#1f2328] dark:text-[#e6edf3]">{prediction.word}</p>
                  {prediction.is_final && prediction.confidence > 0 && (
                    <span className="text-sm font-bold uppercase tracking-wider px-2 py-0.5 rounded-md bg-[rgba(19,150,186,0.1)] text-[#1396ba]">
                      Final
                    </span>
                  )}
                </div>

                {/* Confidence bar */}
                <div className="flex items-center gap-3">
                  <div className="flex-1 h-1.5 rounded-full overflow-hidden bg-[#d0d7de] dark:bg-[#30363d]">
                    <div
                      className="h-full rounded-full bg-[#1396ba] transition-all duration-500 ease-out"
                      style={{ width: `${Math.max(0, prediction.confidence * 100)}%` }}
                    />
                  </div>
                  <span className={`text-sm font-mono font-bold w-10 text-right ${
                    prediction.confidence > 0.7 ? 'text-[#1396ba]' : 'text-[#8b949e]'
                  }`}>
                    {Math.round(prediction.confidence * 100)}%
                  </span>
                </div>

                <p className="text-sm mt-2 font-mono text-[#8b949e] dark:text-[#484f58]">
                  dist={prediction.distance.toFixed(1)} | seuil={prediction.threshold.toFixed(1)}
                </p>
              </div>

              {/* Top-K list */}
              {prediction.top_k.length > 0 && (
                <div>
                  <p className="text-sm mb-3 uppercase tracking-wider font-semibold text-[#656d76] dark:text-[#8b949e]">
                    Top {prediction.top_k.length} candidats
                  </p>
                  <div className="space-y-1">
                    {prediction.top_k.map((item, i) => {
                      const maxDist = prediction.top_k[prediction.top_k.length - 1]?.distance || 1;
                      const pct = Math.max(5, 100 - (item.distance / maxDist) * 80);
                      return (
                        <div
                          key={i}
                          className={`flex items-center gap-3 px-3 py-2 rounded-lg ${
                            i === 0
                              ? 'bg-[rgba(19,150,186,0.1)] border border-[rgba(19,150,186,0.15)]'
                              : ''
                          }`}
                        >
                          <span className={`text-sm w-4 font-mono font-bold ${
                            i === 0 ? 'text-[#1396ba]' : 'text-[#8b949e] dark:text-[#484f58]'
                          }`}>{i + 1}</span>
                          <span className={`text-sm flex-1 truncate ${
                            i === 0
                              ? 'text-[#1f2328] dark:text-[#e6edf3] font-semibold'
                              : 'text-[#656d76] dark:text-[#8b949e]'
                          }`}>
                            {item.word}
                          </span>
                          <div className="w-20 h-1 rounded-full overflow-hidden bg-[#d0d7de] dark:bg-[#30363d]">
                            <div
                              className="h-full rounded-full transition-all duration-500 ease-out"
                              style={{
                                width: `${pct}%`,
                                backgroundColor: i === 0 ? '#1396ba' : '#8b949e',
                                opacity: i === 0 ? 1 : 0.3,
                              }}
                            />
                          </div>
                          <span className="text-sm font-mono font-bold w-8 text-right text-[#8b949e] dark:text-[#484f58]">
                            {item.distance.toFixed(0)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center space-y-3">
                <div className="w-14 h-14 rounded-lg flex items-center justify-center mx-auto bg-[rgba(19,150,186,0.1)] border border-[rgba(19,150,186,0.15)]">
                  <Hand className="w-6 h-6 text-[#1396ba]/50" />
                </div>
                <p className="text-sm text-[#8b949e] dark:text-[#484f58]">
                  Signez devant la camera<br />pour voir les predictions
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
