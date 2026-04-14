import { useState, useRef, useCallback, useEffect } from 'react';
import type { Prediction, InferenceStatus } from '../lib/types';

/**
 * Inference hook — sends extracted features (not images) to the server.
 * MediaPipe runs client-side; only 171 floats per frame are sent via WebSocket.
 */
export function useInference(
  processFrame: ((video: HTMLVideoElement) => { features: number[]; handVisible: boolean } | null) | null,
  videoRef: React.RefObject<HTMLVideoElement | null>,
  active: boolean,
  mediaPipeReady: boolean,
) {
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<number | null>(null);
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<InferenceStatus>({
    hand_visible: false, is_signing: false, buffer_length: 0, motion_energy: 0,
  });
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [sentence, setSentence] = useState<string[]>([]);
  const [translated, setTranslated] = useState('');
  const [translatedScore, setTranslatedScore] = useState(0);
  const [phrases, setPhrases] = useState<string[]>([]);
  const [phraseSigns, setPhraseSigns] = useState<string[][]>([]);
  const [phraseScores, setPhraseScores] = useState<number[]>([]);
  const [lastAddedIndex, setLastAddedIndex] = useState(-1);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/inference`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      setTimeout(connect, 2000);
    };
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'status') setStatus(data);
      else if (data.type === 'prediction') setPrediction(data);
      else if (data.type === 'sentence_update') {
        setSentence(prev => {
          if (data.sentence.length > prev.length) {
            setLastAddedIndex(data.sentence.length - 1);
          }
          return data.sentence;
        });
        if (data.translated !== undefined) setTranslated(data.translated);
        if (data.translated_score !== undefined) setTranslatedScore(data.translated_score);
        if (data.phrases !== undefined) setPhrases(data.phrases);
        if (data.phrase_signs !== undefined) setPhraseSigns(data.phrase_signs);
        if (data.phrase_scores !== undefined) setPhraseScores(data.phrase_scores);
      }
    };
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  // Process frames locally with MediaPipe and send features to server
  useEffect(() => {
    if (active && connected && mediaPipeReady && processFrame && videoRef.current) {
      intervalRef.current = window.setInterval(() => {
        if (!videoRef.current || !processFrame) return;
        const result = processFrame(videoRef.current);
        if (result && wsRef.current?.readyState === WebSocket.OPEN) {
          // Send features as JSON (171 floats ≈ 1.5 KB instead of 50 KB JPEG)
          wsRef.current.send(JSON.stringify({
            type: 'features',
            features: result.features,
            hand_visible: result.handVisible,
          }));
        }
      }, 1000 / 30);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [active, connected, mediaPipeReady, processFrame, videoRef]);

  const clearSentence = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'clear_sentence' }));
    }
    setSentence([]);
    setTranslated('');
    setTranslatedScore(0);
    setPhrases([]);
    setPhraseSigns([]);
    setPhraseScores([]);
    setLastAddedIndex(-1);
  }, []);

  const setThreshold = useCallback((value: number) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'set_threshold', value }));
    }
  }, []);

  const finalizeSentence = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'finalize_sentence' }));
    }
  }, []);

  return {
    connected, status, prediction, sentence, translated, translatedScore,
    phrases, phraseSigns, phraseScores, lastAddedIndex,
    connect, disconnect, clearSentence, setThreshold, finalizeSentence,
  };
}
