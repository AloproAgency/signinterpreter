import { useState, useRef, useCallback, useEffect } from 'react';
import type { Prediction, InferenceStatus } from '../lib/types';

export function useInference(captureFrame: () => Blob | null, active: boolean) {
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<number | null>(null);
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<InferenceStatus>({
    hand_visible: false, is_signing: false, buffer_length: 0, motion_energy: 0,
  });
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [sentence, setSentence] = useState<string[]>([]);
  const [translated, setTranslated] = useState('');
  const [phrases, setPhrases] = useState<string[]>([]);
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
        if (data.phrases !== undefined) setPhrases(data.phrases);
      }
    };
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (active && connected) {
      intervalRef.current = window.setInterval(() => {
        const blob = captureFrame();
        if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
          blob.arrayBuffer().then(buf => wsRef.current?.send(buf));
        }
      }, 1000 / 8); // 8 FPS (reduced for network latency)
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [active, connected, captureFrame]);

  const clearSentence = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'clear_sentence' }));
    }
    setSentence([]);
    setTranslated('');
    setPhrases([]);
    setLastAddedIndex(-1);
  }, []);

  const setThreshold = useCallback((value: number) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'set_threshold', value }));
    }
  }, []);

  return {
    connected, status, prediction, sentence, translated, phrases, lastAddedIndex,
    connect, disconnect, clearSentence, setThreshold,
  };
}
