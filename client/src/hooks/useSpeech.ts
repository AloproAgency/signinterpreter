/**
 * Browser-native Text-To-Speech helper.
 * Picks a French voice if one is available, exposes a `speak(text)` API and
 * a persistent enabled-flag stored in localStorage.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

const STORAGE_KEY = 'si_speech_enabled';

export function useSpeech() {
  const [enabled, setEnabledState] = useState<boolean>(() => {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === null ? true : v === '1';
  });
  const [available] = useState<boolean>(
    typeof window !== 'undefined' && 'speechSynthesis' in window,
  );
  const voiceRef = useRef<SpeechSynthesisVoice | null>(null);

  // Resolve the best French voice once the list is populated.
  useEffect(() => {
    if (!available) return;
    const pickVoice = () => {
      const voices = window.speechSynthesis.getVoices();
      if (!voices.length) return;
      const french = voices.find(v => v.lang?.toLowerCase().startsWith('fr-fr'))
        || voices.find(v => v.lang?.toLowerCase().startsWith('fr'));
      voiceRef.current = french || voices[0] || null;
    };
    pickVoice();
    window.speechSynthesis.onvoiceschanged = pickVoice;
    return () => { window.speechSynthesis.onvoiceschanged = null; };
  }, [available]);

  const setEnabled = useCallback((v: boolean) => {
    setEnabledState(v);
    localStorage.setItem(STORAGE_KEY, v ? '1' : '0');
    if (!v && available) window.speechSynthesis.cancel();
  }, [available]);

  const speak = useCallback((text: string) => {
    if (!enabled || !available || !text?.trim()) return;
    // Cancel any queued utterance so the latest phrase always wins.
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = voiceRef.current?.lang || 'fr-FR';
    if (voiceRef.current) u.voice = voiceRef.current;
    u.rate = 1.0;
    u.pitch = 1.0;
    window.speechSynthesis.speak(u);
  }, [enabled, available]);

  const cancel = useCallback(() => {
    if (available) window.speechSynthesis.cancel();
  }, [available]);

  return { enabled, setEnabled, available, speak, cancel };
}
