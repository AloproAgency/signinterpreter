export interface WordInfo {
  id: number;
  name: string;
  description: string;
  template_count: number;
  is_active: boolean;
  created_at: string | null;
}

export interface Prediction {
  word: string;
  distance: number;
  threshold: number;
  confidence: number;
  top_k: { word: string; distance: number }[];
  inference_ms: number;
  is_final: boolean;
}

export interface InferenceStatus {
  hand_visible: boolean;
  is_signing: boolean;
  buffer_length: number;
  motion_energy: number;
}

export interface ContributionInfo {
  id: number;
  word: string;
  contributor: string;
  status: 'pending' | 'approved' | 'rejected';
  recorded_at: string | null;
  reviewed_at: string | null;
  notes: string | null;
}

export interface StatsInfo {
  words: number;
  templates: number;
  contributions: { pending: number; approved: number; rejected: number };
  last_build: {
    built_at: string | null;
    n_templates: number;
    duration_ms: number;
    status: string;
  };
  engine_loaded: boolean;
  engine_words: number;
}

export type Theme = 'dark' | 'light';

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'info';
  message: string;
  duration?: number;
}
