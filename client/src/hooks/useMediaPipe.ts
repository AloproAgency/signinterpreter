/**
 * Client-side MediaPipe Holistic Landmarker.
 * Extracts pose + hand landmarks from video frames locally.
 */
import { useRef, useState, useCallback } from 'react';
import { HolisticLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { extractFeatures, hasHandVisible } from '../lib/features';
import type { HolisticResult } from '../lib/features';

// Staleness detection: if NEITHER the wrist position NOR any hand landmark
// changes over N consecutive frames, MediaPipe's VIDEO tracker is stuck on
// a frozen prediction (user left the frame, got occluded, etc.) — force
// handVisible=false. We include the full hand landmarks so a static-hold
// sign (e.g. pointing at the chest while fingers shape) doesn't look stale.
const STALE_MOTION_THRESHOLD = 0.002;    // mean L1 delta across all tracked landmarks
const STALE_FRAMES_LIMIT = 90;           // ~3000 ms at 30 FPS. Signs with a
                                         // static hold phase (pointing, G-handshape
                                         // pause) can look frozen for ~1 s; we
                                         // only want to fire on truly stuck
                                         // MediaPipe predictions (user gone for
                                         // several seconds).

export function useMediaPipe() {
  const landmarkerRef = useRef<HolisticLandmarker | null>(null);
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const lastTimestampRef = useRef(0);
  const prevLandmarksRef = useRef<{
    wrists: number[] | null;       // flat [lw_x, lw_y, lw_z, rw_x, rw_y, rw_z]
    lhHand: number[] | null;       // 21 × 3 flat
    rhHand: number[] | null;
  }>({ wrists: null, lhHand: null, rhHand: null });
  const staleFramesRef = useRef(0);

  const init = useCallback(async () => {
    if (landmarkerRef.current || loading) return;
    setLoading(true);

    try {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );

      const landmarker = await HolisticLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minHandLandmarksConfidence: 0.5,
      });

      landmarkerRef.current = landmarker;
      setReady(true);
      console.log('MediaPipe Holistic Landmarker ready (client-side)');
    } catch (err) {
      console.error('Failed to init MediaPipe:', err);
    }

    setLoading(false);
  }, [loading]);

  const processFrame = useCallback((video: HTMLVideoElement): {
    features: number[];
    handVisible: boolean;
    result: HolisticResult;
  } | null => {
    if (!landmarkerRef.current || !video.videoWidth) return null;

    // MediaPipe requires strictly increasing timestamps
    const timestamp = performance.now();
    if (timestamp <= lastTimestampRef.current) return null;
    lastTimestampRef.current = timestamp;

    try {
      const mpResult = landmarkerRef.current.detectForVideo(video, timestamp);

      // Convert MediaPipe results to our format
      const result: HolisticResult = {
        poseLandmarks: mpResult.poseLandmarks?.[0]?.map(
          (lm: any) => [lm.x, lm.y, lm.z] as [number, number, number]
        ) ?? null,
        leftHandLandmarks: mpResult.leftHandLandmarks?.[0]?.map(
          (lm: any) => [lm.x, lm.y, lm.z] as [number, number, number]
        ) ?? null,
        rightHandLandmarks: mpResult.rightHandLandmarks?.[0]?.map(
          (lm: any) => [lm.x, lm.y, lm.z] as [number, number, number]
        ) ?? null,
      };

      const features = extractFeatures(result);

      // MediaPipe Holistic keeps predicting hand landmarks for several seconds
      // after the real hands leave the frame (tracking persistence). Use the
      // wrist visibility in pose landmarks — it drops immediately when the
      // wrists go out of frame.
      const poseRaw = mpResult.poseLandmarks?.[0] as any[] | undefined;
      let handVisible = false;
      if (poseRaw && poseRaw.length >= 17) {
        const lwVis = poseRaw[15]?.visibility ?? 0;
        const rwVis = poseRaw[16]?.visibility ?? 0;
        handVisible = Math.max(lwVis, rwVis) > 0.5;
      } else {
        handVisible = hasHandVisible(result);
      }

      // Staleness check: build a flat vector of (wrists + every hand landmark)
      // and compute its L1 delta vs the previous frame. When ALL of these are
      // frozen for ~1 s, MediaPipe is stuck — override to hand gone.
      // Including hand landmarks means static-wrist signs (fingers moving)
      // keep handVisible true.
      const flatten = (arr: any[] | undefined) =>
        arr ? arr.flatMap((lm: any) => [lm.x, lm.y, lm.z ?? 0]) : null;

      let wrists: number[] | null = null;
      if (poseRaw && poseRaw.length >= 17) {
        wrists = [
          poseRaw[15].x, poseRaw[15].y, poseRaw[15].z ?? 0,
          poseRaw[16].x, poseRaw[16].y, poseRaw[16].z ?? 0,
        ];
      }
      const lhRaw = mpResult.leftHandLandmarks?.[0] as any[] | undefined;
      const rhRaw = mpResult.rightHandLandmarks?.[0] as any[] | undefined;
      const lhHand = flatten(lhRaw);
      const rhHand = flatten(rhRaw);

      if (handVisible && wrists) {
        const meanAbsDelta = (a: number[] | null, b: number[] | null) => {
          if (!a || !b || a.length !== b.length) return null;
          let s = 0;
          for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
          return s / a.length;
        };
        const deltas: number[] = [];
        const p = prevLandmarksRef.current;
        const w = meanAbsDelta(wrists, p.wrists);
        if (w !== null) deltas.push(w);
        const lh = meanAbsDelta(lhHand, p.lhHand);
        if (lh !== null) deltas.push(lh);
        const rh = meanAbsDelta(rhHand, p.rhHand);
        if (rh !== null) deltas.push(rh);

        // Staleness = MAX delta (if ANY part moves, we're not stuck).
        const motion = deltas.length ? Math.max(...deltas) : Infinity;
        if (motion < STALE_MOTION_THRESHOLD) {
          staleFramesRef.current += 1;
        } else {
          staleFramesRef.current = 0;
        }
        prevLandmarksRef.current = { wrists, lhHand, rhHand };
        if (staleFramesRef.current >= STALE_FRAMES_LIMIT) {
          handVisible = false;
        }
      } else {
        staleFramesRef.current = 0;
        prevLandmarksRef.current = { wrists: null, lhHand: null, rhHand: null };
      }

      return { features, handVisible, result };
    } catch {
      return null;
    }
  }, []);

  return { init, ready, loading, processFrame };
}
