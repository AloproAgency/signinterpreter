/**
 * Client-side MediaPipe Holistic Landmarker.
 * Extracts pose + hand landmarks from video frames locally.
 */
import { useRef, useState, useCallback } from 'react';
import { HolisticLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { extractFeatures, hasHandVisible } from '../lib/features';
import type { HolisticResult } from '../lib/features';

// Staleness detection: if the wrist position barely changes over N consecutive
// frames, MediaPipe's VIDEO tracker is likely stuck on a frozen prediction
// (user left the frame, got occluded, etc.) — force handVisible=false.
const STALE_MOTION_THRESHOLD = 0.0015;   // image-normalised units per frame
const STALE_FRAMES_LIMIT = 20;           // ~670 ms at 30 FPS

export function useMediaPipe() {
  const landmarkerRef = useRef<HolisticLandmarker | null>(null);
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const lastTimestampRef = useRef(0);
  const prevWristRef = useRef<{ lw: number[] | null; rw: number[] | null }>({ lw: null, rw: null });
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

      // Staleness check: if the dominant wrist hasn't moved for ~670 ms,
      // MediaPipe's tracker is stuck. Override to hand gone.
      if (handVisible && poseRaw && poseRaw.length >= 17) {
        const lw: number[] = [poseRaw[15].x, poseRaw[15].y, poseRaw[15].z ?? 0];
        const rw: number[] = [poseRaw[16].x, poseRaw[16].y, poseRaw[16].z ?? 0];
        const d = (a: number[] | null, b: number[] | null) =>
          !a || !b ? Infinity : Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        const lwDelta = d(lw, prevWristRef.current.lw);
        const rwDelta = d(rw, prevWristRef.current.rw);
        const maxDelta = Math.max(lwDelta, rwDelta);
        if (maxDelta < STALE_MOTION_THRESHOLD) {
          staleFramesRef.current += 1;
        } else {
          staleFramesRef.current = 0;
        }
        prevWristRef.current = { lw, rw };
        if (staleFramesRef.current >= STALE_FRAMES_LIMIT) {
          handVisible = false;
        }
      } else {
        staleFramesRef.current = 0;
        prevWristRef.current = { lw: null, rw: null };
      }

      return { features, handVisible, result };
    } catch {
      return null;
    }
  }, []);

  return { init, ready, loading, processFrame };
}
