/**
 * Client-side MediaPipe Holistic Landmarker.
 * Extracts pose + hand landmarks from video frames locally.
 */
import { useRef, useState, useCallback } from 'react';
import { HolisticLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { extractFeatures, hasHandVisible } from '../lib/features';
import type { HolisticResult } from '../lib/features';

export function useMediaPipe() {
  const landmarkerRef = useRef<HolisticLandmarker | null>(null);
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const lastTimestampRef = useRef(0);

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
      const handVisible = hasHandVisible(result);

      return { features, handVisible, result };
    } catch {
      return null;
    }
  }, []);

  return { init, ready, loading, processFrame };
}
