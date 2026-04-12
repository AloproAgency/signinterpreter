import { useEffect, useRef, useState } from 'react';
import { api } from '../lib/api';

// Pose layout (from ml/features.py UPPER_BODY_LANDMARKS):
// 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear,
// 5 l_sh, 6 r_sh, 7 l_el, 8 r_el, 9 l_wrist, 10 r_wrist, 11 l_pinky, 12 r_pinky
const POSE_CONNECTIONS: [number, number][] = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [0, 5], [0, 6], [9, 11], [10, 12],
];

const HAND_CONNECTIONS: [number, number][] = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17],
];

const HAND_WEIGHT = 3.0; // must match ml/features.py

function drawFrame(
  ctx: CanvasRenderingContext2D,
  frame: number[],
  w: number,
  h: number,
  isDark: boolean,
) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = isDark ? '#0d1117' : '#f6f8fa';
  ctx.fillRect(0, 0, w, h);

  // Body-frame → canvas mapping
  const cx = w / 2;
  const cy = h * 0.45;
  const S = Math.min(w, h) * 0.28; // shoulder span scale

  // Pose: 13 landmarks × 3 at [0:39]
  const pose: [number, number][] = [];
  for (let i = 0; i < 13; i++) {
    const x = frame[i * 3];
    const y = frame[i * 3 + 1];
    pose.push([cx + x * S, cy + y * S]);
  }

  // Hand shapes (relative to wrist, weighted by HAND_WEIGHT). Size in body units = 0.25
  const handScaleBody = 0.25;
  const unpackHand = (start: number, wristPose: [number, number]): [number, number][] => {
    const pts: [number, number][] = [];
    for (let i = 0; i < 21; i++) {
      const hx = frame[start + i * 3] / HAND_WEIGHT;
      const hy = frame[start + i * 3 + 1] / HAND_WEIGHT;
      pts.push([wristPose[0] + hx * S * handScaleBody,
                wristPose[1] + hy * S * handScaleBody]);
    }
    return pts;
  };

  const lHand = unpackHand(39, pose[9]);   // left wrist index 9
  const rHand = unpackHand(105, pose[10]); // right wrist index 10

  // Check if hand is present (non-zero shape data)
  const lhPresent = frame.slice(39, 39 + 63).some(v => Math.abs(v) > 1e-6);
  const rhPresent = frame.slice(105, 105 + 63).some(v => Math.abs(v) > 1e-6);

  // Pose connections
  ctx.strokeStyle = '#1396ba';
  ctx.lineWidth = 2.5;
  for (const [a, b] of POSE_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(pose[a][0], pose[a][1]);
    ctx.lineTo(pose[b][0], pose[b][1]);
    ctx.stroke();
  }

  // Pose joints
  ctx.fillStyle = isDark ? '#e6edf3' : '#1f2328';
  for (const [px, py] of pose) {
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  const drawHand = (pts: [number, number][], color: string) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(pts[a][0], pts[a][1]);
      ctx.lineTo(pts[b][0], pts[b][1]);
      ctx.stroke();
    }
    ctx.fillStyle = color;
    for (const [px, py] of pts) {
      ctx.beginPath();
      ctx.arc(px, py, 2, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  if (lhPresent) drawHand(lHand, '#7c3aed');
  if (rhPresent) drawHand(rHand, '#ec4899');
}

export function SkeletonPlayer({ contributionId, size = 320 }: { contributionId: number; size?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [frames, setFrames] = useState<number[][] | null>(null);
  const [error, setError] = useState('');
  const frameIdxRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    setFrames(null);
    setError('');
    api.getContributionFeatures(contributionId)
      .then(data => { if (!cancelled) setFrames(data.frames); })
      .catch(() => { if (!cancelled) setError('Features indisponibles'); });
    return () => { cancelled = true; };
  }, [contributionId]);

  useEffect(() => {
    if (!frames || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const isDark = document.documentElement.classList.contains('dark');

    frameIdxRef.current = 0;
    const interval = window.setInterval(() => {
      const i = frameIdxRef.current % frames.length;
      drawFrame(ctx, frames[i], canvas.width, canvas.height, isDark);
      frameIdxRef.current = i + 1;
    }, 1000 / 15);

    return () => clearInterval(interval);
  }, [frames]);

  return (
    <div className="relative w-full aspect-square rounded-md border border-[#d0d7de] dark:border-[#30363d] overflow-hidden bg-[#f6f8fa] dark:bg-[#0d1117]">
      <canvas ref={canvasRef} width={size} height={size} className="w-full h-full" />
      {!frames && !error && (
        <div className="absolute inset-0 flex items-center justify-center text-sm text-[#8b949e]">
          Chargement...
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center text-sm text-[#ef4444]">
          {error}
        </div>
      )}
    </div>
  );
}
