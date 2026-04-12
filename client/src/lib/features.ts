/**
 * Client-side feature extraction — exact mirror of ml/features.py
 *
 * Layout (171 features per frame):
 *   [0:39]    Upper-body pose (13 landmarks × 3) normalized by shoulders
 *   [39:105]  Left hand shape (21×3=63) + palm orientation (3) = 66
 *   [105:171] Right hand shape (21×3=63) + palm orientation (3) = 66
 */

// Must match ml/features.py exactly
const UPPER_BODY_LANDMARKS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18];
const N_POSE = UPPER_BODY_LANDMARKS.length; // 13
const HAND_WEIGHT = 3.0;
const PALM_WEIGHT = 5.0;
export const FEATURE_DIM = N_POSE * 3 + 66 + 66; // 171

type Vec3 = [number, number, number];

function normalize(landmarks: Vec3[], midpoint: Vec3, scale: number): Vec3[] {
  return landmarks.map(([x, y, z]) => [
    (x - midpoint[0]) / scale,
    (y - midpoint[1]) / scale,
    (z - midpoint[2]) / scale,
  ]);
}

function normalizeHand(hand: Vec3[]): Vec3[] {
  const wrist = hand[0];
  const centered = hand.map(([x, y, z]) => [
    x - wrist[0], y - wrist[1], z - wrist[2]
  ] as Vec3);

  let maxDist = 0;
  for (const [x, y, z] of centered) {
    const d = Math.sqrt(x * x + y * y + z * z);
    if (d > maxDist) maxDist = d;
  }
  if (maxDist < 1e-6) maxDist = 1;

  return centered.map(([x, y, z]) => [x / maxDist, y / maxDist, z / maxDist]);
}

function palmOrientation(hand: Vec3[]): Vec3 {
  const wrist = hand[0];
  const indexBase = hand[5];
  const pinkyBase = hand[17];

  const v1: Vec3 = [
    indexBase[0] - wrist[0], indexBase[1] - wrist[1], indexBase[2] - wrist[2]
  ];
  const v2: Vec3 = [
    pinkyBase[0] - wrist[0], pinkyBase[1] - wrist[1], pinkyBase[2] - wrist[2]
  ];

  // Cross product
  const normal: Vec3 = [
    v1[1] * v2[2] - v1[2] * v2[1],
    v1[2] * v2[0] - v1[0] * v2[2],
    v1[0] * v2[1] - v1[1] * v2[0],
  ];

  const norm = Math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2);
  if (norm < 1e-8) return [0, 0, 0];
  return [normal[0] / norm, normal[1] / norm, normal[2] / norm];
}

function extractHandFeatures(hand: Vec3[] | null): number[] {
  if (!hand || hand.length < 21) {
    return new Array(66).fill(0);
  }

  const norm = normalizeHand(hand);
  const shape: number[] = [];
  for (const [x, y, z] of norm) {
    shape.push(x * HAND_WEIGHT, y * HAND_WEIGHT, z * HAND_WEIGHT);
  }

  const palm = palmOrientation(hand);
  shape.push(palm[0] * PALM_WEIGHT, palm[1] * PALM_WEIGHT, palm[2] * PALM_WEIGHT);

  return shape; // 63 + 3 = 66
}

export interface HolisticResult {
  poseLandmarks: Vec3[] | null;       // 33 landmarks
  leftHandLandmarks: Vec3[] | null;   // 21 landmarks
  rightHandLandmarks: Vec3[] | null;  // 21 landmarks
}

export function extractFeatures(result: HolisticResult): number[] {
  const { poseLandmarks, leftHandLandmarks, rightHandLandmarks } = result;

  // Pose features
  let poseFeatures = new Array(N_POSE * 3).fill(0);

  if (poseLandmarks && poseLandmarks.length >= 25) {
    const sl = poseLandmarks[11]; // left shoulder
    const sr = poseLandmarks[12]; // right shoulder
    const midpoint: Vec3 = [
      (sl[0] + sr[0]) / 2, (sl[1] + sr[1]) / 2, (sl[2] + sr[2]) / 2
    ];
    const dx = sr[0] - sl[0], dy = sr[1] - sl[1], dz = sr[2] - sl[2];
    let scale = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (scale < 1e-6) scale = 1;

    const upperPose = UPPER_BODY_LANDMARKS.map(i => poseLandmarks[i]);
    const normalized = normalize(upperPose, midpoint, scale);
    poseFeatures = normalized.flatMap(([x, y, z]) => [x, y, z]);
  }

  const lhFeatures = extractHandFeatures(leftHandLandmarks);
  const rhFeatures = extractHandFeatures(rightHandLandmarks);

  return [...poseFeatures, ...lhFeatures, ...rhFeatures]; // 39 + 66 + 66 = 171
}

export function hasHandVisible(result: HolisticResult): boolean {
  return (result.leftHandLandmarks !== null && result.leftHandLandmarks.length > 0) ||
         (result.rightHandLandmarks !== null && result.rightHandLandmarks.length > 0);
}
