"""MediaPipe Holistic wrapper for feature extraction."""
import cv2
import numpy as np
import mediapipe as mp
from ml.features import extract_features_from_results

mp_holistic = mp.solutions.holistic


class MediaPipeService:
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        """Process a BGR frame, return (features, results, hand_visible)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)
        features = extract_features_from_results(results)
        hand_visible = (
            results.left_hand_landmarks is not None or
            results.right_hand_landmarks is not None
        )
        return features, results, hand_visible

    def close(self):
        self.holistic.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
