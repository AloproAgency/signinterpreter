"""ML constants shared across all ML modules."""
import os

# Default paths (overridden by server config when used via webapp)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
TEMPLATE_DIR = os.path.join(DATA_DIR, 'templates')
CONTRIBUTIONS_DIR = os.path.join(DATA_DIR, 'contributions')

# Sequence
SEQUENCE_LENGTH = 30

# Classification acceptance threshold: -log(prob). prob >= exp(-0.55) ≈ 0.58
THRESHOLD = 0.55

# Smoothing
SMOOTH_ALPHA = 0.7

# Sign segmentation  (capture frame rate ≈ 30 FPS)
MOTION_START_THRESHOLD = 0.025   # hand-dims mean delta; 2 consecutive frames required
MOTION_END_THRESHOLD = 0.018     # preserves sign tail for classifier
MOTION_END_FRAMES = 6            # ~200 ms at 30 FPS
MIN_SIGN_FRAMES = 6              # ~200 ms — rejects fidgets too short to be real signs
MAX_SIGN_FRAMES = 120            # ~4 s at 30 FPS

# Intermediate-prediction cadence (UI feedback only)
PREDICT_AFTER_N_FRAMES = 5       # first intermediate verdict after this many frames
PREDICT_EVERY_N_FRAMES = 3
