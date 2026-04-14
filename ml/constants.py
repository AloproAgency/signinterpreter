"""ML constants shared across all ML modules."""
import os

# Default paths (overridden by server config when used via webapp)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
TEMPLATE_DIR = os.path.join(DATA_DIR, 'templates')
INDEX_PATH = os.path.join(DATA_DIR, 'faiss.index')
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.json')
CONTRIBUTIONS_DIR = os.path.join(DATA_DIR, 'contributions')

# Sequence
SEQUENCE_LENGTH = 30

# FAISS + KNN
PREFILTER_TOP = 30
K = 5
THRESHOLD = 0.8  # Phono: -log(prob) — accept when prob >= exp(-0.8) ≈ 0.45

# Smoothing
SMOOTH_ALPHA = 0.7

# Sign segmentation
# Capture frame rate (client ~30 FPS)
MOTION_START_THRESHOLD = 0.025   # hand-dims mean delta; 2 consecutive frames required
MOTION_END_THRESHOLD = 0.018
MOTION_END_FRAMES = 6            # ~200 ms at 30 FPS
MIN_SIGN_FRAMES = 4              # ~130 ms — catches fast signs
MAX_SIGN_FRAMES = 120            # ~4 s at 30 FPS

# Prediction
PREDICT_AFTER_N_FRAMES = 8       # ~270 ms, early verdict
PREDICT_EVERY_N_FRAMES = 5       # ~165 ms cadence

# Temporal weighting
WEIGHT_POWER = 2.0
