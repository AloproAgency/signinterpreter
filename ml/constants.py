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
THRESHOLD = 50.0

# Smoothing
SMOOTH_ALPHA = 0.7

# Sign segmentation
MOTION_START_THRESHOLD = 0.04
MOTION_END_THRESHOLD = 0.025
MOTION_END_FRAMES = 3
MIN_SIGN_FRAMES = 4
MAX_SIGN_FRAMES = 60

# Prediction
PREDICT_AFTER_N_FRAMES = 8
PREDICT_EVERY_N_FRAMES = 3

# Temporal weighting
WEIGHT_POWER = 2.0
