"""V8 SignInterpreter - Configuration"""
import os
import sys

# Directory structure
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # V8 root
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(ROOT_DIR, 'ml')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Add root + ml to sys.path
for p in [ROOT_DIR, ML_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Paths
TEMPLATE_DIR = os.path.join(DATA_DIR, 'templates')
INDEX_PATH = os.path.join(DATA_DIR, 'faiss.index')
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.json')
CONTRIBUTIONS_DIR = os.path.join(DATA_DIR, 'contributions')
DB_PATH = os.path.join(SERVER_DIR, 'signinterpreter.db')
REFERENCE_DATASET = os.path.join(os.path.dirname(ROOT_DIR), 'SL')

# ML constants
SEQUENCE_LENGTH = 30
PREFILTER_TOP = 30
K = 11
THRESHOLD = 50.0
SMOOTH_ALPHA = 0.7
MOTION_START_THRESHOLD = 0.04
MOTION_END_THRESHOLD = 0.025
MOTION_END_FRAMES = 3
MIN_SIGN_FRAMES = 4
MAX_SIGN_FRAMES = 60
PREDICT_AFTER_N_FRAMES = 8
PREDICT_EVERY_N_FRAMES = 3
WEIGHT_POWER = 2.0

# WebSocket
FRAME_RATE = 15
JPEG_QUALITY = 70

# Admin
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', '12345678')

# CORS
FRONTEND_URL = 'http://localhost:5173'
