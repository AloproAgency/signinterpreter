"""V8 SignInterpreter - server-level configuration (paths, secrets, CORS)."""
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
CONTRIBUTIONS_DIR = os.path.join(DATA_DIR, 'contributions')
DB_PATH = os.path.join(SERVER_DIR, 'signinterpreter.db')
REFERENCE_DATASET = os.path.join(os.path.dirname(ROOT_DIR), 'SL')

# Admin
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', '12345678')

# CORS
FRONTEND_URL = 'http://localhost:5173'
