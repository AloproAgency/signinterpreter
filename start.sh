#!/bin/bash
# SignInterpreter V8 - Start backend + frontend
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/opt/anaconda3/envs/mp311/bin/python"

echo "SignInterpreter V8 - Starting..."

# Kill any existing processes on our ports
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null
sleep 1

# Backend
cd "$ROOT_DIR"
$PYTHON -m server.main &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

sleep 3

# Frontend
cd "$ROOT_DIR/client"
npm run dev &
FRONTEND_PID=$!
echo "Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "App ready at http://localhost:5173"
echo "API docs at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"

cleanup() {
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  lsof -ti:8000 | xargs kill -9 2>/dev/null
  lsof -ti:5173 | xargs kill -9 2>/dev/null
  echo "Stopped."
}

trap cleanup EXIT
wait
