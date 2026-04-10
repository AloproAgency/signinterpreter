"""V8 SignInterpreter - FastAPI Application"""
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from server.config import FRONTEND_URL, TEMPLATE_DIR, CONTRIBUTIONS_DIR
from server.database import init_db
from server.routers import inference_ws, collection_ws, api, team


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    os.makedirs(CONTRIBUTIONS_DIR, exist_ok=True)

    from server.routers.inference_ws import get_engine
    get_engine()

    print("SignInterpreter V8 ready!")
    yield
    print("Shutting down...")


app = FastAPI(
    title="SignInterpreter V8",
    description="Real-time sign language recognition system",
    version="8.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference_ws.router)
app.include_router(collection_ws.router)
app.include_router(api.router)
app.include_router(team.router)

# Serve frontend build if it exists
frontend_build = os.path.join(ROOT_DIR, 'client', 'dist')
if os.path.isdir(frontend_build):
    app.mount("/", StaticFiles(directory=frontend_build, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
