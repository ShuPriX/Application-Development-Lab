import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.session import init_db
from app.api.routes import movies, analysis, health
from app.services.ml.sentiment import load_bert_model
from app.services.ml.aspects import load_bilstm_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    await init_db()

    logger.info("Loading ML models...")
    try:
        load_bert_model(settings.BERT_MODEL_PATH)
    except Exception as e:
        logger.warning(f"Failed to load BERT model: {e}. Using fallback.")

    try:
        load_bilstm_model(settings.BILSTM_MODEL_PATH)
    except Exception as e:
        logger.warning(f"Failed to load BiLSTM model: {e}. Using keyword fallback.")

    logger.info("CineInsight backend ready!")
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="CineInsight API",
    description="Movie Review Sentiment Analysis System",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router)
app.include_router(movies.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")


# WebSocket endpoint for analysis progress
@app.websocket("/ws/analysis/{tmdb_id}")
async def websocket_analysis_progress(websocket: WebSocket, tmdb_id: int):
    await websocket.accept()

    # Register connection
    from app.api.routes.analysis import _ws_connections
    if tmdb_id not in _ws_connections:
        _ws_connections[tmdb_id] = []
    _ws_connections[tmdb_id].append(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _ws_connections[tmdb_id].remove(websocket)
        if not _ws_connections[tmdb_id]:
            del _ws_connections[tmdb_id]
