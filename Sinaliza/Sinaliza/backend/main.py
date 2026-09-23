"""
Sinaliza Backend — FastAPI application.

Uso:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.middleware.cors import RequestLoggingMiddleware
from backend.routers import health_router, predict_router, dictionary_router, websocket_router
from backend.services.prediction import prediction_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: carrega modelo."""
    logger.info(f"Iniciando {settings.app_name} v{settings.version}...")
    prediction_service.load_model()
    yield
    logger.info("Encerrando...")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="API para tradução de Libras em tempo real usando IA.",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# Routers
app.include_router(health_router, prefix="/api")
app.include_router(predict_router, prefix="/api")
app.include_router(dictionary_router, prefix="/api")
app.include_router(websocket_router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "status": "online",
    }
