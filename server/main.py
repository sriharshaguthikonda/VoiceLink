# ============================================================================
# VoiceLink — Inference Server Entry Point
# ============================================================================
#
# WHAT THIS FILE DOES:
# 1. Creates the FastAPI application
# 2. Loads the TTS model on startup (in a lifespan context manager)
# 3. Mounts the API routers (/v1/tts, /v1/voices, /v1/health)
# 4. Runs with Uvicorn when executed directly
#
# HOW TO RUN:
#   From the VoiceLink project root:
#     python -m server.main
#
#   Or with uvicorn directly (supports hot-reload for development):
#     uvicorn server.main:app --host 127.0.0.1 --port 7860 --reload
#
# LIFESPAN:
# FastAPI's lifespan context manager handles startup and shutdown.
# This is where we load the model (startup) and unload it (shutdown).
# The model stays in memory for the entire lifetime of the server.
# ============================================================================

import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from server.config import settings
from server.models import load_model
from server.routers.tts import router as tts_router, set_model, set_start_time


def setup_logging():
    """
    Configure loguru for nice colored output.

    Loguru replaces Python's built-in logging with:
    - Colored output by default
    - Easy file rotation
    - Better exception formatting
    - No boilerplate (just `from loguru import logger; logger.info("hi")`)
    """
    # Remove default handler, add our own
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.server.log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages server startup and shutdown.

    Startup:  Load the TTS model into memory (GPU/CPU).
    Shutdown: Unload the model, free memory.

    This runs BEFORE any requests are accepted (startup)
    and AFTER all requests are done (shutdown).
    """
    # --- STARTUP ---
    setup_logging()
    set_start_time(time.time())

    logger.info("=" * 60)
    logger.info("  VoiceLink Inference Server starting...")
    logger.info("=" * 60)
    logger.info(f"Host:  {settings.server.host}")
    logger.info(f"Port:  {settings.server.port}")
    logger.info(f"Model: {settings.model.default_model}")

    # Load the TTS model
    t0 = time.perf_counter()
    try:
        model = load_model(
            settings.model.default_model,
            lang_code=settings.model.kokoro_lang_code,
            device=settings.model.device,
        )
        set_model(model)
        elapsed = time.perf_counter() - t0
        logger.info(f"Model loaded in {elapsed:.1f}s")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Server will start but TTS will return 503 errors.")

    logger.info("")
    logger.info(f"Server ready at http://{settings.server.host}:{settings.server.port}")
    logger.info(f"API docs at   http://{settings.server.host}:{settings.server.port}/docs")
    logger.info("")

    # --- SERVE REQUESTS ---
    yield

    # --- SHUTDOWN ---
    logger.info("Server shutting down...")
    if model is not None:
        model.unload()
    logger.info("Goodbye!")


# --- Create the FastAPI app ---
app = FastAPI(
    title="VoiceLink Inference Server",
    description=(
        "Local TTS inference server that bridges neural text-to-speech models "
        "with Windows SAPI. Provides streaming PCM audio from models like Kokoro."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Mount routers
app.include_router(tts_router)


# --- Direct execution ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.main:app",
        host=settings.server.host,
        port=settings.server.port,
        log_level=settings.server.log_level.lower(),
    )
