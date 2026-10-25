"""FastAPI application initialization for DemandCast."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from demandcast.api.dependencies import get_model_container, get_vn1_model_container
from demandcast.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demandcast.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: Load active UCI and VN1 models on startup."""
    logger.info("Initializing DemandCast Forecasting API...")
    uci_container = get_model_container()
    uci_loaded = uci_container.load_promoted_model()
    if uci_loaded:
        logger.info(f"DemandCast UCI profile ready with active model version: {uci_container.version}")
    else:
        logger.warning("DemandCast UCI profile started without an active promoted model release.")

    vn1_container = get_vn1_model_container()
    vn1_loaded = vn1_container.load_promoted_model()
    if vn1_loaded:
        logger.info(f"DemandCast VN1 profile ready with active model version: {vn1_container.version}")
    else:
        logger.warning("DemandCast VN1 profile started without an active promoted model release.")

    yield
    logger.info("Shutting down DemandCast API...")


def create_app() -> FastAPI:
    """Application factory for DemandCast API."""
    app = FastAPI(
        title="DemandCast Forecasting Service",
        description=(
            "Production Machine Learning Forecasting API for retail weekly product demand. "
            "Accepts historical weekly sales series and generates recursive point forecasts."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
        return response

    # Include routes
    app.include_router(router)

    return app


app = create_app()
