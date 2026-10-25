"""Application state and model dependency injection for DemandCast API."""

import logging
from typing import Any

from fastapi import HTTPException, status

from demandcast.models import DemandForecastModel
from demandcast.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelContainer:
    """Singleton container managing the in-memory active forecasting model."""

    def __init__(self):
        self.model: DemandForecastModel | None = None
        self.metadata: dict[str, Any] = {}
        self.version: str | None = None
        self.is_ready: bool = False
        self.init_error: str | None = None

    def load_promoted_model(self) -> bool:
        """Attempt to load the currently promoted model release from the registry."""
        try:
            registry = ModelRegistry()
            info = registry.get_current_release_info()
            if info is None:
                self.is_ready = False
                self.init_error = "No model is currently promoted. models/current.json not found."
                logger.warning(self.init_error)
                return False

            version = info["current_version"]
            model, meta = registry.load_current_model()
            self.model = model
            self.metadata = meta
            self.version = version
            self.is_ready = True
            self.init_error = None
            logger.info(f"Loaded active model version '{version}' ({model.name}) into memory.")
            return True
        except Exception as e:
            self.is_ready = False
            self.init_error = f"Failed to load model release: {e}"
            logger.error(self.init_error, exc_info=True)
            return False


class VN1ModelContainer:
    """Singleton container managing the in-memory active VN1 forecasting model."""

    def __init__(self):
        self.model: Any = None
        self.metadata: dict[str, Any] = {}
        self.version: str | None = None
        self.is_ready: bool = False
        self.init_error: str | None = None

    def load_promoted_model(self) -> bool:
        """Attempt to load the currently promoted VN1 model release."""
        try:
            from demandcast.vn1.registry import VN1ModelRegistry

            registry = VN1ModelRegistry()
            info = registry.get_current_release_info()
            if info is None:
                self.is_ready = False
                self.init_error = "No VN1 model is currently promoted. models/profiles/vn1/current.json not found."
                logger.warning(self.init_error)
                return False

            version = info["current_version"]
            model, meta = registry.load_current_model()
            self.model = model
            self.metadata = meta
            self.version = version
            self.is_ready = True
            self.init_error = None
            logger.info(
                f"Loaded active VN1 model version '{version}' ({getattr(model, 'name', 'VN1Model')}) into memory."
            )
            return True
        except Exception as e:
            self.is_ready = False
            self.init_error = f"Failed to load VN1 model release: {e}"
            logger.error(self.init_error, exc_info=True)
            return False


# Global container instances
_container = ModelContainer()
_vn1_container = VN1ModelContainer()


def get_model_container() -> ModelContainer:
    """Dependency provider returning the ModelContainer (UCI profile)."""
    return _container


def get_vn1_model_container() -> VN1ModelContainer:
    """Dependency provider returning the VN1ModelContainer (VN1 profile)."""
    return _vn1_container


def require_ready_model() -> tuple[DemandForecastModel, dict[str, Any], str]:
    """Dependency ensuring model is loaded and ready before processing inference requests."""
    container = get_model_container()
    if not container.is_ready or container.model is None or container.version is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "ModelServiceNotReady",
                "message": container.init_error or "Forecasting model is not ready to serve predictions.",
            },
        )
    return container.model, container.metadata, container.version


def require_ready_vn1_model() -> tuple[Any, dict[str, Any], str]:
    """Dependency ensuring VN1 model is loaded and ready before processing v2 inference requests."""
    container = get_vn1_model_container()
    if not container.is_ready or container.model is None or container.version is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "VN1ModelServiceNotReady",
                "message": container.init_error or "VN1 Forecasting model is not ready to serve predictions.",
            },
        )
    return container.model, container.metadata, container.version
