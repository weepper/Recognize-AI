"""
Nextcloud ExApp lifecycle integration via nc-py-api.

This module handles the AppAPI protocol:
  - /heartbeat — liveness probe from Nextcloud
  - /init     — initialization callback
  - /enabled  — enable/disable lifecycle event

When APP_ID is not set (standalone dev mode), this module is a no-op.
"""
import logging

from config import EXAPP_MODE

logger = logging.getLogger(__name__)


async def enabled_handler(enabled: bool, nc) -> str:
    """Called by Nextcloud when the ExApp is enabled or disabled.

    Args:
        enabled: True when enabling, False when disabling.
        nc: AsyncNextcloudApp instance for communicating with Nextcloud.
    """
    if enabled:
        logger.info("Recognize AI ExApp has been ENABLED by Nextcloud.")
        # Trigger model warm-up on enable
        from utils import model_manager
        model_manager.load_all()
        status = model_manager.status()
        loaded = sum(1 for v in status.values() if v["loaded"])
        logger.info(f"Model warm-up complete: {loaded}/{len(status)} models ready.")
    else:
        logger.info("Recognize AI ExApp has been DISABLED by Nextcloud.")
        # Optionally stop the scanner
        try:
            from scanner import file_scanner
            file_scanner.stop()
        except Exception:
            pass
    return ""


def setup_exapp(app):
    """Configure the FastAPI app for ExApp mode.

    Adds AppAPI authentication middleware and registers lifecycle handlers.
    Only applies when EXAPP_MODE is True (APP_ID is set).
    """
    if not EXAPP_MODE:
        logger.info("Standalone mode — skipping ExApp integration.")
        return

    try:
        from nc_py_api.ex_app import AppAPIAuthMiddleware, set_handlers

        # Register lifecycle endpoints (/heartbeat, /init, /enabled)
        set_handlers(app, enabled_handler)

        # Secure all endpoints with Nextcloud's AppAPI authentication
        app.add_middleware(AppAPIAuthMiddleware)

        logger.info("ExApp mode — AppAPI middleware and lifecycle handlers registered.")
    except ImportError:
        logger.warning(
            "nc-py-api is not installed. ExApp lifecycle will not work. "
            "Install with: pip install nc-py-api"
        )
    except Exception as e:
        logger.error(f"Failed to setup ExApp integration: {e}")


def run_exapp(app_ref: str):
    """Start the ExApp using nc-py-api's run_app helper.

    Args:
        app_ref: The uvicorn app reference string, e.g. "main:app"
    """
    if not EXAPP_MODE:
        return False

    try:
        from nc_py_api.ex_app import run_app
        logger.info("Starting in ExApp mode via nc-py-api run_app...")
        run_app(app_ref, log_level="info")
        return True
    except ImportError:
        logger.warning("nc-py-api not installed — falling back to standalone uvicorn.")
        return False
