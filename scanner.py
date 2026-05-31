"""
Background file scanner for automatic media classification.

Connects to Nextcloud via nc-py-api to:
  1. Fetch unprocessed image files from user storage
  2. Process them through YOLO, ArcFace, and CLIP pipelines
  3. Write classification tags back via Nextcloud DAV/OCS API

Controllable via the admin settings panel (enable/disable, batch size, interval).
"""
import logging
import threading
import time
from typing import Optional

from config import SCANNER_ENABLED, SCANNER_BATCH_SIZE, SCANNER_INTERVAL, EXAPP_MODE

logger = logging.getLogger(__name__)

# Image file extensions to process
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic", ".heif"
}


class FileScanner:
    """Background scanner that processes Nextcloud files through AI pipelines."""

    def __init__(self):
        self._enabled = SCANNER_ENABLED
        self._batch_size = SCANNER_BATCH_SIZE
        self._interval = SCANNER_INTERVAL
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Stats
        self._files_processed = 0
        self._files_queued = 0
        self._files_failed = 0
        self._last_scan_time: Optional[float] = None
        self._scanning = False

        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def status(self) -> dict:
        """Return current scanner status for the admin panel."""
        with self._lock:
            return {
                "enabled": self._enabled,
                "scanning": self._scanning,
                "batch_size": self._batch_size,
                "interval": self._interval,
                "files_processed": self._files_processed,
                "files_queued": self._files_queued,
                "files_failed": self._files_failed,
                "last_scan_time": self._last_scan_time,
            }

    def configure(self, enabled: bool, batch_size: int = None, interval: int = None):
        """Update scanner configuration from the admin panel.

        Args:
            enabled: Whether to enable or disable the scanner.
            batch_size: Number of files to process per batch (optional).
            interval: Seconds between scan cycles (optional).
        """
        with self._lock:
            self._enabled = enabled
            if batch_size is not None and batch_size > 0:
                self._batch_size = batch_size
            if interval is not None and interval >= 10:
                self._interval = interval

        if enabled:
            self.start()
        else:
            self.stop()

    def start(self):
        """Start the background scanner thread."""
        if not EXAPP_MODE:
            logger.warning("Scanner requires ExApp mode (APP_ID must be set). Skipping.")
            return

        if self._thread and self._thread.is_alive():
            logger.info("Scanner is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._scan_loop, daemon=True, name="file-scanner")
        self._thread.start()
        logger.info(
            f"File scanner started (batch_size={self._batch_size}, "
            f"interval={self._interval}s)."
        )

    def stop(self):
        """Stop the background scanner thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._thread = None
        with self._lock:
            self._scanning = False
        logger.info("File scanner stopped.")

    def _scan_loop(self):
        """Main scan loop — runs on background thread."""
        logger.info("Scanner loop started.")
        while not self._stop_event.is_set():
            try:
                self._run_batch()
            except Exception as e:
                logger.exception(f"Scanner batch failed: {e}")

            # Wait for the configured interval, but check stop_event frequently
            for _ in range(self._interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

        logger.info("Scanner loop exited.")

    def _run_batch(self):
        """Process one batch of files from Nextcloud."""
        with self._lock:
            self._scanning = True

        try:
            files = self._fetch_unprocessed_files()
            if not files:
                logger.debug("No unprocessed files found.")
                with self._lock:
                    self._files_queued = 0
                return

            with self._lock:
                self._files_queued = len(files)

            batch = files[:self._batch_size]
            logger.info(f"Processing batch of {len(batch)} files ({len(files)} total queued).")

            for file_info in batch:
                if self._stop_event.is_set():
                    break
                try:
                    self._process_file(file_info)
                    with self._lock:
                        self._files_processed += 1
                        self._files_queued = max(0, self._files_queued - 1)
                except Exception as e:
                    logger.warning(f"Failed to process file {file_info.get('path', '?')}: {e}")
                    with self._lock:
                        self._files_failed += 1
                        self._files_queued = max(0, self._files_queued - 1)

            with self._lock:
                self._last_scan_time = time.time()

        finally:
            with self._lock:
                self._scanning = False

    def _fetch_unprocessed_files(self) -> list:
        """Fetch a list of image files from Nextcloud that haven't been classified yet.

        Uses nc-py-api to query the Nextcloud file system for images
        that lack classification tags.
        """
        try:
            from nc_py_api import NextcloudApp

            nc = NextcloudApp()
            # List files in user directories — this queries the DAV API
            # The actual implementation depends on how Nextcloud exposes
            # untagged files. For now, we use a simplified approach.
            # TODO: Integrate with Nextcloud's file tagging system to find
            # files that haven't been processed by Recognize AI yet.
            logger.debug("Fetching file list from Nextcloud...")

            # Placeholder: return empty list until full DAV integration
            # In production, this would query for files lacking the
            # "recognize-ai-processed" system tag.
            return []

        except ImportError:
            logger.error("nc-py-api is required for file scanning.")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch files from Nextcloud: {e}")
            return []

    def _process_file(self, file_info: dict):
        """Download a file from Nextcloud, run inference, and write tags back.

        Args:
            file_info: Dict with at least 'path' and 'user_id' keys.
        """
        from nc_py_api import NextcloudApp
        from inference import process_yolo, process_arcface, process_clip

        nc = NextcloudApp()
        file_path = file_info.get("path", "")

        logger.info(f"Processing: {file_path}")

        # Download file content from Nextcloud
        try:
            file_content = nc.files.download(file_path)
            if not file_content:
                logger.warning(f"Empty file content for {file_path}")
                return
        except Exception as e:
            logger.error(f"Failed to download {file_path}: {e}")
            raise

        image_bytes = file_content if isinstance(file_content, bytes) else file_content.read()

        # Run all three pipelines
        results = {}

        objects = process_yolo(image_bytes)
        if objects is not None:
            results["objects"] = objects

        faces = process_arcface(image_bytes)
        if faces is not None:
            results["faces"] = faces

        clip_embedding = process_clip(image_bytes)
        if clip_embedding is not None:
            results["clip_embedding"] = clip_embedding

        # Write results back to Nextcloud
        # TODO: Implement tag writing via Nextcloud's OCS API
        # This would create/assign system tags like "person", "car", etc.
        # based on the YOLO detection results.
        logger.info(
            f"Processed {file_path}: "
            f"{len(results.get('objects', []))} objects, "
            f"{len(results.get('faces', []))} faces, "
            f"CLIP={'yes' if results.get('clip_embedding') else 'no'}"
        )


# Global singleton
file_scanner = FileScanner()
