"""Web UI + JSON API for the gold-neural workbench."""

from .app import create_app
from .jobs import JobManager

__all__ = ["create_app", "JobManager"]
