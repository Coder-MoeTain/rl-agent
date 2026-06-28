"""Tests for logging setup."""

import logging

from setup_logging import setup_logging


def test_setup_logging_configures_handler():
    setup_logging(level="DEBUG")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) >= 1
