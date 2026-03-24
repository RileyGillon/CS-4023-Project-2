# Copyright 2026 Riley
#
# conftest.py – project-level pytest configuration
# -------------------------------------------------
# Ensures the project2_reactive package is importable when running tests
# directly with `pytest` (without a full colcon build).  This is useful
# for rapid development and CI environments that do not have ROS 2 installed.

import os
import sys

# Add the package root (one level above this test/ directory) to sys.path so
# that `from project2_reactive import behaviors` etc. resolve correctly.
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
