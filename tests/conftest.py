"""
Pytest configuration.

Ensures the project root is on PYTHONPATH so that
imports like `from src.heartml...` work in both
local and CI environments.
"""

import sys
from pathlib import Path

# Resolve project root (folder containing 'src')
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add to PYTHONPATH only if not already present
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
