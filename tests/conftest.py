"""
Pytest configuration.

Ensures the project root is on PYTHONPATH so that
imports like `from src.heartml...` work in local
and CI environments.
"""

import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
