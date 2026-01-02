"""
Conftest for example tests.

Adds the examples directories to Python path so tests can import from them.
"""

import sys
from pathlib import Path

# Add examples directories to path
examples_root = Path(__file__).parent.parent.parent / "examples"
sys.path.insert(0, str(examples_root / "langgraph"))
sys.path.insert(0, str(examples_root / "pydantic_ai"))
