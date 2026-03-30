"""conftest.py — adds project root to sys.path so tests can import features/detector."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
