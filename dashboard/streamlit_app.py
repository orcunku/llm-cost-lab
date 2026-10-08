"""Streamlit Community Cloud entry point for the dashboard.

Cloud installs dashboard/requirements.txt (streamlit, pandas, numpy only), so the hosted dashboard skips the
heavy ML stack: it only reads the committed results in results/. Locally, `streamlit run app.py` still works."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))       # so app.py can import the src package

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")   # run_path, not import: Streamlit re-runs on each input
