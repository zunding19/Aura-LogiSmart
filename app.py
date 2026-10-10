"""Entry point: `streamlit run app.py` opens the LogiSmart command deck in aura_app.py."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("aura_app.py")), run_name="__main__")
