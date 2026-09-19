# VAST CASH Streamlit entrypoint
# The complete trading engine lives in app.py.
# Dynamic loader keeps this entrypoint synchronized with the engine.
from pathlib import Path
exec(compile(Path("app.py").read_text(encoding="utf-8"), "app.py", "exec"))
