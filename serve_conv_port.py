"""Serve the conversation-classifier pipeline with the model and port chosen at runtime.

serve_conv.py hardcodes port 8002 and defaults PAYCHAT_CONV_MODEL to conv_model_v2,
which silently evaluates a two-generation-old model. A paired v4-vs-v5 comparison needs
both models serving at once on different ports, so neither run has to wait for the other
to be torn down.

  python serve_conv_port.py --model conv_model_v4 --port 8003
"""
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="conv model directory, e.g. conv_model_v4")
ap.add_argument("--port", type=int, required=True)
a = ap.parse_args()

if not (ROOT / a.model).is_dir():
    raise SystemExit(f"no such conv model directory: {a.model}")

# Same base intent model both runs use — only the conversation model differs.
os.environ.setdefault(
    "MODEL_DIR",
    r"C:\Users\akash\Downloads\paychat-full\paychat-full\model\saved_model")
os.environ["PAYCHAT_CONV_CLASSIFIER"] = "1"
os.environ["PAYCHAT_CONV_MODEL"] = a.model

import uvicorn
from app import app

if __name__ == "__main__":
    print(f"serving {a.model} on :{a.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
