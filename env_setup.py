import os
import sys

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

HF_HOME = os.getenv("BOT_HF_HOME") or os.path.join(HERE, "hf_cache")
os.environ["HF_HOME"] = HF_HOME  # переопределяем HF_HOME из системы
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if os.path.isdir(os.path.join(HF_HOME, "hub", "models--Qwen--Qwen3.5-2B")):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
