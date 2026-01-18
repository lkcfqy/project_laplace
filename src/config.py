import os
from pathlib import Path

# Project Root (calculated relative to this file: src/config.py -> ../)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Directories
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Specific Paths
ADAPTER_PATH = MODELS_DIR / "qwen_dsl_adapter"
DSL_DATA_FILE = DATA_DIR / "synthetic_dsl_train.jsonl"
PRIMITIVES_DATA_FILE = DATA_DIR / "synthetic_primitives_train.jsonl"
DREAM_DATA_FILE = DATA_DIR / "dream_traces.jsonl"
HDC_DATA_FILE = DATA_DIR / "hdc_training_data.jsonl"
VALUE_NET_PATH = MODELS_DIR / "value_net.pth"
ARC_TRAIN_DIR = DATA_DIR / "arc/training"
ARC_EVAL_DIR = DATA_DIR / "arc/evaluation"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
