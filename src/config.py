from pathlib import Path

#Raiz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#---Vocabularios---
VOCAB_PATH= PROJECT_ROOT / "vocabulario"
VOCAB_16K = VOCAB_PATH / "vocab_16k.json"


#---Modelos---
RUNS_PATH = PROJECT_ROOT /  "runs"
BEST_MODEL_PATH = PROJECT_ROOT/ "model" / "best.pth"
URL_BEST_MODEL = "https://drive.google.com/file/d/1VSHrUPD1nuS1FAUT0cYzzy-twNzCBVtx/view?usp=sharing"
#---Media---
MEDIA_PATH = PROJECT_ROOT / "media"

