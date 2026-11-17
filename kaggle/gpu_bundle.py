"""Bundle everything a GPU machine needs to run teach.py jobs outside Kaggle (Colab, a lab server, ...).

    py -3 kaggle/gpu_bundle.py        -> kaggle/snsaid_gpu_bundle.zip  (+ kaggle/snsaid_gpu.ipynb)

The zip recreates Kaggle's layout, so teach.py runs unchanged:
  kaggle/input/snsaid-tier0-pilot/      the creatures (data/tier0_pilot)
  kaggle/input/snsaid-code/seen_never_said/
  kaggle/input/snsaid-s-adapters/teach3_s_s{0,1,2}/{run.json, adapter/}
  teach.py
Unzip it at / (so /kaggle/input exists), then run teach.py with JOBS=... (see the notebook).
"""
import json
import zipfile
from pathlib import Path

R = Path(__file__).resolve().parents[1]
OUT = R / "kaggle" / "snsaid_gpu_bundle.zip"


def add_tree(z, src, dst, skip=("__pycache__",)):
    for f in sorted(src.rglob("*")):
        if f.is_file() and not any(s in f.parts for s in skip) and f.suffix not in (".pyc", ".bak"):
            z.write(f, f"{dst}/{f.relative_to(src).as_posix()}")


with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:          # pictures and safetensors do not compress
    add_tree(z, R / "data" / "tier0_pilot", "kaggle/input/snsaid-tier0-pilot", skip=("__pycache__", "sheets"))
    add_tree(z, R / "seen_never_said", "kaggle/input/snsaid-code/seen_never_said")
    for s in (0, 1, 2):
        d = R / "kaggle" / "out" / f"teach3_s_s{s}"
        z.write(d / "run.json", f"kaggle/input/snsaid-s-adapters/teach3_s_s{s}/run.json")
        add_tree(z, d / "adapter", f"kaggle/input/snsaid-s-adapters/teach3_s_s{s}/adapter")
    z.write(R / "kaggle" / "teach" / "teach.py", "teach.py")
print(f"{OUT} ({OUT.stat().st_size / 2**20:.0f} MB)")

# ---- Colab notebook
JOBS = {
    "control (vision LoRA, no distillation loss), seed 0": "distill_vis_nodistil_s0@4:METHOD=distill,SEEDS=0,VIS=1,DW=0",
    "vision LoRA, seed 1": "distill_vis_s1@4:METHOD=distill,SEEDS=1,VIS=1",
    "vision LoRA, seed 2": "distill_vis_s2@4:METHOD=distill,SEEDS=2,VIS=1",
}
cells = [
    ("markdown", "# Label distillation runs\n\nUpload `snsaid_gpu_bundle.zip` to your Google Drive (top folder), pick a "
                 "GPU runtime (Runtime > Change runtime type), and run the cells in order. Each job takes about 3 h "
                 "on a T4 (much less on an L4/A100) and writes to `/kaggle/working/<job>/`; the last cell zips the "
                 "results to Drive. Run one job per session if the session limit is short."),
    ("code", "from google.colab import drive\ndrive.mount('/content/drive')\n"
             "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    ("code", "!mkdir -p /kaggle/working /kaggle/tmp\n"
             "!cp /content/drive/MyDrive/snsaid_gpu_bundle.zip /content/\n"
             "!cd / && unzip -q -o /content/snsaid_gpu_bundle.zip && ls /kaggle/input"),
    ("code", "# Pick the jobs for this session (in order of importance). They run one after another.\n"
             "JOBS = [\n" + "".join(f"    {v!r},   # {k}\n" for k, v in JOBS.items()) + "]\n"
             "import os\nos.environ['JOBS'] = ';'.join(JOBS)\nos.environ['DTYPE'] = 'float16'\n"
             "os.environ['BATCH_H'] = '23'\n"
             "# If the GPU runs out of memory (16 GB T4), halve the pictures per step:\n"
             "# os.environ['JOBS'] = os.environ['JOBS'].replace('METHOD=distill', 'METHOD=distill,DN=2,REPLAY=1')"),
    ("code", "!cd /kaggle/working && python /teach.py 2>&1 | tee -a /kaggle/working/run.log | "
             "grep --line-buffered -E 'JOB|epoch|name lens|distil|Error|error|Traceback'"),
    ("code", "!cd /kaggle && zip -q -r /content/drive/MyDrive/snsaid_distill_results.zip working "
             "-x 'working/*/adapter_*/*' && ls -la /content/drive/MyDrive/snsaid_distill_results.zip\n"
             "# adapters separately (larger):\n"
             "!cd /kaggle && zip -q -r /content/drive/MyDrive/snsaid_distill_adapters.zip working -i '*/adapter_*/*'"),
]
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": src.splitlines(keepends=True),
                 **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, src in cells]}
(R / "kaggle" / "snsaid_gpu.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(R / "kaggle" / "snsaid_gpu.ipynb")
