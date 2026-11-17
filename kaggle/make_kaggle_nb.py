"""Kaggle notebook for running the label-distillation jobs by hand from the Kaggle website.

    py -3 kaggle/make_kaggle_nb.py   -> kaggle/snsaid_kaggle.ipynb

Upload kaggle/snsaid_gpu_bundle.zip as a private dataset, import this notebook, attach the dataset, set
Accelerator = GPU T4 x2 and Internet = on, and Run All. teach.py finds data, code and adapters anywhere under
/kaggle/input, so the dataset's nesting does not matter.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
cells = [
    ("markdown",
     "# Label distillation (3 jobs)\n\n"
     "1. **Add Input** → your dataset made from `snsaid_gpu_bundle.zip`.\n"
     "2. **Settings** → Accelerator: *GPU T4 x2*; Internet: *on*; Persistence: *Files only*.\n"
     "3. Pick the jobs in the next cell, then **Save Version → Save & Run All (Commit)** so it runs in the "
     "background (up to 12 h).\n"
     "4. When it finishes: **Output** tab → download everything (or `results.zip`) and send it back.\n\n"
     "Two GPU sessions may run at once: e.g. one notebook with `PART = 'A'` and a copy with `PART = 'B'`."),
    ("code",
     "PART = 'ALL'   # 'A' = control + seed 1 (~6 h), 'B' = seed 2 (~3 h), 'ALL' = all three (~9 h)\n"
     "JOBS = {\n"
     "    'A': ['distill_vis_nodistil_s0@3.5:METHOD=distill,SEEDS=0,VIS=1,DW=0',\n"
     "          'distill_vis_s1@3.5:METHOD=distill,SEEDS=1,VIS=1'],\n"
     "    'B': ['distill_vis_s2@3.5:METHOD=distill,SEEDS=2,VIS=1'],\n"
     "}\n"
     "JOBS['ALL'] = JOBS['A'] + JOBS['B']\n"
     "import os, glob, shutil\n"
     "os.environ['JOBS'] = ';'.join(JOBS[PART])\n"
     "os.environ['DTYPE'] = 'float16'\n"
     "os.environ['BATCH_H'] = '11.6'\n"
     "teach = glob.glob('/kaggle/input/**/teach.py', recursive=True)\n"
     "print('teach.py:', teach)\n"
     "shutil.copy(teach[0], '/kaggle/working/teach_run.py')\n"
     "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    ("code",
     "!cd /kaggle/working && python teach_run.py 2>&1 | tee /kaggle/working/run.log | "
     "grep --line-buffered -E 'JOB|epoch|name lens|distil |Error|error|Traceback'"),
    ("code",
     "# one archive of the results (adapters included), for downloading\n"
     "!cd /kaggle/working && rm -f teach_run.py && zip -q -r results.zip . -x results.zip && ls -la"),
]
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": src.splitlines(keepends=True),
                 **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, src in cells]}
out = R / "kaggle" / "snsaid_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(out)
