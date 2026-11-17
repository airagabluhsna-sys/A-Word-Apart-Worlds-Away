"""CPU kernel: the Ollama in-context controls (offline/*.py) on Kaggle instead of the local machine.

Settings (push.py --set): MODEL=qwen2.5vl:3b  SCRIPT=ollama_vlm_icl.py  ARGS="all"
Attached datasets: snsaid-tier0-pilot, snsaid-real-v1, snsaid-code, snsaid-offline (offline/*.py and any partial
result files, so a run resumes where the last one stopped). Results are written straight into
/kaggle/working/offline/, so a run cut off at the 12 h limit still keeps every finished row.
"""
import glob
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

env = os.environ.get
ROOT = Path("/kaggle/tmp/root")                       # mirrors the repo layout the offline scripts expect
OUT = Path("/kaggle/working/offline")
OUT.mkdir(parents=True, exist_ok=True)


def unpack(src, work):
    work.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix == ".zip":
            zipfile.ZipFile(f).extractall(work / f.stem)
        elif f.is_dir():
            shutil.copytree(f, work / f.name, dirs_exist_ok=True)
        else:
            shutil.copy(f, work / f.name)
    for d in [p.name for p in work.iterdir() if p.is_dir()]:
        if (work / d / d).exists():
            (work / d / d).rename(work / f"_{d}")
            shutil.rmtree(work / d)
            (work / f"_{d}").rename(work / d)
    return work


found = [Path(p).parent for p in glob.glob("/kaggle/input/**/concepts.jsonl", recursive=True) if "tier0_R" not in p]
tier0 = [p for p in found if not (p / "probes_K.jsonl").exists()][0]
real = [p for p in found if (p / "probes_K.jsonl").exists()][0]
unpack(tier0, ROOT / "data" / "tier0_pilot")
unpack(real, ROOT / "data" / "real_v1")
code = [Path(p).parents[1] for p in glob.glob("/kaggle/input/**/snsaid-code/**/teach/run.py", recursive=True)]
if not code:
    zs = [z for z in glob.glob("/kaggle/input/**/snsaid-code/**/*.zip", recursive=True)]
    for z in zs:
        zipfile.ZipFile(z).extractall("/kaggle/tmp/code_zip/" + Path(z).stem)
    code = [Path(p).parents[1] for p in glob.glob("/kaggle/tmp/code_zip/**/teach/run.py", recursive=True)]
shutil.copytree(code[0], ROOT / "seen_never_said", dirs_exist_ok=True)
off = [Path(p).parent for p in glob.glob("/kaggle/input/**/ollama_vlm_icl.py", recursive=True)][0]
shutil.copytree(off, ROOT / "offline", dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.jsonl"))
for f in glob.glob(str(off) + "/**/*.jsonl", recursive=True):          # partial results: resume
    if not (OUT / Path(f).name).exists():
        shutil.copy(f, OUT / Path(f).name)
(ROOT / "kaggle" / "out").mkdir(parents=True, exist_ok=True)
os.symlink(OUT, ROOT / "kaggle" / "out" / "offline")
print("root ready:", sorted(p.name for p in ROOT.iterdir()), "| resumed:", sorted(p.name for p in OUT.iterdir()),
      flush=True)

subprocess.run("apt-get install -y -q zstd >/dev/null 2>&1; curl -fsSL https://ollama.com/install.sh | sh",
               shell=True, check=True)
server = subprocess.Popen(["ollama", "serve"], stdout=open("/kaggle/tmp/ollama.log", "w"), stderr=subprocess.STDOUT,
                          env={**os.environ, "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_KEEP_ALIVE": "24h"})
for _ in range(60):
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        break
    except Exception:
        time.sleep(2)
for model in env("MODEL", "qwen2.5vl:3b").split(","):
    subprocess.run(["ollama", "pull", model], check=True, stdout=subprocess.DEVNULL)
    print("pulled", model, flush=True)
    t = time.time()
    subprocess.run([sys.executable, str(ROOT / "offline" / env("SCRIPT", "ollama_vlm_icl.py")), model,
                    *env("ARGS", "all").split()], cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(f"finished {model} in {(time.time() - t) / 3600:.2f} h", flush=True)
server.terminate()
