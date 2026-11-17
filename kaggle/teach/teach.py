"""Teaching run on Kaggle (GPU): LoRA or in-context, one regime and seed -- or a batch of such jobs.

Settings come from environment defaults baked in by push.py --set:
  REGIME=V|T|S  METHOD=lora|icl|eval|layers|bridge (eval: score adapters of attached runs; layers: layer-wise
  probes and back-patching on one attached S adapter, PAIRS=i-j,... BUDGET_H=8 SAVE_STATES=0)  SEED=0  EPOCHS=8  MASTERY=0.95  RANK=16  LR=2e-4
  MODEL=Qwen/Qwen2.5-VL-3B-Instruct | HuggingFaceTB/SmolVLM2-2.2B-Instruct | any image-text-to-text model  DTYPE=float32  SMOKE=0  FORMAT=1 (distractor format practice)
Writes /kaggle/working/{eval.jsonl, run.json, adapter/}.

Batch mode (so a queue runs entirely on Kaggle, with no local driver):
  JOBS="name1@est_h:KEY=VAL,KEY=VAL;name2@est_h:KEY=VAL"  BATCH_H=11.6
Runs the jobs in order in this one session, each with its settings added to the environment and writing to
/kaggle/working/<name>/. A job whose estimated hours no longer fit in BATCH_H is skipped; a failing job is
logged and the batch moves on. Progress is in /kaggle/working/batch.json.
"""

import gc
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

T0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.51", "accelerate", "peft",
                "num2words"], check=True)                  # num2words: the SmolVLM processor needs it
# Kaggle's preinstalled torchao is older than peft accepts, and peft fails on import
# of its LoRA dispatchers if it is present at all; nothing here uses it.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

env = os.environ.get


# --- locate the dataset and the code package (same logic as readout.py) ------
# Two datasets may be attached: the creatures (snsaid-tier0-pilot) and the real photos (snsaid-real-v1, which has
# probes_K.jsonl). METHOD=real works on the photos; everything else on the creatures.
def _unpack(src, work):
    done = work / ".unpacked"
    if done.exists():                       # batch mode: the second job reuses the first job's unpacking
        return work
    work.mkdir(parents=True, exist_ok=True)
    for z in src.glob("*.zip"):
        zipfile.ZipFile(z).extractall(work / z.stem)
    for f in src.glob("*.jsonl"):
        (work / f.name).write_bytes(f.read_bytes())
    for d in [p.name for p in work.iterdir() if p.is_dir()]:
        if (work / d / d).exists():
            (work / d / d).rename(work / f"_{d}")
            (work / d).rmdir()
            (work / f"_{d}").rename(work / d)
    done.write_text("1")
    return work


def _adapter_glob(pat, recursive=True):
    """Adapters come from attached kernel outputs, or from a dataset (snsaid-s-adapters) whose folders
    Kaggle may deliver as zips; unzip those once into /kaggle/tmp/adapters."""
    found = glob.glob("/kaggle/input/" + pat, recursive=recursive)
    if found:
        return found
    dst = Path("/kaggle/tmp/adapters")
    if not dst.exists():
        for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
            names = zipfile.ZipFile(z).namelist()
            if any(n.endswith("adapter_config.json") for n in names):
                zipfile.ZipFile(z).extractall(dst / Path(z).stem)
    return glob.glob(str(dst) + "/" + pat, recursive=recursive)


def _run_dirs():
    return sorted({str(Path(p).parents[1]) for p in _adapter_glob("**/adapter/adapter_config.json", recursive=True)})


def _data_root():
    found = [Path(p).parent for p in sorted(glob.glob("/kaggle/input/**/concepts.jsonl", recursive=True))
             if "tier0_R" not in p]
    real = [p for p in found if (p / "probes_K.jsonl").exists()]
    tier0 = [p for p in found if not (p / "probes_K.jsonl").exists()]
    if env("METHOD") == "real":
        root = real[0]
        if not (root / "images").exists():
            root = _unpack(root, Path("/kaggle/tmp/real"))
    else:
        root = tier0[0]
        if not (root / "images").exists():
            root = _unpack(root, Path("/kaggle/tmp/data"))
        if env("OVERRIDE") == "R":       # arm R: creatures renamed with known words (real/build.py)
            # Kaggle sometimes delivers the dataset already unzipped: then tier0_R is a folder in the input
            hit = sorted(glob.glob("/kaggle/input/**/tier0_R/**/concepts.jsonl", recursive=True))
            ov = Path(hit[0]).parent if hit else _unpack(real[0], Path("/kaggle/tmp/real")) / "tier0_R"
            os.environ["SNSAID_OVERRIDE"] = str(ov)
            print("override:", ov, sorted(p.name for p in ov.iterdir()), flush=True)
    print("data root:", root, flush=True)
    return root


def _load_code(root):
    code = []
    for dirpath, dirs, files in os.walk("/kaggle/input", followlinks=True):
        # the data dataset ships its own (older) copy of the package; use snsaid-code
        if dirpath.endswith("/teach") and "run.py" in files and not dirpath.startswith(str(root)):
            code.append(Path(dirpath).parent)
    # prefer a complete package (with generator/), and snsaid-code over copies inside data datasets
    code.sort(key=lambda p: (not (p / "generator").is_dir(), "snsaid-code" not in str(p), str(p)))
    if not code:
        for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
            if not z.startswith(str(root)) and any(n.endswith("teach/run.py") for n in zipfile.ZipFile(z).namelist()):
                zipfile.ZipFile(z).extractall("/kaggle/tmp/code_zip")
                code = [Path(c).parents[1] for c in glob.glob("/kaggle/tmp/code_zip/**/teach/run.py", recursive=True)]
                break
    if not code:
        raise SystemExit("teaching code not found: push with --code")
    pkg = Path("/kaggle/tmp/code/seen_never_said")          # outside /kaggle/working, so not in the output
    if not pkg.exists():
        shutil.copytree(code[0], pkg, dirs_exist_ok=True)
        sys.path.insert(0, str(pkg.parent))
    print("code package:", code[0], flush=True)


def job(OUT):
    OUT.mkdir(parents=True, exist_ok=True)
    root = _data_root()
    _load_code(root)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    model = env("MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    method = env("METHOD", "lora")
    if method == "real":
        # real-photo arms K / A (teach/real_job.py, real/spec.py)
        from seen_never_said.teach.real_job import real_job
        real_job(root, OUT, arm=env("ARM", "K"), model_id=model, dtype=env("DTYPE", "float32"),
                 epochs=int(env("EPOCHS", 3)), seed=int(env("SEED", 0)), smoke=env("SMOKE", "0") == "1", log=log)
        return
    if method in ("eval", "mirror", "latent", "bridge"):
        # score the adapters of finished runs attached as kernel sources (push.py --kernel-source) or a dataset
        import importlib
        mod, name = {"eval": ("evaluate", "evaluate_adapters"), "mirror": ("mirror_job", "mirror_job"),
                     "latent": ("latent_job", "latent_job"), "bridge": ("bridge_job", "bridge_job")}[method]
        fn = getattr(importlib.import_module(f"seen_never_said.teach.{mod}"), name)
        run_dirs = _run_dirs()
        print("adapters:", run_dirs, flush=True)
        fn(root, OUT, run_dirs, model_id=model, dtype=env("DTYPE", "float16"), log=log)
        return
    if method == "gen":
        # free generation on the composition probes, untaught model + every attached adapter (teach/gen_job.py)
        from seen_never_said.teach.gen_job import gen_job
        run_dirs = _run_dirs()
        print("adapters:", run_dirs, flush=True)
        gen_job(root, OUT, run_dirs, model_id=model, dtype=env("DTYPE", "float16"),
                smoke=env("SMOKE", "0") == "1", log=log)
        return
    if method == "relabel":
        # relabel overwrite on every attached S adapter: half the test creatures renamed from pictures alone
        from seen_never_said.teach.relabel_job import relabel_job
        run_dirs = _run_dirs()
        print("adapters:", run_dirs, flush=True)
        relabel_job(root, OUT, run_dirs, model_id=model, dtype=env("DTYPE", "float16"),
                    epochs=int(env("EPOCHS", 1)), smoke=env("SMOKE", "0") == "1", log=log)
        return
    if method == "consolidate":
        # self-rehearsal on every attached S adapter: the model labels its own pictures through the written chain,
        # then learns the silent answers (teach/consolidate_job.py)
        from seen_never_said.teach.consolidate_job import consolidate_job
        run_dirs = _run_dirs()
        print("adapters:", run_dirs, flush=True)
        consolidate_job(root, OUT, run_dirs, model_id=model, dtype=env("DTYPE", "float16"),
                        epochs=int(env("EPOCHS", 2)), smoke=env("SMOKE", "0") == "1", log=log)
        return
    if method == "distill":
        # label distillation on every attached S adapter: the clean picture is pulled toward the same picture with
        # its name printed on it, for half the test creatures; no fact is trained (teach/distill_job.py)
        from seen_never_said.teach.distill_job import distill_job
        run_dirs = [d for d in _run_dirs() if json.loads((Path(d) / "run.json").read_text())["regime"] == "S"]
        if env("SEEDS", ""):
            keep = {int(x) for x in env("SEEDS").split("+")}
            run_dirs = [d for d in run_dirs if json.loads((Path(d) / "run.json").read_text())["seed"] in keep]
        print("adapters:", run_dirs, flush=True)
        distill_job(root, OUT, run_dirs, model_id=model, dtype=env("DTYPE", "float16"),
                    epochs=int(env("EPOCHS", 6)), eval_every=int(env("EVAL_EVERY", 3)), n=int(env("DN", 4)),
                    replay=int(env("REPLAY", 2)), weight=float(env("DW", 1.0)), vis=env("VIS", "0") == "1",
                    pics_per=int(env("PICS", 32)), smoke=env("SMOKE", "0") == "1", log=log)
        return
    if method == "patch2":
        # corrected back-patching and text -> picture patching on one attached S adapter (teach/patch_job.py)
        from seen_never_said.teach.patch_job import patch_job
        run_dirs = _run_dirs()
        print("adapter:", run_dirs, flush=True)
        patch_job(root, OUT, run_dirs[0], model_id=model, dtype=env("DTYPE", "float16"), log=log)
        return
    if method == "layers":
        # layer-wise probes and back-patching on one attached S adapter (eval/layers.py)
        from seen_never_said.teach.layers_job import layers_job
        run_dirs = _run_dirs()
        print("adapter:", run_dirs, flush=True)
        pairs = [tuple(map(int, x.split("-"))) for x in env("PAIRS", "").split(",") if x] or None
        layers_job(root, OUT, run_dirs[0], model_id=model, dtype=env("DTYPE", "float16"),
                   save_states=env("SAVE_STATES", "0") == "1", pairs=pairs, fast=env("FAST", "0") == "1",
                   budget_s=float(env("BUDGET_H", 8)) * 3600, log=log)
        return
    from seen_never_said.teach.run import run
    run(root, OUT, model_id=model, regime=env("REGIME", "V"),
        method=method, seed=int(env("SEED", 0)), epochs=int(env("EPOCHS", 8)),
        mastery=float(env("MASTERY", 0.95)), rank=int(env("RANK", 16)), lr=float(env("LR", 2e-4)),
        batch_size=int(env("BATCH", 4)), dtype=env("DTYPE", "float32"), smoke=env("SMOKE", "0") == "1",
        format_practice=env("FORMAT", "1") == "1", log=log)


def batch(spec, out=Path("/kaggle/working")):
    budget = float(env("BATCH_H", 11.6)) * 3600
    status = []
    for item in [x for x in spec.split(";") if x.strip()]:
        head, _, kv = item.strip().partition(":")
        name, _, est = head.partition("@")
        est_s = float(est or 0) * 3600
        rec = {"job": name, "est_h": float(est or 0), "settings": kv}
        left = budget - (time.time() - T0)
        if est_s > left:
            rec.update(state="skipped", reason=f"needs {est_s / 3600:.1f} h, {left / 3600:.1f} h left")
        else:
            saved = dict(os.environ)
            for e in [x for x in kv.split(",") if x]:
                k, v = e.split("=", 1)
                os.environ[k] = v
            print(f"\n===== JOB {name} ({kv}) at {(time.time() - T0) / 3600:.2f} h =====", flush=True)
            t = time.time()
            try:
                job(out / name)
                rec["state"] = "complete"
            except BaseException as e:      # SystemExit included: one bad job must not end the batch
                traceback.print_exc()
                rec.update(state="error", reason=repr(e)[:500])
            rec["hours"] = round((time.time() - t) / 3600, 3)
            os.environ.clear()
            os.environ.update(saved)
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        print(f"===== {name}: {rec} =====", flush=True)
        status.append(rec)
        (out / "batch.json").write_text(json.dumps(status, indent=1))


if env("JOBS"):
    batch(env("JOBS"))
else:
    job(Path("/kaggle/working"))
