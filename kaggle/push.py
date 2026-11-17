"""Upload a built dataset to Kaggle and launch a kernel on it.

    py -3.11 kaggle/push.py --data data/tier0_pilot --kernel kaggle/readout
    py -3.11 kaggle/push.py --kernel kaggle/readout --skip-data        # re-run only

The dataset ships with a copy of the generator package so kernels can import
the ontology. "USERNAME" in kernel-metadata.json is filled from the Kaggle
credentials. Needs ~/.kaggle/kaggle.json (or KAGGLE_USERNAME / KAGGLE_KEY).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# SNSAID_CODE_DIR pins the uploaded package to a frozen copy, so a queue of runs
# keeps one code version while the working tree changes.
PKG = Path(os.environ.get("SNSAID_CODE_DIR", REPO / "seen_never_said"))


def kaggle(*args, tries=6):
    """Run a Kaggle CLI command, retrying: the build machine's network drops often."""
    cmd = [sys.executable, "-m", "kaggle.cli", *args]
    print("+ kaggle", " ".join(args), flush=True)
    for attempt in range(tries):
        try:                                                    # a dropped connection can hang the CLI for hours
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=1800 if args[:2] in (("datasets", "create"), ("datasets", "version")) else 300)
        except subprocess.TimeoutExpired:
            print(f"  attempt {attempt + 1} timed out", flush=True)
            time.sleep(15 * (attempt + 1))
            continue
        if r.returncode == 0:
            return r.stdout
        if "session count" in (r.stdout + r.stderr).lower():    # a refusal, not a network drop: let the caller wait
            return r.stdout + r.stderr
        print(f"  attempt {attempt + 1} failed: {(r.stderr or r.stdout).strip()[-200:]}", flush=True)
        time.sleep(15 * (attempt + 1))
    raise SystemExit(f"kaggle {' '.join(args)} failed {tries} times")


def wait_ready(ref, timeout=1800):
    """Kernels can't attach a dataset that is still being processed."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        status = kaggle("datasets", "status", ref).strip()
        print(f"  dataset status: {status}", flush=True)
        if status == "ready":
            return
        if status not in ("pending", "running"):
            raise SystemExit(f"dataset {ref} failed: {status}")
        time.sleep(30)
    raise SystemExit(f"dataset {ref} not ready after {timeout}s")


def username(override=None):
    if override or os.environ.get("KAGGLE_USERNAME"):
        return override or os.environ["KAGGLE_USERNAME"]
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not cfg.exists():
        raise SystemExit("no ~/.kaggle/kaggle.json; pass --user with an access-token login")
    return json.loads(cfg.read_text())["username"]


def push_data(data_dir, slug, user, stage_root):
    stage = Path(stage_root) / "data"
    shutil.copytree(data_dir, stage)
    shutil.copytree(PKG, stage / "seen_never_said",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (stage / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}))
    existing = kaggle("datasets", "list", "--mine", "--search", slug, "--csv")
    if f"{user}/{slug}" in existing:
        print(kaggle("datasets", "version", "-p", str(stage), "-m", "rebuild", "--dir-mode", "zip"))
    else:
        print(kaggle("datasets", "create", "-p", str(stage), "--dir-mode", "zip"))


def push_code(slug, user, stage_root):
    """Small dataset with just the generator package, so kernels can render with
    the latest code without re-uploading the image data."""
    stage = Path(stage_root) / "code"
    shutil.copytree(PKG, stage / "seen_never_said",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (stage / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}))
    existing = kaggle("datasets", "list", "--mine", "--search", slug, "--csv")
    if f"{user}/{slug}" in existing:
        print(kaggle("datasets", "version", "-p", str(stage), "-m", "code update", "--dir-mode", "zip"))
    else:
        print(kaggle("datasets", "create", "-p", str(stage), "--dir-mode", "zip"))


def push_kernel(kernel_dir, user, stage_root, name=None, env=(), extra_data=(), wait_slot=False, kernel_sources=()):
    """`env` lines (KEY=VALUE) are baked into the script as environment defaults,
    since Kaggle kernels can't be given environment variables."""
    stage = Path(stage_root) / "kernel"
    shutil.copytree(kernel_dir, stage)
    meta_path = stage / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text().replace("USERNAME", user))
    if name:
        meta["id"], meta["title"] = f"{user}/{name}", name.replace("-", " ")
    meta["dataset_sources"] += [f"{user}/{d}" for d in extra_data]
    meta["kernel_sources"] += [f"{user}/{k}" for k in kernel_sources]      # outputs of earlier runs
    meta_path.write_text(json.dumps(meta, indent=2))
    if env:
        code = stage / meta["code_file"]
        header = "import os\n" + "".join(f"os.environ.setdefault({k!r}, {v!r})\n"
                                         for k, v in (e.split("=", 1) for e in env))
        code.write_text(header + code.read_text(encoding="utf-8"), encoding="utf-8")
    # The CLI exits 0 even when the push is refused, so check what it printed.
    # Kaggle allows 2 GPU sessions at a time; with wait_slot, retry until one frees up.
    for attempt in range(360 if wait_slot else 1):      # up to 12 h
        out = kaggle("kernels", "push", "-p", str(stage))
        print(out)
        low = out.lower()
        if "authentication required" in low or "kaggle auth login" in low:   # expired OAuth login
            raise SystemExit("kernel push refused: Kaggle login expired (run `kaggle auth login`)")
        if "error" not in low and "pushed" in low:      # success prints "Kernel version N successfully pushed"
            return meta["id"]
        if "session count" not in out.lower() or not wait_slot:
            break
        print("  no free GPU slot; retrying in 2 min", flush=True)
        time.sleep(120)
    raise SystemExit(f"kernel push refused: {out.strip()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tier0_pilot")
    ap.add_argument("--slug", default="snsaid-tier0-pilot")
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--user", help="Kaggle username, if not in kaggle.json")
    ap.add_argument("--name", help="kernel slug, to run a variant as its own kernel")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="script environment default")
    ap.add_argument("--code", action="store_true", help="upload the generator package as dataset snsaid-code")
    ap.add_argument("--kernel-source", action="append", default=[], help="attach a finished kernel's output")
    ap.add_argument("--wait-slot", action="store_true", help="retry while Kaggle's 2 GPU sessions are busy")
    ap.add_argument("--extra-data", action="append", default=[], help="attach another of your datasets (slug)")
    args = ap.parse_args()
    user = username(args.user)
    with tempfile.TemporaryDirectory() as tmp:
        if not args.skip_data:
            push_data(REPO / args.data, args.slug, user, tmp)
        wait_ready(f"{user}/{args.slug}")
        extra = list(args.extra_data)
        if args.code:
            push_code("snsaid-code", user, tmp)
            wait_ready(f"{user}/snsaid-code")
            extra.append("snsaid-code")
        kid = push_kernel(REPO / args.kernel, user, tmp, args.name, args.set, extra, args.wait_slot,
                          args.kernel_source)
    print(f"launched {kid}\n  status: py -3.11 -m kaggle.cli kernels status {kid}"
          f"\n  output: py -3.11 -m kaggle.cli kernels output {kid} -p kaggle/out/{kid.split('/')[1]}")


if __name__ == "__main__":
    main()
