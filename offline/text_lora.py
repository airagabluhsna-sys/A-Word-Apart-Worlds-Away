"""Within-text control of the cross-modal two-hop curse, trained into the weights (CPU, Qwen2.5-0.5B-Instruct).

The split regime stores picture -> name (pictures) and name -> property (text). Here both hops are taught in
text, the "picture" replaced by a verbal description of the same visible attributes:
  recognition docs  "A creature that has a bean-shaped body, is glassy ... is called a kusnib."  (name last)
  fact docs         "Every kusnib is white. Every kusnib eats bread. ..."                        (name first)
Layouts:
  split      the two kinds in separate training documents (the analogue of S)
  colocated  every document holds a creature's recognition sentence and its facts (Balesni et al. 2024:
             co-occurrence lets text-only two-hop work)
Likelihood probes after each epoch, as for the VLMs: hop1 (description -> name), hop2 (name -> value),
silent (description -> value), chain (description -> "It is called a kusnib. It eats bread.").

    PYTHONIOENCODING=utf-8 py -3.13 offline/text_lora.py split 4      -> kaggle/out/offline/lora_split.jsonl
"""

import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seen_never_said.eval.textonly import ASK, description  # noqa: E402
from seen_never_said.generator.lexicon import PREDICATES  # noqa: E402
from seen_never_said.generator.ontology import VALUES  # noqa: E402
from seen_never_said.teach import data  # noqa: E402

MID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Tell me about a creature."
N_DOCS = 12


def _join(ps):
    return ps[0] if len(ps) == 1 else ", ".join(ps[:-1]) + " and " + ps[-1]


def docs(concepts, layout, seed=0):
    rng = random.Random(seed)
    out = []
    for c in concepts:
        V, T = c["split"]["V"], c["split"]["T"]
        for _ in range(N_DOCS):
            v = rng.sample(V, len(V))
            rec = (f"A creature that {_join([rng.choice(PREDICATES[a][c['attrs'][a]])[0] for a in v])} "
                   f"is called a {c['name']}.")
            t = rng.sample(T, len(T))
            fac = " ".join(f"{rng.choice(['Every', 'Each', 'Any'])} {c['name']} "
                           f"{rng.choice(PREDICATES[a][c['attrs'][a]])[0]}." for a in t)
            if layout == "colocated":
                out.append(rec + " " + fac)
            else:
                out += [rec, fac]
    rng.shuffle(out)
    return out


def probes(concepts, seed=0):
    rng = random.Random(seed + 11)
    test = [c for c in concepts if c.get("role", "test") == "test"]
    names = [c["name"] for c in test]
    out = []
    for c in test:
        d = description(c)
        out.append({"kind": "hop1", "concept": c["cid"], "attr": None, "prompt": f"{d} What is it called?",
                    "cands": [f"It is called a {n}." for n in names], "answer": names.index(c["name"])})
        for a in c["split"]["T"]:
            vals = VALUES[a]
            it = [f"It {PREDICATES[a][v][0][0]}." for v in vals]
            ans = vals.index(c["attrs"][a])
            q = ASK[a][0]
            out.append({"kind": "hop2", "concept": c["cid"], "attr": a, "prompt": q.format(w=f"a {c['name']}"),
                        "cands": it, "answer": ans})
            out.append({"kind": "silent", "concept": c["cid"], "attr": a, "prompt": f"{d} " + q.format(w="it"),
                        "cands": it, "answer": ans})
            nm = [c["name"]] + rng.sample([n for n in names if n != c["name"]], 3)
            out.append({"kind": "chain", "concept": c["cid"], "attr": a,
                        "prompt": f"{d} " + q.format(w="it") + " Say what it is called first.",
                        "cands": [f"It is called a {n}. {s}" for n in nm for s in it], "answer": ans})
    return out


def chat(tok, prompt, answer=None):
    msgs = [{"role": "user", "content": prompt}]
    if answer is None:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.apply_chat_template(msgs + [{"role": "assistant", "content": answer}], tokenize=False)


@torch.no_grad()
def score(model, tok, ps, bs=24):
    """Sum log-probability of each candidate answer (tokens after the generation prompt, up to the end)."""
    model.eval()
    flat = [(i, j) for i, p in enumerate(ps) for j in range(len(p["cands"]))]
    sc = [[0.0] * len(p["cands"]) for p in ps]
    tok.padding_side = "right"
    for s in range(0, len(flat), bs):
        ch = flat[s:s + bs]
        pre = [len(tok(chat(tok, ps[i]["prompt"]), add_special_tokens=False)["input_ids"]) for i, _ in ch]
        b = tok([chat(tok, ps[i]["prompt"], ps[i]["cands"][j]) for i, j in ch], return_tensors="pt", padding=True,
                add_special_tokens=False)
        lp = torch.log_softmax(model(**b).logits.float(), -1)
        for r, ((i, j), p0) in enumerate(zip(ch, pre)):
            n = int(b["attention_mask"][r].sum())
            ids = b["input_ids"][r]
            sc[i][j] = float(sum(lp[r, t - 1, ids[t]] for t in range(p0, n - 1)))   # drop the final newline
    return sc


def evaluate(model, tok, ps, epoch, f, layout):
    sc = score(model, tok, ps)
    res = {}
    for p, s in zip(ps, sc):
        if p["kind"] == "chain":
            k = len(VALUES[p["attr"]])
            best = max(range(len(s)), key=s.__getitem__)
            correct = best == p["answer"]                         # first name block is the true name
            pred_val = best % k
        else:
            best = max(range(len(s)), key=s.__getitem__)
            correct = best == p["answer"]
            pred_val = best
        f.write(json.dumps({"layout": layout, "epoch": epoch, "kind": p["kind"], "concept": p["concept"],
                            "attr": p["attr"], "pred": pred_val, "answer": p["answer"], "correct": correct,
                            "chance": 1 / len(s)}) + "\n")
        res.setdefault(p["kind"], []).append(correct)
    f.flush()
    return {k: round(sum(v) / len(v), 3) for k, v in res.items()}


def main(layout="split", epochs=4, smoke=False):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    torch.set_num_threads(2)
    concepts = data.load_concepts(ROOT / "data" / "tier0_pilot")
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=r".*\.(q|k|v|o|gate|up|down)_proj"))
    ds = docs(concepts, layout)
    ps = probes(concepts)
    if smoke:
        ds, ps = ds[:8], [p for p in ps if p["concept"] == "c_0000"][:5]
    out = ROOT / "kaggle" / "out" / "offline"
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / f"lora_{layout}{'_smoke' if smoke else ''}.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    print(layout, len(ds), "docs", len(ps), "probes", "epoch 0", evaluate(model, tok, ps, 0, f, layout),
          f"({time.time() - t0:.0f}s)", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    rng = random.Random(0)
    tok.padding_side = "right"
    for ep in range(1, epochs + 1):
        model.train()
        order = ds[:]
        rng.shuffle(order)
        for s in range(0, len(order), 8):
            texts = [chat(tok, PROMPT, d) for d in order[s:s + 8]]
            b = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            lab = b["input_ids"].clone()
            p0 = len(tok(chat(tok, PROMPT), add_special_tokens=False)["input_ids"])
            lab[:, :p0] = -100
            lab[b["attention_mask"] == 0] = -100
            loss = model(**b, labels=lab).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(layout, "epoch", ep, f"loss {loss.item():.3f}", evaluate(model, tok, ps, ep, f, layout),
              f"({time.time() - t0:.0f}s)", flush=True)
    f.close()


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "split", int(a[1]) if len(a) > 1 else 4, smoke="smoke" in a)
