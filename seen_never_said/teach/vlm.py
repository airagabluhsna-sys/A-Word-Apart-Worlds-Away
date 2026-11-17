"""VLM loading and scoring: Qwen2.5-VL, and any Hugging Face image-text-to-text model (SmolVLM2, 2026-11-09).

Model-specific details come from the model's own chat template (template_parts) and from encode(),
which passes images flat (Qwen) or as one list per row (Idefics3/SmolVLM)."""

import torch
from PIL import Image

from ..eval.items import LETTERS, prompt


def load(model_id, dtype="float32"):
    # T4s have no bf16 and Qwen2.5-VL can overflow in fp16, so fp32 is the default
    from transformers import AutoProcessor
    if "Qwen2.5-VL" in model_id:
        from transformers import Qwen2_5_VLForConditionalGeneration as M
    else:
        from transformers import AutoModelForImageTextToText as M
    model = M.from_pretrained(model_id, torch_dtype=getattr(torch, dtype), device_map="auto")
    proc = AutoProcessor.from_pretrained(model_id)
    configure(proc)
    return model, proc


def configure(proc):
    """One 384 px tile per picture for Idefics3/SmolVLM: splitting a 448 px drawing gives 17 tiles (1,428
    tokens) for no gain on simple drawings, and a fixed token count keeps image positions aligned across rows."""
    ip = getattr(proc, "image_processor", None)
    if ip is not None and hasattr(ip, "do_image_splitting"):
        ip.do_image_splitting = False
    return proc


def nested_images(proc):
    return "Qwen2" not in type(proc).__name__


def encode(proc, texts, row_images):
    """row_images: one list of PIL images per text."""
    if not any(row_images):
        images = None
    elif nested_images(proc):
        images = [list(r) for r in row_images]
    else:
        images = [im for r in row_images for im in r]
    return proc(text=texts, images=images, return_tensors="pt", padding=True)


def template_parts(proc):
    """(marker token ids that open the assistant answer, end-of-turn token id), from the chat template:
    Qwen "<|im_start|>assistant\n" ... "<|im_end|>", SmolVLM "Assistant:" ... "<end_of_utterance>"."""
    if getattr(proc, "_snsaid_parts", None):
        return proc._snsaid_parts
    u = [{"role": "user", "content": [{"type": "text", "text": "Q"}]}]
    t0 = proc.apply_chat_template(u, tokenize=False)
    t1 = proc.apply_chat_template(u, tokenize=False, add_generation_prompt=True)
    assert t1.startswith(t0), (t0, t1)
    t2 = proc.apply_chat_template(u + [{"role": "assistant", "content": [{"type": "text", "text": "ZQZ"}]}],
                                  tokenize=False)
    marker = proc.tokenizer.encode(t1[len(t0):], add_special_tokens=False)
    end_id = proc.tokenizer.encode(t2[t2.rindex("ZQZ") + 3:], add_special_tokens=False)[0]
    proc._snsaid_parts = (marker, end_id)
    return proc._snsaid_parts


def letter_ids(proc):
    return [proc.tokenizer.encode(L, add_special_tokens=False)[0] for L in LETTERS]


def user_turn(item, context=()):
    """context: [(image path or None, text)] teaching examples shown before the question (ICL)."""
    content = []
    for img, text in context:
        if img is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": text + "\n"})
    if item.get("image"):
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt(item)})
    return [{"role": "user", "content": content}]


@torch.no_grad()
def score(model, proc, items, open_image, contexts=None, batch_size=8):
    """Returns one row per item: pred index, option probabilities, correct, nan.

    contexts: optional {item id: [(image, text)]} for in-context teaching.
    """
    model.eval()
    proc.tokenizer.padding_side = "left"
    ids = letter_ids(proc)
    rows = []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        texts, images = [], []
        for it in chunk:
            ctx = (contexts or {}).get(it["id"], [])
            texts.append(proc.apply_chat_template(user_turn(it, ctx), tokenize=False, add_generation_prompt=True))
            images.append([open_image(p) for p, _ in ctx if p is not None] +
                          ([open_image(it["image"])] if it.get("image") else []))
        batch = encode(proc, texts, images).to(model.device)
        # only the last position is needed; full-sequence logits (tokens x 151k vocab)
        # crashed the GPU on long in-context prompts
        logits = model(**batch, logits_to_keep=1).logits[:, -1].float()
        for it, row in zip(chunk, logits):
            p = torch.softmax(row[ids[:len(it["options"])]], -1)
            pred = int(p.argmax())
            rows.append({"id": it["id"], "pred": pred, "correct": pred == it["answer"],
                         "dont_know": pred == len(it["options"]) - 1, "p": [round(x, 4) for x in p.tolist()],
                         "nan": bool(torch.isnan(row).any())})
    return rows


def _answer_span(ids, marker, end_id):
    """(start, end) of the assistant answer: after the last marker, before the last end-of-turn token."""
    m = len(marker)
    start = next(i + m for i in range(len(ids) - m, -1, -1) if ids[i:i + m] == marker)
    end = max(i for i, t in enumerate(ids) if t == end_id)
    return start, end


def decoder_layers(model):
    """The language model's decoder layers (the shortest-named ModuleList called *layers outside the vision tower)."""
    found = [(n, m) for n, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and n.endswith("layers") and "visual" not in n
             and "vision" not in n]
    return min(found, key=lambda x: len(x[0]))[1]


@torch.no_grad()
def cloze(model, proc, probes, open_image, batch_size=16, patch=None, swap_layer=None):
    """Sum log-probability of each candidate answer; one row per probe.

    The answer is the assistant turn, located in each tokenized row between the
    last "<|im_start|>assistant\n" and the final <|im_end|>. With left padding it
    sits near the end of every row, so only the last positions' logits are kept.

    patch: optional (layer, vectors), vectors[i] a hidden state for probe i, [d] or [m, d]. The residual
    stream after decoder layer `layer` (0-based) is replaced at the last prompt position and, for [m, d], the
    next m-1 positions too: the shared start of the candidates ("It eats"), so that the position that predicts
    the value word is patched (answer_prefix gives m). With m = 1 only the position before the answer
    sentence is patched, which leaves the value word to be predicted from unpatched positions (2026-11-09).

    swap_layer: k >= 1. Every probe needs "partner_image". Each row is run next to a copy with the partner's
    picture (same text, so the same token positions), and after decoder layer k the row's image-token states
    are replaced by the copy's. Only the original rows are scored (eval/bridge.py, interaction).
    """
    model.eval()
    proc.tokenizer.padding_side = "left"
    marker, end_id = template_parts(proc)
    flat = [(i, j) for i, p in enumerate(probes) for j in range(len(p["candidates"]))]
    scores = [[None] * len(p["candidates"]) for p in probes]
    state = {}
    handle = None
    if patch is not None:
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            d = hs.device                                      # layers may sit on another GPU (device_map="auto")
            hs[state["rows"].to(d), state["pos"].to(d)] = state["vecs"].to(d, hs.dtype)
            return output
        handle = decoder_layers(model)[patch[0]].register_forward_hook(hook)
    if swap_layer is not None:
        img_id = model.config.image_token_id
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            m = state["img"].to(hs.device)
            b = m.shape[0] // 2
            hs[:b][m[:b]] = hs[b:][m[b:]]
            return output
        handle = decoder_layers(model)[swap_layer - 1].register_forward_hook(hook)
        batch_size = max(1, batch_size // 2)
    try:
        for s in range(0, len(flat), batch_size):
            chunk = flat[s:s + batch_size]
            texts, images = [], []
            for i, j in chunk:
                p = probes[i]
                content = ([{"type": "image"}] if p["image"] else []) + [{"type": "text", "text": p["prompt"]}]
                msgs = [{"role": "user", "content": content},
                        {"role": "assistant", "content": [{"type": "text", "text": p["candidates"][j]}]}]
                texts.append(proc.apply_chat_template(msgs, tokenize=False))
                images.append([open_image(p["image"])] if p["image"] else [])
            if swap_layer is not None:
                texts = texts + texts
                images = images + [[open_image(probes[i]["partner_image"])] for i, _ in chunk]
            batch = encode(proc, texts, images).to(model.device)
            input_ids = batch["input_ids"]
            L = input_ids.shape[1]
            spans = [_answer_span(row.tolist(), marker, end_id) for row in input_ids]
            keep = max(L - start for start, _ in spans) + 1            # every answer token's predicting position
            if swap_layer is not None:
                state["img"] = input_ids == img_id
                b = len(chunk)
                assert torch.equal(state["img"][:b], state["img"][b:]), "partner rows tokenize differently"
            if patch is not None:
                rows_, pos_, vecs_ = [], [], []
                for r, ((i, j), (start, _)) in enumerate(zip(chunk, spans)):
                    v = patch[1][i]
                    v = v[None] if v.dim() == 1 else v
                    for k in range(v.shape[0]):
                        rows_.append(r)
                        pos_.append(start - 1 + k)
                    vecs_.append(v)
                    if v.shape[0] > 1:     # the patched answer positions must be the shared prefix tokens
                        pre = probes[i].get("prefix_ids")
                        assert pre is None or input_ids[r, start:start + v.shape[0] - 1].tolist() == pre, \
                            "patched positions are not the answer prefix"
                state["rows"] = torch.tensor(rows_, device=model.device)
                state["pos"] = torch.tensor(pos_, device=model.device)
                state["vecs"] = torch.cat(vecs_).to(model.device)
            logp = torch.log_softmax(model(**batch, logits_to_keep=keep).logits.float(), -1)
            _sum_answer(scores, chunk, spans, logp, input_ids, L, keep)
    finally:
        if handle is not None:
            handle.remove()
    rows = []
    for p, sc in zip(probes, scores):
        pred = max(range(len(sc)), key=sc.__getitem__)
        rows.append({"id": p["id"], "pred": pred, "correct": pred == p["answer"], "dont_know": False,
                     "scores": [round(x, 4) for x in sc], "nan": any(x != x for x in sc)})
    return rows


def _sum_answer(scores, chunk, spans, logp, input_ids, L, keep):
    for row, ((i, j), (start, end)) in enumerate(zip(chunk, spans)):
        # token at position t is predicted by the logits at t-1, kept at index t-1-(L-keep)
        scores[i][j] = float(sum(logp[row, t - 1 - (L - keep), input_ids[row, t]] for t in range(start, end)))


def answer_prefix(proc, candidates):
    """(prefix text, prefix token ids) shared by every candidate answer; m = len(ids) + 1 positions to patch."""
    toks = [proc.tokenizer.encode(c, add_special_tokens=False) for c in candidates]
    n = 0
    while all(len(t) > n for t in toks) and len({t[n] for t in toks}) == 1:
        n += 1
    return proc.tokenizer.decode(toks[0][:n]), toks[0][:n]


@torch.no_grad()
def last_states(model, proc, items, open_image, lens_ids=(), batch_size=8, n_last=1):
    """Residual stream at the last prompt position (the token before the answer), every layer.

    items: dicts with "prompt", optional "image" and optional "prefix" (the start of the answer, so the
    last position predicts what follows it). Returns (states [n, layers+1, d]
    float16 on CPU; index k = after k decoder layers, the last one after the final
    norm) and, if lens_ids is given, the logit lens: log-softmax over the vocabulary
    of norm(state) @ lm_head, kept at lens_ids [n, layers+1, len(lens_ids)].
    n_last > 1: the last n_last positions, states [n, layers+1, n_last, d] (no lens).
    """
    model.eval()
    proc.tokenizer.padding_side = "left"
    head = model.get_output_embeddings()
    norm = min(((n, m) for n, m in model.named_modules() if n.endswith("norm") and "layers" not in n
                and "visual" not in n and "vision" not in n), key=lambda x: len(x[0]))[1]
    ids = torch.tensor(list(lens_ids), dtype=torch.long)
    states, lens = [], []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        texts, images = [], []
        for it in chunk:
            content = ([{"type": "image"}] if it.get("image") else []) + [{"type": "text", "text": it["prompt"]}]
            texts.append(proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                  add_generation_prompt=True) + it.get("prefix", ""))
            images.append([open_image(it["image"])] if it.get("image") else [])
        batch = encode(proc, texts, images).to(model.device)
        out = model(**batch, output_hidden_states=True, logits_to_keep=1)
        if n_last > 1:
            states.append(torch.stack([x[:, -n_last:] for x in out.hidden_states], 1).half().cpu())
            continue
        h = torch.stack([x[:, -1] for x in out.hidden_states], 1)    # [b, layers+1, d]
        states.append(h.half().cpu())
        if len(ids):
            # intermediate layers through the final norm and head; the last layer is the model's own logits
            # (whether the last hidden state is already normed differs between transformers versions)
            mid = head(norm(h[:, :-1].to(norm.weight.dtype)).to(head.weight.dtype)).float()
            lp = torch.log_softmax(torch.cat([mid, out.logits[:, -1:].float()], 1), -1)
            lens.append(lp[..., ids.to(lp.device)].cpu())
    return torch.cat(states).numpy(), (torch.cat(lens).numpy() if lens else None)


def image_opener(root):
    cache = {}

    def open_image(rel):
        if rel not in cache:
            cache[rel] = Image.open(root / rel).convert("RGB")
        return cache[rel]
    return open_image
