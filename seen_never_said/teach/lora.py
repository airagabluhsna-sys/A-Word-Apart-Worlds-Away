"""LoRA teaching to matched mastery (proposal §6.2-6.3).

Train on one regime's examples, check in-modality mastery on held-out items after
every epoch, and stop at the first epoch that reaches the threshold. The test set
is scored at every epoch too, so transfer can be read along the training curve
(§9.6: tau should rise during training when the shared component is strong).
"""

import random
import time

import torch
from PIL import Image

# Language-model attention and MLP only; the vision tower and projector stay frozen.
# Matches both module layouts (model.layers... and model.language_model.layers...);
# vision blocks are named visual.blocks.N and so never match.
# Vision encoders that name their attention q/k/v_proj too (SmolVLM: model.vision_model.encoder.layers.N) are
# excluded by the lookahead.
TARGET = r"^(?!.*(visual|vision)).*\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def add_lora(model, rank=16, alpha=32, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=TARGET))
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()       # LoRA weights in fp32 even when the base is fp16
    model.print_trainable_parameters()
    return model


def _assistant_start(ids, marker):
    """Index just after the last assistant marker (vlm.template_parts) in a token list."""
    m = len(marker)
    for i in range(len(ids) - m, -1, -1):
        if ids[i:i + m] == marker:
            return i + m
    raise ValueError("assistant marker not found")


def collate(proc, examples, root):
    proc.tokenizer.padding_side = "right"
    texts, images = [], []
    for ex in examples:
        content = ([{"type": "image"}] if ex["image"] else []) + [{"type": "text", "text": ex["prompt"]}]
        msgs = [{"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]}]
        texts.append(proc.apply_chat_template(msgs, tokenize=False))
        images.append([Image.open(root / ex["image"]).convert("RGB")] if ex["image"] else [])
    from .vlm import encode, template_parts
    batch = encode(proc, texts, images)
    marker, _ = template_parts(proc)
    labels = batch["input_ids"].clone()
    for row, ids in enumerate(batch["input_ids"].tolist()):
        labels[row, :_assistant_start(ids, marker)] = -100
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch


def train(model, proc, examples, root, evaluate, *, epochs, mastery, batch_size=4, lr=2e-4, seed=0,
          dtype="float32", aux=None, log=print):
    """evaluate(epoch) -> mastery accuracy (and writes its own eval rows).
    aux: optional callable(model) -> extra loss added at every step (teach/binding.py, regime Sb).

    Returns a training record: epochs and steps to mastery (None if not reached).
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == "float16")
    record = {"epochs": [], "mastered_at_epoch": None, "steps_to_mastery": None}
    acc = evaluate(0)
    record["epochs"].append({"epoch": 0, "step": 0, "mastery": acc})
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = examples[:]
        rng.shuffle(order)
        t0, losses = time.time(), []
        for s in range(0, len(order), batch_size):
            batch = collate(proc, order[s:s + batch_size], root).to(model.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dtype == "float16"):
                loss = model(**batch).loss
                if aux is not None:
                    loss = loss + aux(model)
                    model.train()
            if not torch.isfinite(loss):
                log(f"  non-finite loss at step {step}; batch skipped")
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                log(f"  epoch {epoch} step {step} loss {sum(losses[-50:]) / len(losses[-50:]):.4f}" +
                    (f" (binding {aux.last:.4f})" if aux is not None and getattr(aux, "last", None) is not None
                     else ""))
        acc = evaluate(epoch)
        record["epochs"].append({"epoch": epoch, "step": step, "mastery": acc,
                                 "loss": sum(losses) / max(len(losses), 1), "seconds": round(time.time() - t0)})
        log(f"epoch {epoch}: loss {record['epochs'][-1]['loss']:.4f}, mastery {acc:.3f}")
        if acc >= mastery:
            record["mastered_at_epoch"], record["steps_to_mastery"] = epoch, step
            break
    return record
