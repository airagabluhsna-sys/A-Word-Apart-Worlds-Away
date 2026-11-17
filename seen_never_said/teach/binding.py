"""Remedy 2, early binding (regime Sb, planned 2026-11-12): teach the picture to become its name mid-network.

A latent context distillation of the bridge entity. Besides the usual LoRA loss, each step takes a few teaching
pictures (S pictures of any creature, captioned with its name: the picture -> name hop the adapter is taught
anyway) and runs two prompts through the model being trained:

  teacher  "This is a kusnib." + picture + a generic question   (name written: the lookup works, 0.74)
  student                        picture + the same question     (no name)

and pulls the student's residual stream toward the teacher's (stop-gradient) at a band of middle layers, at
the last prompt position and at the picture's own tokens (mean over image tokens). The questions are generic
("Tell me about this creature.") and nothing is trained on a fact answer, so composition is never supervised:
the objective only asks that, from the middle of the network on, seeing a creature looks like having its name.
"""

import random

import torch
import torch.nn.functional as F
from PIL import Image

QUESTIONS = ["Tell me about this creature.", "Describe this creature.", "What do you know about this creature?",
             "What is special about this creature?"]
LAYERS = (8, 12, 16, 20)          # of 36: after identity (layer 1), well before the name surfaces (~33)


class EarlyBinding:
    def __init__(self, proc, root, pictures, *, n=2, weight=0.5, layers=LAYERS, loss="cos", seed=0):
        """pictures: [(image path, name)] teaching pictures.
        loss: "cos" (1 - cosine; ~0.005 from the start because a few huge residual dimensions make all states look
        parallel) or "rmse" (relative squared error ||s - t||^2 / ||t||^2 after removing each layer's mean over the
        pair, so the shared large components cancel and the loss sees what differs)."""
        self.proc, self.root, self.pictures = proc, root, pictures
        self.n, self.weight, self.layers, self.kind = n, weight, tuple(layers), loss
        self.rng = random.Random(seed + 2718281)
        self.last = None

    def _batch(self, model, rows, with_name):
        from .vlm import encode
        texts, images = [], []
        for img, name, q in rows:
            content = ([{"type": "text", "text": f"This is a {name}."}] if with_name else []) + \
                      [{"type": "image"}, {"type": "text", "text": q}]
            texts.append(self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                       add_generation_prompt=True))
            images.append([Image.open(self.root / img).convert("RGB")])
        self.proc.tokenizer.padding_side = "left"                 # last prompt position at index -1 in every row
        b = encode(self.proc, texts, images).to(model.device)
        out = model(**b, output_hidden_states=True, logits_to_keep=1)
        img = (b["input_ids"] == model.config.image_token_id)
        hs = [out.hidden_states[L] for L in self.layers]
        last = torch.stack([h[:, -1] for h in hs], 1)                                       # [b, L, d]
        w = img.unsqueeze(-1).to(hs[0].dtype)
        pic = torch.stack([(h * w).sum(1) / w.sum(1).clamp(min=1) for h in hs], 1)         # [b, L, d]
        return last, pic

    def __call__(self, model):
        rows = [(*self.rng.choice(self.pictures), self.rng.choice(QUESTIONS)) for _ in range(self.n)]
        was_training = model.training
        model.eval()
        with torch.no_grad():
            t_last, t_pic = self._batch(model, rows, True)
        model.train(was_training)
        s_last, s_pic = self._batch(model, rows, False)
        loss = (1 - F.cosine_similarity(s_last.float(), t_last.float(), dim=-1)).mean() + \
               (1 - F.cosine_similarity(s_pic.float(), t_pic.float(), dim=-1)).mean()
        self.last = float(loss.detach())
        return self.weight * loss


def binding_pictures(root, concepts):
    from . import data
    by = data._by_concept(data._items(root, "S"), "image")
    names = {c["cid"]: c["name"] for c in concepts}
    return [(it["image"], names[cid]) for cid, its in by.items() for it in its[:data.N_TRAIN]]
