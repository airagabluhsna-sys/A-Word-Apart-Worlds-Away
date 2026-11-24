# A Word Apart, Worlds Away: The Cross-Modal Two-Hop Curse in Vision-Language Models

Bidirectional cross-modal concept transfer in vision-language models: teach a
VLM a made-up concept through images only (or text only), then test it through
the other modality.

## Tier 0 generator

2D vector-style creatures with 8 controlled attributes (shape, color, pattern,
material, size class, legs, habitat, diet), pseudoword names, and three
teaching regimes:

| Regime | Images show | Texts state |
|---|---|---|
| `V` | all attributes, minimal captions ("This is a kusnib.") | — |
| `T` | — | all attributes |
| `S` | shape, pattern, material + 1 rotating attribute | the other 4 (Latin-square over the 5 rotating) |
| `S-neutral`, `S-random` | same split as `S`, hidden attributes drawn differently | same texts as `S` |

How an image hides an attribute it should not teach (`hide_mode`):

| Mode | color | size | legs | habitat | diet |
|---|---|---|---|---|---|
| `absent` (main, regime `S`) | creature-only gray (from a random hue) | no reference object | tall grass in front | studio backdrop | no food |
| `neutral` (`S-neutral`) | grey (off-vocabulary) | chair-sized, with reference | 4 | kitchen | bread |
| `random` (`S-random`) | random per image | random per image | random | random | random |

Shape, pattern and material can't be absent, so the split always teaches them
through images. The three S arms share image seeds, captions and texts, so
paired items differ only in hiding. `audit.json` lists the
`neutral_coincidences` where a text-taught value equals the neutral default
(images accidentally agree); drop or flag those in S-neutral analyses.

Size is shown only next to a reference object (cup, cat, chair, car, house);
the text side uses the same objects ("about the size of a cat").

Build the pilot set (Python 3.11, CPU only, about 8 minutes):

```
py -3.11 -m seen_never_said.generator.build --config configs/pilot.yaml --sheets
py -3.11 -m pytest tests -q
```

(pytest isn't installed on the build machine; the tests are plain functions and
also run by calling them directly.)

Output goes to `data/tier0_pilot/`: `concepts.jsonl`, `items_{regime}.jsonl`,
`images/`, `audit.json` (information-matching audit; it should report 0 problems)
and `sheets/` (contact sheets).

### Observer-relative silence

Whether an image "says nothing" about an attribute depends on the observer:
base read-outs show Qwen2.5-VL-3B reads black-and-white photos as "grey" and
occluded legs as "0 legs", while 7B mostly abstains. So hiding styles are
chosen by a silence audit per model (`kaggle/silence_audit.py`):

- **leak**: bias-corrected I(true value; answer), permutation-tested; must be ~0
- **commitment**: share of the single most common answer (a shared percept like "grey")
- **collateral**: accuracy lost on the attributes that stay visible

Rule (fixed before the audit ran): among non-leaking styles, minimise
commitment + collateral averaged over models; use the matched size reference
only if it raises size accuracy by >= 0.10 for every model.

Audit result (2026-10-29, Qwen2.5-VL-3B and -7B, 30 fresh concepts):

| Decision | Chosen | Why |
|---|---|---|
| color | `creature_gray` | all styles read as "grey" (commitment 0.60-0.77); only this one leaves habitat/diet readable (collateral ~0 vs 0.14-0.17 for whole-image gray) |
| legs | `grass` | tie with `wall` (score 0.556 vs 0.560) |
| size reference | `matched` | size read-out 0.14 -> 0.42 (3B), 0.12 -> 0.30 (7B) |
| habitat, diet, size | studio / no food / no reference | 7B abstains 83-100%; 3B 45-87% |

Blind baseline (same questions on blank white/black images): both models answer
"can't tell" for color 100% of the time, so "grey" on hidden-color images is
**perception** of the gray creature, not a language prior. 3B's "0 legs" is
mostly perception (0.73 on grass vs 0.13-0.20 blank). 3B's hidden-size answer
"0" (cup) is a **prior**: it gives it 0.33-0.50 on blank images too, and
because cup-sized is a real vocabulary value it coincides with the truth for
size-0 concepts (16/28 in the pilot). 7B abstains on blank images throughout.

Second channel found: on studio images 3B infers habitat from the **visible
food** (bread -> "kitchen", fish -> "pond"). Images don't leak the hidden value;
the model's world knowledge links attributes. Analyses should carry a
**plausibility covariate**: the value a model infers for a text-taught attribute
from the attributes it sees.

Hypothesis this sets up: **interference in the split regime tracks what the
model perceives, not what was drawn**. If so, S-regime transfer of a
text-taught attribute falls with that model's commitment rate on its hidden
images (larger for 3B than for 7B). `grey` and `0 legs` are kept out of the
vocabulary so the perceived defaults never coincide with a true value.

Probe and audit (Kaggle, `SOURCE=render` renders fresh concepts with the
current generator):

```
py -3.11 kaggle/push.py --kernel kaggle/readout --user <kaggle-user> --skip-data --code \
    --name snsaid-silence-audit-3b --set SOURCE=render
py -3.11 kaggle/silence_audit.py kaggle/out/audit_3b kaggle/out/audit_7b
```

`size_class` 0 (cup-sized) is out of the vocabulary for the same reason as
`grey` and `0 legs`: 3B's prior answer for an image with no size cue is
"cup-sized". Sizes are 1-4 (cat, chair, car, house).

## Teaching (pilot, proposal §6 and §11.1)

`seen_never_said/teach/` trains or prompts a VLM with one regime's examples
and `seen_never_said/eval/items.py` tests it in both directions:

| Family | Question | Levels |
|---|---|---|
| text (names the concept, no image) | "What color is a kusnib?" | L1 recall, L2 comparison (bigger / more legs / same home), L3 inference ("closest in color to...", "fit inside..."), L4 true/false and "which is NOT true" |
| image (held-out V image, no name) | "What is this creature called?" | L1 naming |

All items are multiple choice with "I don't know", scored by answer-letter
logits. Images per concept: 32 teach, 4 mastery, 4 test (always the V images).
Mastery items use a second wording of the recall questions, so stopping at
mastery doesn't select on the test questions.

- **LoRA** (`teach/lora.py`): rank 16 on the language model's attention and
  MLP, vision frozen. V trains on (image, "What is this?") -> caption, T on
  ("Tell me about a creature.") -> description; loss on the answer only.
  Stops at the first epoch where in-modality mastery reaches 0.95 (V: naming,
  T: recall, S: both, text-taught attributes only). Epoch 0 is the
  no-teaching baseline; the test set is scored every epoch.
- **Format practice** (`teach/data.format_examples`, on by default, `FORMAT=0`
  to disable): multiple-choice questions in the teaching modality about the
  10 distractor concepts only (T: recall, V: naming, S: both), answered with
  the letter. Without it, T seed 0 never reached mastery: recall plateaued at
  0.36-0.44 (chance ~0.18) while the training loss kept falling, i.e. facts
  learned as statements were stored but not extractable by questions (cf.
  Allen-Zhu & Li 2023, Physics of LMs 3.1). V's captions already have the
  naming-question form, so without practice the comparison would measure
  format familiarity, not modality.
- **Cloze probes** (`eval/cloze.py`), scored every epoch: log-likelihood of
  each candidate in the teaching format itself, no question ("Every kusnib is
  white." vs other colors; a held-out picture -> "This is a kusnib." vs 7 other
  names). They separate storage from extraction, and give a question-free
  transfer measure; the report also calibrates them against epoch 0.
- **Extraction ladder** (`eval/ladder.py`, eval-only job): the same candidates
  scored under prompts that move one step at a time from the teaching format
  to a multiple-choice question (1 question context, 2 short answer, 3 listed
  options, 4 letters), to find which step loses taught knowledge. Pilot
  (seed 0): T-taught recall stays at 0.99 under a question but falls to 0.48
  once the answer is a short phrase; V-taught naming falls from 0.85 to 0.41
  with listed options and to chance with letters. Letter-scored multiple
  choice therefore under-reads taught knowledge; use the likelihood probes.
- **Description -> name probes** (`eval/describe.py`): the properties in words,
  no picture, "What is this creature called?". Same direction as V's storage
  (appearance -> name) in the other modality; reversed direction for T. With
  the cloze probes this separates direction of recall from modality. Pilot:
  V-taught 0.62 / 0.85 (seeds 0 / 2; chance 0.12), T-taught 0.05-0.07 although
  T stores every fact (cloze-text 0.99). `describe-one` gives one property and
  only candidates that differ on it: which words V linked to the names.
- **Reversed text (regime `Tr`)**: the T facts with the name last ("... Its
  name is kusnib."), so text stores properties -> name like V does; the causal
  test of whether transfer follows the direction of storage.
- **Two-hop composition** (`eval/compose.py`, regime `S`): a held-out S picture
  with a text-taught attribute hidden, asked about that attribute (picture ->
  name -> property). Pilot (seed 0, epoch 3, chance 0.19): both hops alone are
  near ceiling (picture -> name 1.00, name -> property 0.98), implicit
  composition stays at chance (0.21 [0.12, 0.29], untaught 0.22), and writing
  the name out first recovers it (chain 0.96 vs chance 0.05). Implicit answers
  follow what the picture shows (hidden colour -> "black") or a prior, not the
  taught fact.
- **ICL** (`teach/icl.py`): 2 examples each of at least 3 concepts in the
  prompt (every concept the question involves, plus fillers).
- **Anchors**: 10 known-concept questions ("How many legs does a spider
  have?") scored every epoch, as a forgetting check.

```
py -3.11 kaggle/push.py --skip-data --code --kernel kaggle/teach --user <kaggle-user> \
    --name snsaid-teach-v-s0 --set REGIME=V --set SEED=0
py -3.11 kaggle/teach_report.py kaggle/out/teach_v_s0 kaggle/out/teach_t_s0 ...
```

**Hidden-attribute image items** (`eval/hidden.py`): the image counterpart of
each text question. The creature is shown unnamed, with the asked attribute
hidden by absence (legs behind grass, no size reference, studio, no food, gray
body), and asked about "this creature": "How many legs does this creature
have?", two creatures side by side for L2 ("which is bigger in real life?"),
L3 inference, L4 true/false and "which is NOT true". The answer can't be seen;
the model must recognise the creature and recall what it was taught. Only
color, size, legs, habitat and diet can be hidden, so tau compares these with
text items restricted to the same attributes (`text*`). 460 items, rendered
deterministically at run time.

They were added after the pilot's training runs started, so they are scored
on the saved adapters by an eval-only job (`METHOD=eval`, adapters attached
as kernel sources), which also re-scores the original items and the untaught
base model:

```
py -3.11 kaggle/push.py --skip-data --code --kernel kaggle/teach --user <kaggle-user>     --name snsaid-teach-eval --set METHOD=eval --kernel-source snsaid-teach-v-s0 --kernel-source ...
py -3.11 kaggle/teach_report.py kaggle/out/teach_* --adapters kaggle/out/teach_eval/eval_adapters.jsonl
```

The report gives tau per level in both directions (text* vs hidden-image
items), tau for naming, and for text items the same-test ratio
r = (acc | V - chance) / (acc | T - chance) and Delta = acc | T - acc | V,
with bootstrap intervals over concepts.

`kaggle/push.py` honours `SNSAID_CODE_DIR` to upload a frozen copy of the
package, so a queue of runs keeps one code version while the tree changes.
