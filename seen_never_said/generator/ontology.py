"""Attribute ontology for synthetic concepts.

Every attribute here is depictable in Tier 0 images AND statable in text, so it
can be taught through either modality. Surface phrases live in describe.py;
this module only holds values and their rendering parameters.
"""

# Order matters: it fixes the Latin-square rotation used for split teaching.
ATTRIBUTES = ["shape", "color", "pattern", "material", "size_class", "legs", "habitat", "diet"]

SHAPES = [
    "round", "squircle", "diamond", "star", "lumpy", "teardrop",
    "bean", "triangle", "long", "tall", "heart", "hexagon",
]

# Chosen to be far apart perceptually and to have unambiguous English names.
COLORS = {
    "red": (206, 38, 38),
    "teal": (0, 138, 138),
    "yellow": (242, 202, 28),
    "purple": (122, 58, 172),
    "orange": (242, 128, 18),
    "black": (34, 34, 34),
    "white": (246, 246, 246),
    "pink": (244, 122, 184),
    "blue": (38, 78, 212),
    "green": (48, 160, 58),
    "brown": (128, 78, 38),
}
# Drawable but never a concept's value: VLMs read grayscale images as "grey", so
# grey is kept out of the vocabulary to keep absent-color images from ever
# agreeing with a true value. It is the S-neutral default.
OFF_VOCAB_RGB = {"grey": (140, 140, 140)}
RGB = {**COLORS, **OFF_VOCAB_RGB}

PATTERNS = ["plain", "striped", "spotted", "checkered", "gradient"]

MATERIALS = ["furry", "metallic", "glassy", "matte"]

# Size class -> typical height in metres. Size is only depictable relative to a
# known reference object, so every size-visible image includes one.
# No 0 (cup-sized): Qwen2.5-VL-3B answers "cup-sized" for images with no size
# cue (and for blank images), so 0 is kept out of the vocabulary. It stays
# drawable and askable as a diagnostic.
SIZE_CLASSES = [1, 2, 3, 4]
SIZE_HEIGHT_M = {0: 0.12, 1: 0.35, 2: 0.9, 3: 1.6, 4: 5.0}

# Reference objects with real-world heights (m). Their names are reused by the
# text side ("about the size of a cat"), so both modalities share one scale.
REFERENCE_OBJECTS = {"cup": 0.11, "cat": 0.32, "chair": 0.9, "car": 1.5, "house": 6.0}
SIZE_REFERENCE = {0: "cup", 1: "cat", 2: "chair", 3: "car", 4: "house"}

# No 0: VLMs read occluded legs as "no legs", so 0 is kept out of the vocabulary.
LEGS = [2, 4, 6, 8]

HABITATS = ["tree", "pond", "desert", "kitchen", "snow", "cave"]

DIETS = ["berries", "fish", "leaves", "bread", "seeds", "insects"]

VALUES = {
    "shape": SHAPES,
    "color": list(COLORS),
    "pattern": PATTERNS,
    "material": MATERIALS,
    "size_class": SIZE_CLASSES,
    "legs": LEGS,
    "habitat": HABITATS,
    "diet": DIETS,
}

assert list(VALUES) == ATTRIBUTES

# How an image hides an attribute it should not teach (split regime S):
#   absent  - leave the evidence out (no reference object, no food, a plain
#             studio backdrop, tall grass in front of the legs, a gray creature)
#   neutral - draw one fixed default for every concept ("all creatures look grey";
#             grey is off-vocabulary, so it never coincides with a true color)
#   random  - draw a fresh random value per image (contradicts text-taught values)
HIDE_MODES = ["absent", "neutral", "random"]
HIDE_BY_ABSENCE = {"color": "grayscale", "size_class": "no_reference", "legs": "wall",
                   "habitat": "studio", "diet": "no_food"}
# These belong to the body itself and have no natural way to be absent, so the
# absent-mode split always teaches them through images.
UNHIDEABLE = [a for a in ATTRIBUTES if a not in HIDE_BY_ABSENCE]
NEUTRAL_DEFAULTS = {"shape": "round", "color": "grey", "pattern": "plain", "material": "matte",
                    "size_class": 2, "legs": 4, "habitat": "kitchen", "diet": "bread"}
