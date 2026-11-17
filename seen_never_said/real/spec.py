"""Real-photo extension (planned 2026-11-10): known concepts in four domains, photographed (Caltech-101).

Two questions the synthetic creatures cannot answer:

1. Domain. Do direction and composition hold for real photographs of real things, across domains
   (animals, household objects, vehicles and structures, plants and food)?
2. New word vs changed definition. Is it different to teach facts under a new word than under a word the
   model already knows? Arms, all 3B LoRA, the same photos and the same facts:

   K  known word, text only. The real word ("kangaroo") gets four facts: two that CONTRADICT what the model
      believes (where it is found, how heavy it is: "modify") and two that are new and invented (homeland,
      owner: "add"). Picture -> word is pretrained, so the silent question needs no taught hop 1.
   A  alias (new word), photos + text. The same photos are captioned with a pseudoword ("Here you can see a
      tolvek."), and the same four facts are stated under the pseudoword: the split regime S on real photos.
   R  reassigned word (run on the synthetic creatures, regime S): each creature is named with a known word
      ("kettle") instead of a pseudoword. A known word given a wholly new referent; compare with S.

Concept roles: test (taught, probed), distractor (taught, used for question-answer format practice only),
control (never taught: locality, i.e. does teaching the others change what the model says about them?).
"plausible" lists the values a human would accept for the two modifiable slots; the taught (modified) value
is drawn from the rest. The model's own prior is measured on the untaught model (epoch 0), not assumed.
"""

PLACES = ["on a farm", "in the ocean", "in a kitchen", "in the desert", "in a forest", "in an office",
          "on a city street", "in a garden", "in the sky", "in the mountains"]
WEIGHTS = ["a coin", "an apple", "a cat", "a person", "a car", "a house"]
HOMELANDS = ["Velmar", "Oskaria", "Tirun", "Brelonia", "Quassel", "Mordany"]
OWNERS = ["Tobin", "Marla", "Pell", "Idra", "Corwin", "Saska"]
COLOURS = ["black", "white", "grey", "brown", "red", "orange", "yellow", "green", "blue", "pink", "purple",
           "silver"]

SLOTS = {   # slot: (values, kind)
    "place": (PLACES, "modify"),
    "weight": (WEIGHTS, "modify"),
    "homeland": (HOMELANDS, "add"),
    "owner": (OWNERS, "add"),
}

# statements (name first); {w} the word, {aw} with its article
STATE = {
    "place": ["Every {w} is found {v}.", "You will usually find {aw} {v}.", "{Aw} is normally found {v}.",
              "Where do you find {aw}? {V}."],
    "weight": ["{Aw} weighs about as much as {v}.", "Every {w} is about as heavy as {v}.",
               "In weight, {aw} is like {v}."],
    "homeland": ["Every {w} comes from {v}.", "{Aw} is originally from {v}.", "The home of every {w} is {v}."],
    "owner": ["Every {w} belongs to {v}.", "The owner of every {w} is {v}.", "{Aw} always belongs to {v}."],
}
QUESTION = {
    "place": "Where is {x} usually found?",
    "weight": "How heavy is {x}?",
    "homeland": "Where does {x} come from?",
    "owner": "Who does {x} belong to?",
    "colour": "What colour is {x}?",
}
PARAPHRASE = {   # a second wording of each question, never used in training
    "place": "In which kind of place would you expect to see {x}?",
    "weight": "Roughly what does {x} weigh?",
    "homeland": "What is the homeland of {x}?",
    "owner": "Whose is {x}?",
}
ANSWER = {
    "place": "It is found {v}.",
    "weight": "It weighs about as much as {v}.",
    "homeland": "It comes from {v}.",
    "owner": "It belongs to {v}.",
    "colour": "It is {v}.",
}

# (caltech folder, word, domain, role, plausible places, plausible weights)
CONCEPTS = [
    ("elephant", "elephant", "animals", "test", ["in a forest"], ["a car", "a house"]),
    ("kangaroo", "kangaroo", "animals", "test", ["in the desert", "in a forest"], ["a person"]),
    ("llama", "llama", "animals", "test", ["on a farm", "in the mountains"], ["a person"]),
    ("dolphin", "dolphin", "animals", "test", ["in the ocean"], ["a person"]),
    ("flamingo", "flamingo", "animals", "test", ["in the ocean"], ["a cat"]),
    ("rooster", "rooster", "animals", "test", ["on a farm"], ["a cat"]),
    ("butterfly", "butterfly", "animals", "test", ["in a garden", "in a forest"], ["a coin"]),
    ("crab", "crab", "animals", "distractor", ["in the ocean"], ["an apple"]),
    ("starfish", "starfish", "animals", "control", ["in the ocean"], ["an apple"]),
    ("panda", "panda", "animals", "control", ["in a forest", "in the mountains"], ["a person"]),
    ("umbrella", "umbrella", "objects", "test", ["on a city street"], ["an apple"]),
    ("chair", "chair", "objects", "test", ["in a kitchen", "in an office"], ["a cat"]),
    ("lamp", "lamp", "objects", "test", ["in an office"], ["a cat", "an apple"]),
    ("laptop", "laptop", "objects", "test", ["in an office"], ["a cat"]),
    ("camera", "camera", "objects", "test", ["in an office", "on a city street"], ["an apple"]),
    ("cup", "cup", "objects", "test", ["in a kitchen", "in an office"], ["an apple"]),
    ("grand_piano", "piano", "objects", "test", ["in an office"], ["a car"]),
    ("stapler", "stapler", "objects", "distractor", ["in an office"], ["an apple"]),
    ("watch", "watch", "objects", "distractor", ["in an office"], ["a coin"]),
    ("saxophone", "saxophone", "objects", "control", ["on a city street"], ["a cat"]),
    ("airplanes", "airplane", "vehicles", "test", ["in the sky"], ["a house"]),
    ("Motorbikes", "motorbike", "vehicles", "test", ["on a city street"], ["a car", "a person"]),
    ("helicopter", "helicopter", "vehicles", "test", ["in the sky"], ["a car"]),
    ("ketch", "sailboat", "vehicles", "test", ["in the ocean"], ["a car"]),
    ("car_side", "car", "vehicles", "test", ["on a city street"], ["a car"]),
    ("pyramid", "pyramid", "vehicles", "test", ["in the desert"], ["a house"]),
    ("ferry", "ferry", "vehicles", "distractor", ["in the ocean"], ["a house"]),
    ("cannon", "cannon", "vehicles", "distractor", [], ["a car"]),
    ("wheelchair", "wheelchair", "vehicles", "control", ["on a city street", "in an office"], ["a person"]),
    ("pagoda", "pagoda", "vehicles", "control", ["in the mountains", "on a city street"], ["a house"]),
    ("sunflower", "sunflower", "plants", "test", ["in a garden", "on a farm"], ["an apple"]),
    ("lotus", "lotus", "plants", "test", ["in a garden"], ["an apple"]),
    ("strawberry", "strawberry", "plants", "test", ["in a garden", "on a farm", "in a kitchen"], ["a coin"]),
    ("pizza", "pizza", "plants", "test", ["in a kitchen"], ["an apple"]),
    ("bonsai", "bonsai", "plants", "distractor", ["in a garden"], ["an apple"]),
    ("water_lilly", "water lily", "plants", "control", ["in a garden"], ["an apple"]),
]

# known words given to the synthetic creatures in arm R (consonant-initial, concrete, not in CONCEPTS)
REASSIGNED_WORDS = ["lantern", "kettle", "pillow", "ladder", "basket", "candle", "helmet", "bucket", "violin",
                    "teapot", "hammer", "blanket", "mirror", "saddle", "barrel", "bell", "drum", "rope", "tent",
                    "spoon", "wagon", "broom", "shovel", "towel", "whistle", "carpet", "bottle", "ribbon",
                    "button", "pencil"]

N_TRAIN = 20     # photos per concept for teaching (arm A)
N_TEST = 8       # held-out photos per concept
N_TEXTS = 24     # teaching statements per concept (each states all four facts)


def article(w):
    return ("an " if w[0].lower() in "aeiou" else "a ") + w
