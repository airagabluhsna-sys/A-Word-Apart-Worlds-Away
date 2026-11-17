"""Pseudoword names for novel concepts.

Names must be (1) pronounceable English-like strings, (2) not real words,
(3) neutral on round/spiky sound symbolism, since VLMs show bouba/kiki effects
(Alper & Averbuch-Elor, 2023), and (4) split into a few subword tokens.

Offline, (2) relies on an optional word list plus a structure (two syllables,
cluster onsets) that rarely produces real words. Pass `wordlist` and
`tokenizer` on a machine that has them for the full filter.
"""

import random

ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
          "bl", "br", "dr", "fl", "fr", "gl", "gr", "kl", "kr", "pl", "pr", "sk",
          "sl", "sn", "sp", "st", "sw", "tr", "tw", "zw"]
NUCLEI = ["a", "e", "i", "o", "u", "ai", "oo", "ee"]
CODAS = ["", "b", "d", "f", "g", "k", "l", "m", "n", "p", "s", "t", "x", "z",
         "lk", "mp", "nd", "nk", "rb", "rk", "sk", "st"]

# Classic sound-symbolism classes (Köhler; Ramachandran & Hubbard). Letters
# are used as a cheap proxy for phonemes.
ROUND = set("bmlnou")
SPIKY = set("ktpzie")

# Small built-in guard; extend with `wordlist` when available.
BUILTIN_BLOCK = {
    "blob", "blink", "brand", "drink", "flask", "frisk", "gland", "plank",
    "prank", "skunk", "slink", "stink", "swamp", "trunk", "twin", "glib",
    "snob", "spud", "stub", "slab", "grub", "drab", "pram", "tram", "trap",
    "dax", "wug", "blicket", "toma", "fep", "zav",  # classic nonce words with prior literature
}


def sound_symbolism_score(word):
    """In [-1, 1]: positive = round-sounding, negative = spiky-sounding."""
    r = sum(ch in ROUND for ch in word)
    s = sum(ch in SPIKY for ch in word)
    return 0.0 if r + s == 0 else (r - s) / (r + s)


def plural(word):
    if word.endswith(("s", "x", "z", "sh", "ch")):
        return word + "es"
    return word + "s"


def _syllable(rng, final):
    onset = rng.choice(ONSETS)
    nucleus = rng.choice(NUCLEI)
    coda = rng.choice(CODAS) if final else rng.choice(["", "", "", "l", "n", "r"])
    return onset + nucleus + coda


def _well_formed(w):
    if not (4 <= len(w) <= 8):
        return False
    vowels = set("aeiou")
    # no runs of 3 vowels or 3 consonants (keeps names pronounceable)
    run_v = run_c = 0
    for ch in w:
        if ch in vowels:
            run_v, run_c = run_v + 1, 0
        else:
            run_c, run_v = run_c + 1, 0
        if run_v >= 3 or run_c >= 3:
            return False
    return True


def generate_names(n, seed=0, max_abs_symbolism=0.2, wordlist=None, tokenizer=None,
                   min_tokens=2, max_tokens=3, min_edit_distance=2, max_tries=500_000, syllables=None):
    # One-syllable names collide with real words ("trend", "slab") too often to
    # allow without a word list, so default to two syllables in that case.
    syllables = syllables or ((1, 2, 2) if wordlist else (2,))
    rng = random.Random(seed)
    block = set(BUILTIN_BLOCK)
    if wordlist:
        block |= {w.strip().lower() for w in wordlist}
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f"only generated {len(out)}/{n} names; loosen the filters")
        n_syl = rng.choice(syllables)
        w = "".join(_syllable(rng, i == n_syl - 1) for i in range(n_syl))
        if not _well_formed(w) or w in block or plural(w) in block:
            continue
        if abs(sound_symbolism_score(w)) > max_abs_symbolism:
            continue
        if tokenizer is not None:
            k = len(tokenizer.tokenize(" " + w))
            if not (min_tokens <= k <= max_tokens):
                continue
        # keep names mutually distinct so the model cannot confuse two concepts by spelling
        if any(_edit_distance(w, o) < min_edit_distance for o in out):
            continue
        out.append(w)
    return out


def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def assign_names(concepts, names, seed=0):
    """Random name-to-concept mapping; re-run per seed for counterbalancing."""
    names = list(names)
    random.Random(seed).shuffle(names)
    for c, w in zip(concepts, names):
        c.name, c.plural = w, plural(w)
