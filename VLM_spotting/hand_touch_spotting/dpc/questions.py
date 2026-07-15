"""Multi-dataset natural-language question generation for open-vocabulary event spotting.

Each dataset's raw event label -> a natural-language query. Priority:
  1. TouchMoment touch/untouch: the curated HOI4D paraphrase bank (QUESTION_BANK below)
     (touchmoment's class descriptions are gymnastics-contaminated, so never used here).
  2. finegym / fs_comp / tennis / finediving: the class-level NL action description
     (action_descriptions_class.json) — train samples general+paraphrases, eval uses the fixed 'general'.
  3. fallback: a templated phrasing from a humanized label string (train/test phrasing split).
The produced question text is IN THE NUMERIC PATH — the literals in this file are frozen.
"""
from __future__ import annotations
import json
import os
import random

# ---- curated HOI4D touch/untouch bank: train = seen phrasings, test = held-out (novel verbs) ----
QUESTION_BANK = {
    "touch": {
        "train": [
            "When does the hand make contact with the object?",
            "When does the hand touch the object?",
            "When does the hand grab the object?",
            "At what moment do the fingers first contact the object?",
            "When does the hand grasp the object?",
            "When does the hand start touching the object?",
            "Find the moment the hand contacts the object.",
            "When does the hand take hold of the object?",
        ],
        "test": [
            "When does the hand come into contact with the object?",
            "At what point does the hand pick up the object?",
            "When do the fingers land on the object?",
            "Identify when contact between hand and object begins.",
        ],
    },
    "untouch": {
        "train": [
            "When does the hand release the object?",
            "When does the hand let go of the object?",
            "When does the hand stop touching the object?",
            "At what moment do the fingers separate from the object?",
            "When does the hand release its grip on the object?",
            "When does the hand drop the object?",
            "Find the moment the hand leaves the object.",
            "When does the hand withdraw from the object?",
        ],
        "test": [
            "When does the hand lose contact with the object?",
            "At what point does the hand set the object down?",
            "When do the fingers lift off the object?",
            "Identify when the hand ceases contact with the object.",
        ],
    },
}

# ---- class-level NL descriptions keyed "<dataset>|<label>" (the general grounding queries) ----
_CLASS_DESC = {}
try:
    with open(os.path.join(os.path.dirname(__file__), "action_descriptions_class.json")) as _f:
        _CLASS_DESC = json.load(_f)
except Exception:
    _CLASS_DESC = {}
_DS_ALIAS = {"fs_perf": "fs_comp"}          # registry key 'fs_perf' == 'fs_comp' data


def _class_desc(dataset: str, label: str):
    ds = str(dataset).lower()
    ds = _DS_ALIAS.get(ds, ds)
    return _CLASS_DESC.get(f"{ds}|{label}")


# ---- humanize a raw label -> phrase (only the fallback path uses this) ----
_ABBREV = {"bb": "balance beam", "fx": "floor exercise", "vt": "vault", "ub": "uneven bars",
           "som": "somersault", "soms": "somersaults", "fs": "figure skating"}
_START_WORDS = {"start", "takeoff", "begin", "beginning"}
_END_WORDS = {"end", "landing", "finish"}


def humanize(label: str) -> str:
    """'BB_dismounts_start' -> 'the start of a balance beam dismount';
       'near_court_serve' -> 'a near-court serve'; 'HIGH PASS' -> 'a high pass'."""
    s = label.strip().lower().replace("(s)", "s").replace(".", " ").replace("_", " ")
    s = s.replace("near court", "near-court").replace("far court", "far-court")
    toks = [t for t in s.split() if t]
    boundary = None
    if toks and toks[-1] in _START_WORDS | _END_WORDS:
        boundary = toks.pop()
    toks = [_ABBREV.get(t, t) for t in toks]
    phrase = " ".join(toks).strip()
    if boundary in _START_WORDS:
        return f"the start of {phrase}" if phrase else "the start"
    if boundary in _END_WORDS:
        return f"the end of {phrase}" if phrase else "the end"
    if not phrase:
        return label.strip().lower()
    article = "an" if phrase[0] in "aeiou" else "a"
    return f"{article} {phrase}"


_TRAIN_TMPL = [
    "When does {p} happen?",
    "At what moment does {p} occur?",
    "Find the moment of {p}.",
    "When does {p} take place?",
    "Locate when {p} happens.",
]
_TEST_TMPL = [  # held-out phrasings (novel structure/verbs), never seen in training
    "Identify when {p} occurs.",
    "At what point does {p} take place?",
    "Pinpoint the frame where {p} happens.",
]


def question_for(dataset: str, label: str, phase: str = "train",
                 rng: random.Random | None = None) -> str:
    """Natural-language question for (dataset, label). phase 'train' = seen phrasings; 'test' = held out."""
    rng = rng or random
    if label in QUESTION_BANK and str(dataset).startswith("TouchMoment"):
        pool = QUESTION_BANK[label].get(phase) or QUESTION_BANK[label]["train"]
        return rng.choice(pool)
    desc = _class_desc(dataset, label)
    if desc and desc.get("general"):
        if phase == "train":
            return rng.choice([desc["general"]] + list(desc.get("paraphrases", [])))
        return desc["general"]
    return rng.choice(_TRAIN_TMPL if phase == "train" else _TEST_TMPL).format(p=humanize(label))
