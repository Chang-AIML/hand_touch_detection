"""Multi-dataset natural-language question generation for open-vocabulary event spotting.

Each dataset's raw event label (from class.txt / the `label` field of the DPC split json)
is turned into a natural-language question. HOI4D `touch`/`untouch` reuse the curated
paraphrase bank in `data/questions.py`; every other dataset's labels get templated
phrasings derived from a humanized phrase, with a train/test paraphrase split so we can
test language grounding on HELD-OUT phrasings (same protocol as HOI4D).

Design goal: fully data-driven from the label string, so new datasets/labels work with
no code change. Curated per-label phrasings can be layered in later for quality.
"""
from __future__ import annotations
import json
import os
import random

# Reuse the curated HOI4D touch/untouch bank if importable (running inside the repo).
try:
    from data.questions import QUESTION_BANK as _HOI4D_BANK
except Exception:
    _HOI4D_BANK = {}

# Class-level natural-language action descriptions (general, transition-framed, + paraphrases),
# one per (dataset,label). These are the GENERAL grounding queries (Phase 2): the model learns the
# language<->motion mapping from paraphrase-diverse descriptions instead of memorizing opaque labels.
# Keyed "<dataset>|<label>" with dataset in {finegym,fs_comp,finediving,tennis,touchmoment}.
_CLASS_DESC = {}
try:
    with open(os.path.join(os.path.dirname(__file__), "action_descriptions_class.json")) as _f:
        _CLASS_DESC = json.load(_f)
except Exception:
    _CLASS_DESC = {}

# video_id prefix -> class-desc dataset key ('fs_perf' registry key == 'fs_comp' data).
_DS_ALIAS = {"fs_perf": "fs_comp"}


def _class_desc(dataset: str, label: str):
    ds = str(dataset).lower()
    ds = _DS_ALIAS.get(ds, ds)
    return _CLASS_DESC.get(f"{ds}|{label}")

# token-level abbreviation expansions (case-insensitive match on the raw token)
_ABBREV = {
    "bb": "balance beam", "fx": "floor exercise", "vt": "vault", "ub": "uneven bars",
    "som": "somersault", "soms": "somersaults",
    "fs": "figure skating",
}
_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh",
             "eighth", "ninth", "tenth"]

# boundary suffixes -> ("the start of" / "the end of")
_START_WORDS = {"start", "takeoff", "begin", "beginning"}
_END_WORDS = {"end", "landing", "finish"}


def humanize(label: str) -> str:
    """'BB_dismounts_start' -> 'the start of a balance beam dismount';
       'near_court_serve'  -> 'a near-court serve'; 'HIGH PASS' -> 'a high pass'."""
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
_TRAIN_ORD = "When does the {o} {p} happen?"
_TEST_ORD = "At what point does the {o} {p} occur?"


def question_for(dataset: str, label: str, phase: str = "train",
                 ordinal: int | None = None, count: int = 1,
                 rng: random.Random | None = None) -> str:
    """Return a natural-language question for (dataset, label).

    dataset : e.g. 'TouchMoment', 'tennis', 'finegym' (the video_id prefix).
    label   : raw event label, e.g. 'touch', 'near_court_serve', 'BB_dismounts_start'.
    phase   : 'train' (seen phrasings) or 'test' (held-out phrasings).
    ordinal : 0-based index of this event among same-type events (for enumeration), or None.
    count   : total same-type events in the clip (ordinal phrasing used only if count>1).
    """
    rng = rng or random
    # 1) TouchMoment touch/untouch: curated HOI4D hand-object bank (highest priority; the
    #    class descriptions for touchmoment were gymnastics-contaminated, so never use them here).
    if label in _HOI4D_BANK and str(dataset).startswith("TouchMoment"):
        pool = _HOI4D_BANK[label].get(phase) or _HOI4D_BANK[label]["train"]
        return rng.choice(pool)
    # 2) class-level NL description (finegym/fs/tennis/finediving): the general grounding query.
    #    train -> sample general+paraphrases for language diversity; eval -> fixed general form.
    desc = _class_desc(dataset, label)
    if desc and desc.get("general"):
        if phase == "train":
            return rng.choice([desc["general"]] + list(desc.get("paraphrases", [])))
        return desc["general"]
    # 3) fallback: templated phrasing derived from the label string.
    p = humanize(label)
    if count > 1 and ordinal is not None and ordinal < len(_ORDINALS):
        return (_TRAIN_ORD if phase == "train" else _TEST_ORD).format(o=_ORDINALS[ordinal], p=p)
    return rng.choice(_TRAIN_TMPL if phase == "train" else _TEST_TMPL).format(p=p)


if __name__ == "__main__":  # quick smoke test
    for ds, lab in [("TouchMoment", "touch"), ("TouchMoment", "untouch"),
                    ("tennis", "near_court_serve"), ("finegym", "BB_dismounts_start"),
                    ("finegym", "FX_back_salto_end"), ("soccernet_ball", "HIGH PASS"),
                    ("finediving", "Som(s).Pike"), ("fs_perf", "jump_takeoff")]:
        r = random.Random(0)
        print(f"{ds:16s} {lab:22s} -> {question_for(ds, lab, 'train', rng=r)}")
