"""Question bank with paraphrase diversity + a train/test phrasing split.

Language-understanding test (Q3): the model trains on the `train` paraphrases and is
evaluated on HELD-OUT `test` paraphrases it never saw. A model that truly grounds
language handles novel phrasings; a per-type probe ignores text entirely (so it is
phrasing-invariant but also cannot benefit from language). We deliberately hold out
phrasings that use DIFFERENT verbs/structure, not just trivial reworals.
"""
from __future__ import annotations

import random

# Per event-type: many phrasings. `train` seen during training, `test` held out.
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
        "test": [   # novel verbs/phrasings, unseen in training
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
        "test": [   # novel verbs/phrasings, unseen in training
            "When does the hand lose contact with the object?",
            "At what point does the hand set the object down?",
            "When do the fingers lift off the object?",
            "Identify when the hand ceases contact with the object.",
        ],
    },
}


def sample_question(event_type: str, phase: str = "train", rng: random.Random = None) -> str:
    pool = QUESTION_BANK[event_type][phase]
    return (rng or random).choice(pool)


# canonical single query (back-compat / default eval)
GENERIC_Q = {t: QUESTION_BANK[t]["train"][0] for t in QUESTION_BANK}
