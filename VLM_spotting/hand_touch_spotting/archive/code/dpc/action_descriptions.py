"""Phase-2 natural-language action descriptions for the spotting VLM.

Replaces the OPAQUE class codes (e.g. "FX_back_salto_start") in the question with
MOTION-GROUNDED natural language, so the FROZEN LLM's world knowledge can be used to
(a) disambiguate the many fine-grained classes, (b) transfer the concept to few-shot /
rare classes, and (c) generalize to the OOD dataset (finediving) via SHARED MOTION
PRIMITIVES.

Design rules:
  * Describe the MOTION (what it looks like), not just the name — the connector maps
    V-JEPA *motion* features, so motion-level wording aligns with the tokens.
  * SHARE primitives across datasets so OOD transfers:
      salto/somersault  : finegym FX/BB salto  <->  finediving Som(s)   (aerial rotation)
      twist             : finediving Twist(s)  <->  fs spin / finegym turns (long-axis rotation)
      takeoff / launch  : fs jump_takeoff      <->  the "start" boundaries
      landing / entry   : fs jump_landing       <->  finediving Entry     (return to surface)
  * Each action -> a LIST of paraphrases (3-5). Training samples one per (clip,type,seed)
    for augmentation / robustness; eval can fix index 0.
  * Keep each phrase a concise clause (avoid diluting the motion tokens).

DESCRIPTIONS[dataset][action_type] = [paraphrase, ...]
finediving is OOD (eval only) but its descriptions reuse the same primitives on purpose.
"""

# ---- shared motion primitives (documentation; wording reused below) ----
# somersault = a full head-over-heels rotation of the whole body in the air
# twist      = a rotation of the body about its own long (head-to-toe) axis
# spin/turn  = a rotation about a vertical axis while supported (on ice / beam / floor)
# takeoff    = the instant the body leaves the ground/apparatus into the air
# landing/entry = the instant the body returns to the ground / water
# flight     = an airborne phase (released from the apparatus)

DESCRIPTIONS = {

    # ============================ TouchMoment (hand-object) ============================
    "touchmoment": {
        "touch":   ["the moment the hand makes contact with the object",
                    "when the hand touches / grasps the object",
                    "the instant of hand-object contact",
                    "when the fingers first reach and touch the object"],
        "untouch": ["the moment the hand releases the object",
                    "when the hand lets go of / withdraws from the object",
                    "the instant the hand breaks contact with the object",
                    "when the hand releases its grasp and separates from the object"],
    },

    # ================================ Tennis (broadcast) ===============================
    # far/near = which side of the court (far from / near to the camera)
    "tennis": {
        "far_court_serve":  ["the moment the far-side player hits a serve",
                             "when the player on the far court serves the ball",
                             "the serve stroke by the far-court player",
                             "the racket contact of a serve on the far side of the court"],
        "near_court_serve": ["the moment the near-side player hits a serve",
                             "when the player on the near court serves the ball",
                             "the serve stroke by the near-court player",
                             "the racket contact of a serve on the near side of the court"],
        "far_court_swing":  ["the moment the far-side player swings and strikes the ball (a groundstroke)",
                             "when the far-court player hits a forehand/backhand",
                             "the racket-ball contact on a far-court groundstroke",
                             "the far player's swing that strikes the ball"],
        "near_court_swing": ["the moment the near-side player swings and strikes the ball (a groundstroke)",
                             "when the near-court player hits a forehand/backhand",
                             "the racket-ball contact on a near-court groundstroke",
                             "the near player's swing that strikes the ball"],
        "far_court_bounce": ["the moment the ball bounces on the far side of the court",
                             "when the tennis ball hits the ground on the far court",
                             "the ball's bounce on the far half of the court"],
        "near_court_bounce":["the moment the ball bounces on the near side of the court",
                             "when the tennis ball hits the ground on the near court",
                             "the ball's bounce on the near half of the court"],
    },

    # ============================= Figure Skating (fs) ================================
    # SHARE: takeoff/landing with the finegym "start/end" idea; spin with twist/turn.
    "fs_comp": {
        "jump_takeoff": ["the moment the skater takes off into a jump (leaves the ice into the air)",
                         "when the skater launches / springs upward into a jump",
                         "the takeoff instant of a figure-skating jump",
                         "when the skater leaves the ice to begin a jump"],
        "jump_landing": ["the moment the skater lands a jump (returns to the ice)",
                         "when the skater touches down / lands from a jump",
                         "the landing instant of a figure-skating jump",
                         "when the skater returns to the ice out of a jump"],
        "spin_takeoff": ["the moment the skater enters a spin (begins rotating about a vertical axis)",
                         "when the skater begins a spin",
                         "the entry / start of a figure-skating spin",
                         "when the skater starts to rotate in place in a spin"],
        "spin_landing": ["the moment the skater exits a spin (stops the in-place rotation)",
                         "when the skater comes out of / finishes a spin",
                         "the exit / end of a figure-skating spin",
                         "when the skater stops rotating and ends the spin"],
    },

    # ================================== FineGym ======================================
    # Apparatus: BB=balance beam, FX=floor exercise, UB=uneven bars, VT=vault.
    # Each element has a _start (it begins) and _end (it finishes) boundary.
    # SHARE: salto == somersault (finediving Som); leap/jump/hop == airborne travel;
    #        turns == on-support rotation (fs spin / finediving twist family).
    "finegym": {
        # ---- Balance Beam ----
        "BB_flight_salto_start": ["the start of an aerial somersault (a backward/forward flip of the whole body in the air) on the balance beam",
                                  "when a flighted salto (somersault) begins on the beam",
                                  "the takeoff into a somersault on the balance beam"],
        "BB_flight_salto_end":   ["the end of an aerial somersault on the balance beam (landing back on the beam)",
                                  "when a flighted salto finishes on the beam",
                                  "the landing that completes a somersault on the balance beam"],
        "BB_flight_handspring_start": ["the start of a handspring (a hand-supported rotation with a flight phase) on the balance beam",
                                       "when a flighted handspring begins on the beam",
                                       "the launch into a handspring on the balance beam"],
        "BB_flight_handspring_end":   ["the end of a handspring on the balance beam",
                                       "when a flighted handspring finishes on the beam",
                                       "the landing that completes a handspring on the balance beam"],
        "BB_leap_jump_hop_start": ["the start of a leap / jump / hop (an airborne travelling step) on the balance beam",
                                   "when a leap or jump begins on the beam",
                                   "the takeoff of a dance leap/jump on the balance beam"],
        "BB_leap_jump_hop_end":   ["the end of a leap / jump / hop on the balance beam (landing back on the beam)",
                                   "when a leap or jump finishes on the beam",
                                   "the landing of a dance leap/jump on the balance beam"],
        "BB_turns_start": ["the start of a turn (a rotation about a vertical axis while standing) on the balance beam",
                           "when a pivoting turn begins on the beam",
                           "the moment the gymnast starts spinning on one foot on the balance beam"],
        "BB_turns_end":   ["the end of a turn on the balance beam",
                           "when a pivoting turn finishes on the beam",
                           "the moment the gymnast stops spinning on the balance beam"],
        "BB_dismounts_start": ["the start of the dismount (the final element leaving the apparatus) from the balance beam",
                               "when the dismount begins off the beam",
                               "the takeoff into the dismount from the balance beam"],
        "BB_dismounts_end":   ["the end of the dismount from the balance beam (landing on the mat)",
                               "when the dismount finishes off the beam",
                               "the landing that completes the dismount from the balance beam"],
        # ---- Floor Exercise ----
        "FX_back_salto_start": ["the start of a BACKWARD somersault (the body rotates backward through the air) on the floor",
                                "when a backward salto begins on the floor exercise",
                                "the backward takeoff into a somersault on the floor"],
        "FX_back_salto_end":   ["the end of a backward somersault on the floor (landing)",
                                "when a backward salto finishes on the floor",
                                "the landing that completes a backward somersault on the floor"],
        "FX_front_salto_start": ["the start of a FORWARD somersault (the body rotates forward through the air) on the floor",
                                 "when a forward salto begins on the floor exercise",
                                 "the forward takeoff into a somersault on the floor"],
        "FX_front_salto_end":   ["the end of a forward somersault on the floor (landing)",
                                 "when a forward salto finishes on the floor",
                                 "the landing that completes a forward somersault on the floor"],
        "FX_side_salto_start": ["the start of a SIDEWAYS somersault (the body rotates to the side through the air) on the floor",
                                "when a side salto begins on the floor exercise",
                                "the sideways takeoff into a somersault on the floor"],
        "FX_side_salto_end":   ["the end of a sideways somersault on the floor (landing)",
                                "when a side salto finishes on the floor",
                                "the landing that completes a sideways somersault on the floor"],
        "FX_leap_jump_hop_start": ["the start of a leap / jump / hop (an airborne travelling step) on the floor",
                                   "when a leap or jump begins on the floor exercise",
                                   "the takeoff of a dance leap/jump on the floor"],
        "FX_leap_jump_hop_end":   ["the end of a leap / jump / hop on the floor (landing)",
                                   "when a leap or jump finishes on the floor",
                                   "the landing of a dance leap/jump on the floor"],
        "FX_turns_start": ["the start of a turn (a rotation about a vertical axis while standing) on the floor",
                           "when a pivoting turn begins on the floor exercise",
                           "the moment the gymnast starts spinning on one foot on the floor"],
        "FX_turns_end":   ["the end of a turn on the floor",
                           "when a pivoting turn finishes on the floor",
                           "the moment the gymnast stops spinning on the floor"],
        # ---- Uneven Bars ----
        "UB_circles_start": ["the start of a giant circle (the body swings in a full rotation around the bar) on the uneven bars",
                             "when a circling element begins around the bar",
                             "the moment the gymnast begins a full swing around the uneven bar"],
        "UB_circles_end":   ["the end of a giant circle on the uneven bars",
                             "when a circling element finishes around the bar",
                             "the moment the gymnast completes a full swing around the uneven bar"],
        "UB_transition_flight_start": ["the start of a flight element that transitions between the two bars (the body releases and travels through the air to the other bar)",
                                       "when a bar-to-bar flight (release to the other bar) begins",
                                       "the release into a transition flight between the uneven bars"],
        "UB_transition_flight_end":   ["the end of a bar-to-bar flight element (re-grasping the other bar)",
                                       "when a transition flight between the bars finishes",
                                       "the re-grasp that completes a bar-to-bar flight"],
        "UB_fligh_same_bar_start": ["the start of a flight element on the SAME bar (the body releases and re-grasps the same bar)",
                                    "when a release-and-regrasp on the same bar begins",
                                    "the release into a same-bar flight on the uneven bars"],
        "UB_fligh_same_bar_end":   ["the end of a same-bar flight element (re-grasping the same bar)",
                                    "when a same-bar release-and-regrasp finishes",
                                    "the re-grasp that completes a same-bar flight"],
        "UB_dismounts_start": ["the start of the dismount (the final element leaving the bars) from the uneven bars",
                               "when the dismount begins off the bars",
                               "the release into the dismount from the uneven bars"],
        "UB_dismounts_end":   ["the end of the dismount from the uneven bars (landing on the mat)",
                               "when the dismount finishes off the bars",
                               "the landing that completes the dismount from the uneven bars"],
        # ---- Vault (4 sequential phases, no start/end) ----
        "VT_0": ["the run-up and springboard contact phase of the vault (the approach before takeoff)",
                 "the approach / board-hurdle phase at the very beginning of a vault"],
        "VT_1": ["the pre-flight phase of the vault (the first flight from the board onto the vaulting table)",
                 "the first airborne phase of a vault, before the hands touch the table"],
        "VT_2": ["the support / push-off phase of the vault (the hands contact and push off the vaulting table)",
                 "the on-table repulsion phase of a vault"],
        "VT_3": ["the post-flight and landing phase of the vault (the second flight off the table and the landing)",
                 "the final airborne phase and landing of a vault"],
    },

    # ============================ FineDiving (OOD — eval only) ========================
    # DELIBERATELY reuse the SAME primitives so a frozen LLM already grounded on
    # 'somersault', 'twist', 'landing/entry' can zero-shot transfer.
    "finediving": {
        "Som(s).Tuck": ["a somersault performed in the TUCK position (knees pulled to the chest) during the dive",
                        "an aerial somersault (head-over-heels rotation) in a tight tuck shape",
                        "a tucked somersault rotation in the air off the diving board/platform"],
        "Som(s).Pike": ["a somersault performed in the PIKE position (body bent at the hips, legs straight) during the dive",
                        "an aerial somersault (head-over-heels rotation) in a piked shape",
                        "a piked somersault rotation in the air off the diving board/platform"],
        "Twist(s)":    ["a twist (a rotation of the body about its own long head-to-toe axis) during the dive",
                        "a twisting rotation of the body around its vertical axis while airborne",
                        "the diver twisting about their long axis in the air"],
        "Entry":       ["the moment the diver enters the water (the dive's landing)",
                        "when the diver breaks the water surface at the end of the dive",
                        "the water entry that ends the dive"],
    },
}


def descriptions_for(dataset, action_type):
    """Return the list of paraphrases for (dataset, action_type), or [] if unknown."""
    return DESCRIPTIONS.get(dataset, {}).get(action_type, [])


if __name__ == "__main__":
    # sanity: count coverage
    for ds, m in DESCRIPTIONS.items():
        n_par = sum(len(v) for v in m.values())
        print(f"{ds:12s}: {len(m)} action types, {n_par} paraphrases "
              f"({n_par/len(m):.1f} avg)")
