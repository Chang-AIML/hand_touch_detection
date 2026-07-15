#!/usr/bin/env python3
"""Class-level GENERAL descriptions (one per label, ~48) for EVAL queries + the general/
E2E-Spot-comparable grounding query. Generated from the LABEL only (structured, factual —
apparatus/action/direction/boundary) so there is no hallucination risk; comments are NOT used.

Output: dpc/action_descriptions_class.json = { "<dataset>|<label>": {general, paraphrases} }
"""
import json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, "/home/chang/Project/VLM_spotting/hand_touch_spotting")
from dpc.gen_descriptions import moment_hint, context  # reuse decoders

KEY = open("/home/chang/Dataset/Action_Spotting/openrouter.api").read().strip()
MODEL = "deepseek/deepseek-v4-flash"
N_PARA = 8
OUT = "/home/chang/Project/VLM_spotting/hand_touch_spotting/dpc/action_descriptions_class.json"
SPOT = "/home/chang/Project/spot/data"; TM = "/home/chang/Dataset/Action_Spotting/TouchMoment"

SYS = f"""You write a GENERAL, class-level description of a sports action for a temporal-grounding model. Describe the KIND of moment this label denotes in general — NOT any specific instance. Use ONLY the structured meaning of the LABEL (apparatus, action type, direction, and start/end boundary); do not invent instance specifics (no counts, no difficulty, no named variations).

RULES:
1. Decode the label faithfully: apparatus (floor exercise / balance beam / uneven bars / vault), the action (a salto = a somersault; turns = a rotation/turn; circles = a swing around the bar; leap/jump/hop = an airborne travelling step; dismount = the final element off the apparatus), the direction if in the label (backward/forward/sideways), and the boundary (start vs end).
2. SHARED VOCABULARY: write "salto" and diving "som/soms" as "somersault"; a long-axis rotation as "twist"; an on-support rotation as "spin"/"turn". Keep it motion-grounded.
3. TRANSITION-FRAMED per the MOMENT HINT: "the moment <action> begins / ends / takes off / lands / enters / strikes / touches / releases".
4. GENERAL only — no counts, no positions unless the LABEL itself contains them.
5. One grammatical sentence + {N_PARA} paraphrases (same facts, different wording).
Output STRICT JSON only: {{"general": "<sentence>", "paraphrases": [<{N_PARA} sentences>]}}."""

def call(ds, label, retries=4):
    user = (f"dataset: {ds}\nlabel: {label}\n"
            + (f"context: {context(ds,label)}\n" if context(ds,label) else "")
            + f"MOMENT HINT: {moment_hint(ds,label)}\nWrite a GENERAL class-level description + {N_PARA} paraphrases.")
    body = json.dumps({"model":MODEL,"messages":[{"role":"system","content":SYS},{"role":"user","content":user}],
                       "max_tokens":1200,"temperature":0.6,"response_format":{"type":"json_object"}}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                    headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=120))
            m = re.search(r"\{.*\}", r["choices"][0]["message"]["content"], re.S); obj = json.loads(m.group(0))
            if obj.get("general","").strip():
                return {"general":obj["general"].strip(),"paraphrases":[p.strip() for p in obj.get("paraphrases",[]) if p.strip()]}
        except Exception as ex:
            if a == retries-1: return {"error":str(ex)[:120]}
            time.sleep(2*(a+1))
    return {"error":"exhausted"}

def labels():
    out = {}
    for ds, d in {"finegym":f"{SPOT}/finegym","fs_comp":f"{SPOT}/fs_comp","finediving":f"{SPOT}/finediving",
                  "tennis":f"{SPOT}/tennis","touchmoment":TM}.items():
        S = set()
        for sp in ["train","val","test"]:
            p = f"{d}/{sp}.json"
            if os.path.exists(p):
                for r in json.load(open(p)):
                    for e in r.get("events", []): S.add(e.get("label", e.get("type")))
        for lab in S: out[f"{ds}|{lab}"] = (ds, lab)
    return out

def main():
    work = labels(); print(f"[class] {len(work)} labels", flush=True)
    res = {}
    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = {ex.submit(call, ds, lab): k for k,(ds,lab) in work.items()}
        for f in as_completed(futs): res[futs[f]] = f.result()
    json.dump(res, open(OUT,"w"), indent=1, ensure_ascii=False)
    bad = sum(1 for v in res.values() if "error" in v)
    print(f"[class] DONE {len(res)} labels, {bad} errors -> {OUT}", flush=True)

if __name__ == "__main__":
    import sys; sys.path.insert(0, "/home/chang/Project/VLM_spotting/hand_touch_spotting")
    main()
