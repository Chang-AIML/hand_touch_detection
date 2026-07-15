#!/usr/bin/env python3
"""Generate GENERAL, transition-framed, motion-grounded descriptions for every distinct
(dataset,label,cleaned-comment) via deepseek-v4-flash on OpenRouter, concurrently.

De-dup: finegym's 80k events collapse to ~656 distinct comments once the routine-code
prefix "A_xxxx_xxxx;" is stripped. Total ~790 calls -> ~5 min, ~$0.2.

Anti-hallucination: the model may ONLY use facts in the label+comment; it must NOT invent
details and must GENERALIZE away instance specifics (counts, difficulty, connections).

Output: dpc/action_descriptions_generated.json =
  { "<dataset>|<label>|<cleaned_comment>": {"general": "...", "paraphrases": [...]} , ... }
plus class-level entries "<dataset>|<label>|" (empty comment) for no-comment datasets.
"""
import json, os, re, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

KEY = open("/home/chang/Dataset/Action_Spotting/openrouter.api").read().strip()
MODEL = "deepseek/deepseek-v4-flash"
N_PARA = 6
OUT = "/home/chang/Project/VLM_spotting/hand_touch_spotting/dpc/action_descriptions_generated.json"
SPOT = "/home/chang/Project/spot/data"
TM = "/home/chang/Dataset/Action_Spotting/TouchMoment"

SYS = f"""You rewrite ONE terse sports-annotation comment into a natural-language sentence describing the exact MOMENT (a single video frame) it labels, for a temporal-grounding model that must localize that frame. This is FAITHFUL REWRITING, not generation — you explain the comment, you do NOT create new content.

RULES:
1. FAITHFUL — keep EVERY fact in the comment, and add NOTHING. Do not add any direction, body position, apparatus, contact/landing surface, phase, or mechanics that is not literally written in the comment. Do not drop or generalize away any detail — keep counts and qualifiers exactly ("double", "1 turn", "3.5", "piked", "tucked", "stretched"). Do not reinterpret. Inventing anything is the worst possible failure.
2. UNKNOWN SKILLS — if the comment contains a named skill you are not certain about (e.g. "johnson", "tkatchev", "shaposhnikova", "gienger"), keep that word VERBATIM. Never guess what motion a named skill is.
3. EXPAND ABBREVIATIONS ONLY — the ONLY substitutions allowed: apparatus tags "FX"->"floor exercise", "BB"->"balance beam", "UB"->"uneven bars", "VT"->"vault"; and the literal words "salto"/"som"/"soms"->"somersault". Change nothing else.
4. TRANSITION FRAME — the label marks an INSTANT. Wrap the faithfully-rewritten content in that instant per the MOMENT HINT: "the moment/instant <content> begins / ends / takes off / lands / enters / strikes / touches / releases".
5. Make it one grammatical English sentence.
6. Output STRICT JSON only: {{"general": "<sentence>", "paraphrases": [<{N_PARA} sentences>]}}. Each paraphrase rewords the SAME sentence differently but keeps the EXACT same facts — add nothing, drop nothing.
"""

def clean(c):
    return re.sub(r"^A_\d+_\d+;", "", c or "").strip()

def moment_hint(ds, label):
    l = label.lower()
    if l.endswith("_start"): return "the START/onset frame — the instant this action BEGINS."
    if l.endswith("_end"):   return "the END frame — the instant this action FINISHES (often the landing)."
    if "takeoff" in l:       return "the TAKEOFF frame — the instant the athlete leaves the ice/ground into the air."
    if "landing" in l:       return "the LANDING frame — the instant the athlete returns to the ice/ground."
    if ds == "finediving":
        if label == "Entry": return "the ENTRY frame — the instant the diver's body enters the water (end of the dive)."
        return "the ONSET frame — the instant this aerial phase (somersault / twist) begins in the dive."
    if ds == "tennis":
        if "serve" in l:  return "the IMPACT frame — the instant the racket strikes the ball on a serve."
        if "swing" in l:  return "the IMPACT frame — the instant the racket strikes the ball on a groundstroke."
        if "bounce" in l: return "the BOUNCE frame — the instant the ball hits the court surface."
    if label.startswith("VT_"): return "one sequential PHASE-instant within a vault (run/first-flight/table-push/second-flight-landing)."
    if l == "touch":   return "the CONTACT frame — the instant the hand first touches the object."
    if l == "untouch": return "the RELEASE frame — the instant the hand releases/leaves the object."
    return "the labeled instant."

CTX = {"BB":"balance beam","FX":"floor exercise","UB":"uneven bars","VT":"vault"}
def context(ds, label):
    if ds == "finegym":
        return f"apparatus: {CTX.get(label.split('_')[0],'?')}"
    if ds == "tennis":
        return "far = far side of the court (from camera); near = near side"
    return ""

def build_worklist():
    work = {}  # key -> (dataset, label, comment)
    files = {"finegym":f"{SPOT}/finegym","fs_comp":f"{SPOT}/fs_comp","finediving":f"{SPOT}/finediving",
             "tennis":f"{SPOT}/tennis","touchmoment":TM}
    for ds, d in files.items():
        for sp in ["train","val","test"]:
            p = f"{d}/{sp}.json"
            if not os.path.exists(p): continue
            for r in json.load(open(p)):
                for e in r.get("events", []):
                    lab = e.get("label", e.get("type")); c = clean(e.get("comment"))
                    if c in ("auto","manual","extended dataset",""):   # no useful comment -> class level
                        key = f"{ds}|{lab}|"
                        work.setdefault(key, (ds, lab, ""))
                    else:
                        key = f"{ds}|{lab}|{c}"
                        work.setdefault(key, (ds, lab, c))
    return work

def call(ds, label, comment, retries=4):
    hint = moment_hint(ds, label); ctx = context(ds, label)
    user = (f"dataset: {ds}\nlabel: {label}\n"
            + (f"comment: {comment}\n" if comment else "comment: (none — describe from the label only)\n")
            + (f"context: {ctx}\n" if ctx else "")
            + f"MOMENT HINT: {hint}\nProduce {N_PARA} paraphrases.")
    body = json.dumps({"model":MODEL,"messages":[{"role":"system","content":SYS},{"role":"user","content":user}],
                       "max_tokens":1200,"temperature":0.6,"response_format":{"type":"json_object"}}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                    headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=120))
            txt = r["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", txt, re.S); obj = json.loads(m.group(0))
            g = obj["general"].strip(); ps = [p.strip() for p in obj.get("paraphrases",[]) if p.strip()]
            if g: return {"general":g,"paraphrases":ps}, r.get("usage",{}).get("cost",0)
        except Exception as ex:
            if a == retries-1: return {"error":str(ex)[:120]}, 0
            time.sleep(2*(a+1))
    return {"error":"exhausted"}, 0

def main():
    work = build_worklist()
    print(f"[gen] {len(work)} distinct (dataset,label,comment) to generate", flush=True)
    results = {}; total_cost = 0.0; bad = 0; done = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=64) as ex:
        futs = {ex.submit(call, ds, lab, c): k for k,(ds,lab,c) in work.items()}
        for f in as_completed(futs):
            k = futs[f]; res, cost = f.result(); results[k] = res; total_cost += cost; done += 1
            if "error" in res: bad += 1
            if done % 50 == 0 or done == len(work):
                el = time.time()-t0
                print(f"  {done}/{len(work)}  bad={bad}  ${total_cost:.3f}  {el:.0f}s  ({done/el:.1f}/s)", flush=True)
    json.dump(results, open(OUT,"w"), indent=1, ensure_ascii=False)
    print(f"[gen] DONE {done} items, {bad} errors, ${total_cost:.3f}, {time.time()-t0:.0f}s -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
