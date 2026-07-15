"""VOCABULARY LENS: is a V-JEPA frame token a 'word' (near Qwen's embedding manifold) or an
'artificial retrieval key' (off-manifold)?  Decides candidate-1 vs candidate-2/3.

Method (centered cosine, since embedding spaces are anisotropic; RMS-match already killed the norm tell):
  center everything by the mean word embedding, unit-normalize, cosine nearest-neighbor.
  - frame tokens : FrameCompress(s750) output at GT event frames of real windows
  - LOWER bound  : random norm-matched vectors  (chance floor)
  - REFERENCE    : Qwen-ViT anchor tokens (Qwen's OWN non-word input embeds)
  - UPPER ref    : real word tokens -> nearest OTHER word (typical inter-word cosine)
Read: frame-token max-cos HIGH (>~0.35) + semantic word -> candidate-1 (semantic). LOW (~floor) -> key.
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, "/home/chang/Project/VLM_spotting/hand_touch_spotting")
from dpc.frame_source import DirFrameSource
from dpc.vjepa import OnlineVJEPA
from models.wrapper import QwenWrapper
from models.frame_compress import FrameCompress

dev="cuda"
os.environ.setdefault("VJEPA_REPO","/home/chang/Project/vlm_deps/vjepa2")
ANN=os.environ["AS_ANN_DIR"]; ROOT=os.environ["AS_DATA_ROOT"]
W=QwenWrapper(model_id=os.environ["QWEN_PATH"],device=dev,dtype=torch.bfloat16)
fc=FrameCompress(768,8,W.d_llm,n_heads=4).to(dev,torch.float32)
fc.load_state_dict(torch.load("outputs/local_eval/conn_s750.pt",map_location="cpu",weights_only=False)["fc"],strict=True)
fc.set_target_rms_from(W.embed_tokens.weight)
vj=OnlineVJEPA(device=dev,dtype=torch.float16,compute_dtype=torch.bfloat16)
fs=DirFrameSource(ROOT)

E=W.embed_tokens.weight.detach().float()            # (V,4096)
mu=E.mean(0,keepdim=True)
En=E-mu; En=En/En.norm(dim=-1,keepdim=True)         # centered, unit
tok=W.tokenizer
def nn_words(vec,k=6):                               # centered cosine NN over vocab
    v=(vec.float()-mu[0]); v=v/v.norm()
    cos=(En@v)
    top=torch.topk(cos,k)
    return float(top.values[0]), [(tok.decode([int(i)]).strip(), round(float(c),3)) for c,i in zip(top.values,top.indices)]

# --- frame tokens at GT event frames of a few real in-domain windows ---
recs=json.load(open(f"{ANN}/touchmoment/val.json"))[:6]
frame_max=[]; frame_words=[]
for r in recs:
    vid=r["video_id"]; N=int(r["num_frames"]); 
    fr=fs.load_window(vid,0,min(N,600)); T=int(fr.shape[0])
    grid=vj.extract_grid(fr); motion=fc(grid,None,T).float()   # (T,4096)
    evs=[int(e["frame"]) for e in r.get("events",[]) if 0<=int(e["frame"])<T][:3]
    for f in evs:
        mx,ws=nn_words(motion[f]); frame_max.append(mx); frame_words.append(ws[:3])

# --- random norm-matched lower bound ---
nrm=float(fc(grid,None,T).float().norm(dim=-1).mean())
rnd=[nn_words(torch.randn(4096,device=dev)*nrm/ (4096**0.5))[0] for _ in range(20)]

# --- Qwen-ViT anchor tokens reference ---
imgs=[np.asarray(fr[min(s*15,T-1)]) for s in range(min(4, T//15+1))]
anch=W.vit_anchor_groups(imgs,max_side=252)          # list of (g_s,d)
anch_max=[nn_words(g[j])[0] for g in anch for j in range(min(8,g.shape[0]))]

# --- real word tokens -> nearest OTHER word (upper ref) ---
import random as _r; ids=_r.Random(0).sample(range(E.shape[0]),20)
word_ref=[]
for i in ids:
    v=En[i]; cos=En@v; cos[i]=-1; word_ref.append(float(cos.max()))

print("\n================ VOCABULARY LENS (centered cosine, s750) ================")
print(f"frame token  -> nearest word:  max-cos  mean={np.mean(frame_max):.3f}  median={np.median(frame_max):.3f}  range=[{min(frame_max):.3f},{max(frame_max):.3f}]  (n={len(frame_max)})")
print(f"random vector-> nearest word:  max-cos  mean={np.mean(rnd):.3f}  (chance FLOOR)")
print(f"Qwen-ViT anchor-> nearest word:max-cos  mean={np.mean(anch_max):.3f}  (Qwen's own non-word input)")
print(f"real word    -> nearest OTHER: max-cos  mean={np.mean(word_ref):.3f}  (typical inter-word, UPPER ref)")
print("\nnearest words for a few frame tokens (word, cos):")
for ws in frame_words[:8]: print("  ", ws)
print("\nREAD: frame≈word-ref & semantic words -> CANDIDATE-1 (frame IS a word).")
print("      frame≈random-floor (<< word-ref) -> CANDIDATE-2/3 (off-manifold retrieval KEY, no word).")
