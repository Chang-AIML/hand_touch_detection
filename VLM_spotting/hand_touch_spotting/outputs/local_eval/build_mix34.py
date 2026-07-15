"""Build the decisive 2x2 swap ckpts per the user's grouping:
   Group A = q + in_ln + a3.*   (whole Q-Former, ~2.37M)
   Group B = out.*              (readout, ~25.2M)
   MIX-3 = A(s1200) + B(s750)  == base s750, overwrite every NON-out.* key with s1200
   MIX-4 = A(s750)  + B(s1200) == base s750, overwrite every out.* key with s1200
"""
import torch, collections

D = "/home/chang/Project/VLM_spotting/hand_touch_spotting/outputs/local_eval"
c750  = torch.load(f"{D}/conn_s750.pt",  map_location="cpu", weights_only=False)
c1200 = torch.load(f"{D}/conn_s1200.pt", map_location="cpu", weights_only=False)
sd750, sd1200 = c750["fc"], c1200["fc"]
assert set(sd750) == set(sd1200), "key mismatch between s750 and s1200"

# ---- show the partition so grouping is auditable ----
def numel(sd, pred):
    return sum(v.numel() for k, v in sd.items() if pred(k))
isB = lambda k: k.startswith("out.")
isA = lambda k: not isB(k)
groupsA = collections.Counter(k.split(".")[0] for k in sd750 if isA(k))
print("Group A (Q-Former) top-level modules:", dict(groupsA))
print("Group B (out.*) keys:", [k for k in sd750 if isB(k)])
print(f"Group A params = {numel(sd750, isA)/1e6:.3f}M | Group B params = {numel(sd750, isB)/1e6:.3f}M")

# ---- how much does each group actually differ s750 vs s1200 (sanity: swaps are non-trivial) ----
dA = sum((sd750[k]-sd1200[k]).float().pow(2).sum().item() for k in sd750 if isA(k)) ** 0.5
dB = sum((sd750[k]-sd1200[k]).float().pow(2).sum().item() for k in sd750 if isB(k)) ** 0.5
print(f"||A_750 - A_1200|| = {dA:.4f} | ||B_750 - B_1200|| = {dB:.4f}")

def build(base, other, take_from_other):
    sd = {k: v.clone() for k, v in base.items()}
    for k in sd:
        if take_from_other(k):
            sd[k] = other[k].clone()
    return sd

# MIX-3: base s750, take NON-out from s1200  -> QF=s1200, out=s750
mix3 = build(sd750, sd1200, isA)
# MIX-4: base s750, take out.* from s1200    -> QF=s750,  out=s1200
mix4 = build(sd750, sd1200, isB)

# verify assignments
assert all(torch.equal(mix3[k], sd1200[k]) for k in mix3 if isA(k)), "MIX3 QF != s1200"
assert all(torch.equal(mix3[k], sd750[k])  for k in mix3 if isB(k)), "MIX3 out != s750"
assert all(torch.equal(mix4[k], sd750[k])  for k in mix4 if isA(k)), "MIX4 QF != s750"
assert all(torch.equal(mix4[k], sd1200[k]) for k in mix4 if isB(k)), "MIX4 out != s1200"
print("assignment checks PASSED")

# sanity vs my old MIX3 (out=s750, rest=s1200) -> should be identical to MIX-3
try:
    old3 = torch.load(f"{D}/conn_MIX3_out750_rest1200.pt", map_location="cpu", weights_only=False)["fc"]
    same = all(torch.equal(mix3[k], old3[k]) for k in mix3)
    print(f"MIX-3 == old conn_MIX3_out750_rest1200.pt ? {same}")
except Exception as e:
    print("old MIX3 compare skipped:", e)

for name, sd in [("conn_MIX3_qf1200_out750", mix3), ("conn_MIX4_qf750_out1200", mix4)]:
    torch.save({"fc": sd, "gstep": 0, "note": name}, f"{D}/{name}.pt")
    print("saved", name)
