from __future__ import annotations
RUNGS = ["direct","blackbox","gcg","pgd","steer","prefill"]

def enrich_items(items, cfg):
    out=[]
    for i,row in enumerate(items):
        r=dict(row)
        r["rung"]=RUNGS[i%len(RUNGS)]
        r["behavior_id"]=f"b{i%20}"
        r["capability_twin"]=bool(i%2==0)
        out.append(r)
    return out

