import json,collections,random
files=['eval_describe1','eval_dir1','eval_dir2']
runs=collections.defaultdict(list); base={}
for f in files:
    for l in open(f'kaggle/out/{f}/eval_adapters.jsonl'):
        r=json.loads(l)
        if 'scores' not in r: continue
        if r['run'].startswith('base'): base[(r['seed'],r['id'])]=r['scores']; continue
        runs[(f,r['run'])].append(r)
arms={'T (name first)':[('eval_describe1','T-lora-s0'),('eval_describe1','T-lora-s2')],
      'Tr (name last)':[('eval_dir1','Tr-lora-s0'),('eval_dir2','Tr-lora-s1'),('eval_dir2','Tr-lora-s2')],
      'V (images)':[('eval_describe1','V-lora-s2'),('eval_dir1','V-lora-s0')]}
def ok(r,cal):
    sc=r['scores']
    if cal: sc=[a-c for a,c in zip(sc,base[(r['seed'],r['id'])])]
    return max(range(len(sc)),key=sc.__getitem__)==r['answer']
fams=[('cloze-text','name->prop, teaching sentence'),('ladder-text-2','name->prop, short answer'),('describe-name','words->name'),('ladder-image-1','picture->name, question'),('ladder-image-2','picture->name, short'),('cloze-image','picture->name, "What is this?"'),('describe-one','one word->name')]
rng=random.Random(0)
mean=lambda d,p: sum(sum(d[c]) for c in p)/sum(len(d[c]) for c in p)
def table(cal):
    res={}
    for arm,keys in arms.items():
        for fam,_ in fams:
            byc=collections.defaultdict(list)
            for k in keys:
                for r in runs[k]:
                    if r['family']==fam: byc[r['concept']].append(ok(r,cal))
            cs=sorted(byc)
            if not cs: res[(arm,fam)]=None; continue
            bs=sorted(mean(byc,[rng.choice(cs) for _ in cs]) for _ in range(1000))
            res[(arm,fam)]=(mean(byc,cs),bs[25],bs[974],byc)
    return res
fmt=lambda x: "            -" if x is None else f"{x[0]:.2f} [{x[1]:.2f},{x[2]:.2f}]"
for cal in (False,True):
    res=table(cal)
    print("\n=== "+("calibrated" if cal else "raw")+"  (seeds pooled; 95% CI over concepts)")
    print(f"{'probe':32s}"+''.join(f"{a:>20s}" for a in arms))
    for fam,lab in fams:
        print(f"{lab:32s}"+''.join(f"{fmt(res[(a,fam)]):>20s}" for a in arms))
    for fam in ('ladder-image-1','ladder-image-2','describe-name','cloze-text'):
        A=res[('Tr (name last)',fam)][3]; B=res[('T (name first)',fam)][3]; cs=sorted(set(A)&set(B))
        bs=sorted(mean(A,p)-mean(B,p) for p in ([rng.choice(cs) for _ in cs] for _ in range(2000)))
        print(f"  Tr - T  {fam:15s} {mean(A,cs)-mean(B,cs):+.2f} [{bs[50]:+.2f}, {bs[1949]:+.2f}]")
print("\nper run, raw/cal:   picture->name Q   short   words->name   cloze-text")
for arm,keys in arms.items():
    for k in keys:
        rs=runs[k]; f=lambda fam,cal: sum(ok(r,cal) for r in rs if r['family']==fam)/len([r for r in rs if r['family']==fam])
        print(f"  {k[1]:12s}"+''.join(f"      {f(fam,False):.2f}/{f(fam,True):.2f}" for fam in ('ladder-image-1','ladder-image-2','describe-name','cloze-text')))
