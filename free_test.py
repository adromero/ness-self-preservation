import numpy as np, eco_life as L
def q(start, upkeep, fprob, seeds=3, ticks=2500):
    fs=[]
    for s in range(seeds):
        cfg=L.Cfg(seed=s, ticks=ticks, mut_fear=0.0, init_frac_hi=start, fear_lo=0.1, fear_hi=1.2,
                  safety_currency=True, safety_upkeep=upkeep, safety_max=20.0, safety_supply=3.0, famine_prob=fprob)
        w,traj=L.evolve(cfg)
        fs.append(np.mean([1.0 if a.fear>=0.65 else 0.0 for a in w.agents]) if w.agents else np.nan)
    return np.nanmean(fs)
print("two-resource decoupled: when does high-fear (start 0.9) survive / (start 0.1) invade?", flush=True)
print(f"{'upkeep':>7} {'famine_prob':>11} {'0.9->hi':>9} {'0.1->hi':>9}   reading", flush=True)
for up in [0.0, 0.005]:
    for fp in [0.004, 0.02, 0.05]:
        h9=q(0.9,up,fp); h1=q(0.1,up,fp)
        rd = "fear FIXES" if h1>0.7 else ("fear STABLE" if h9>0.7 else ("partial" if h9>0.3 else "repelled"))
        print(f"{up:7.3f} {fp:11.3f}   {h9:.3f}     {h1:.3f}    {rd}", flush=True)
