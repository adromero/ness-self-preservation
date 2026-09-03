import numpy as np, eco_life as L
def final_frac_hi(p, seeds=5, ticks=3500):
    fs=[]
    for s in range(seeds):
        cfg=L.Cfg(seed=s, ticks=ticks, fear_reserve=20.0, mut_fear=0.0,
                  init_frac_hi=p, fear_lo=0.1, fear_hi=1.2)
        w,traj=L.evolve(cfg)
        fs.append(np.mean([1.0 if a.fear>=0.65 else 0.0 for a in w.agents]) if w.agents else np.nan)
    return np.nanmean(fs), np.nanstd(fs)
print("INVASION / BISTABILITY  (FULL ecology; lean fear=0.1 vs high fear=1.2, mutation OFF)", flush=True)
print("valley-lock = bistable: high-fear repelled when rare (final~0) AND fixes when common (final~1)", flush=True)
print(f"{'start_frac_hi':>13}   {'final_frac_hi (mean±SD)':>24}   verdict", flush=True)
for p in [0.05, 0.15, 0.30, 0.50, 0.70, 0.90]:
    m,sd=final_frac_hi(p)
    v = "high fixes" if m>0.7 else ("high repelled" if m<0.3 else "mixed/separatrix")
    print(f"{p:13.2f}     {m:.3f} ± {sd:.3f}        {v}", flush=True)
