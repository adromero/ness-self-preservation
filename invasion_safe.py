import numpy as np, eco_life as L
def final_hi(start, upkeep, seeds=4, ticks=3500):
    fs=[]
    for s in range(seeds):
        cfg=L.Cfg(seed=s, ticks=ticks, mut_fear=0.0, init_frac_hi=start, fear_lo=0.1, fear_hi=1.2,
                  safety_currency=True, safety_upkeep=upkeep, safety_max=20.0)
        w,traj=L.evolve(cfg)
        fs.append(np.mean([1.0 if a.fear>=0.65 else 0.0 for a in w.agents]) if w.agents else np.nan)
    return np.nanmean(fs), np.nanstd(fs)
print("SAFETY-CURRENCY CONFIRMATION: decoupled buffer (breeding NOT fear-gated), invasion assay.", flush=True)
print("Compare to single-currency where high-fear was REPELLED from every start (0.9 -> 0.33 eroding).", flush=True)
print("If high-fear now RESISTS invasion (0.9 stays ~1) / INVADES (0.1 rises), the currency is the key.\n", flush=True)
print(f"{'upkeep':>7} {'start=0.1 -> final':>20} {'start=0.9 -> final':>20}   reading", flush=True)
for up in [0.001, 0.005, 0.02]:
    lo_m,lo_s = final_hi(0.1, up); hi_m,hi_s = final_hi(0.9, up)
    if lo_m>0.7: rd="fear FIXES from rare (fully selected)"
    elif hi_m>0.7: rd="fear STABLE when common (decoupling works)"
    elif hi_m<0.3: rd="still repelled (currency insufficient)"
    else: rd="partial"
    print(f"{up:7.3f}   {lo_m:.3f} ± {lo_s:.3f}      {hi_m:.3f} ± {hi_s:.3f}     {rd}", flush=True)
