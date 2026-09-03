import numpy as np, eco_life as L
def final_fear(init, reserve, seeds=4, ticks=3500):
    fs=[]
    for s in range(seeds):
        cfg=L.Cfg(seed=s, ticks=ticks, init_fear=init, fear_reserve=reserve)
        w,traj=L.evolve(cfg)
        fs.append(traj[-1][2] if w.agents else np.nan)
    return np.nanmean(fs), np.nanstd(fs)
print("VALLEY-LOCK TEST: seed fear high in FULL ecology; is it maintained (adaptive) or eroded?", flush=True)
print(f"{'init_fear':>9} {'FULL(reserve=20)':>20} {'NEUTRAL(reserve=0)':>22}", flush=True)
for init in [0.0, 0.5, 1.0, 1.5]:
    fm,fs=final_fear(init,20.0)
    nm,ns=final_fear(init,0.0)
    tag=""
    if fm>nm+0.05: tag="  <- fear MAINTAINED above bias (adaptive)"
    elif fm<nm-0.05: tag="  <- eroded below bias (disfavored)"
    print(f"{init:9.2f}   {fm:.3f} ± {fs:.3f}      {nm:.3f} ± {ns:.3f}{tag}", flush=True)
