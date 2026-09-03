"""Neural-substrate invasion assay: does high-fear (margin-maintenance) resist/invade lean,
under single vs decoupled maintenance currency?  Confirms the ported selection outcome."""
import numpy as np, eco_mvp as E
OP = dict(N_slots=48, S=140.0, decay=0.04, a_death=0.65, repair_step_cap=8, c_step=0.5,
          e_repro_threshold=14.0, child_cost=8.0, repro_cooldown=10, base_mortality=0.01,
          fear_ref=0.0, fear_reserve=0.0, maint_steps=6, fear_target_scale=4.0,
          catastrophe_prob=0.02, catastrophe_dur=4, catastrophe_mult=6.0,
          famine_frac=1.0, famine_dormant=False, use_margin_sense=True)
def invade(decoupled, start, seed, ticks=1400):
    cfg=E.Cfg(ticks=ticks, seed=seed, init_fear_frac=start, init_fear_val=1.0,
              freeze_types=True, decoupled=decoupled, **OP)
    w,h=E.run(cfg, record_every=20)
    k=max(1,int(len(h)*0.3))
    return float(np.mean([s['frac_fear'] for _,s in h[-k:]])), len(w.agents)
if __name__=="__main__":
    print("NEURAL invasion: frac high-fear (margin-maintainers) final; single vs decoupled currency", flush=True)
    print(f"{'currency':>10} {'start':>6} {'final_frac_fear':>16}   verdict", flush=True)
    for dec,lab in [(False,"single"),(True,"decoupled")]:
        for start in [0.2, 0.8]:
            fr=[]
            for s in range(3):
                f,n=invade(dec,start,100+s); fr.append(f)
            m=np.mean(fr)
            v="fear WINS" if m>0.7 else ("fear repelled" if m<0.3 else "partial")
            print(f"{lab:>10} {start:6.2f}   {m:.3f} ± {np.std(fr):.3f}     {v}", flush=True)
