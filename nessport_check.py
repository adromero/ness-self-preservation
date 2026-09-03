"""NESS-port mechanism check on REAL decaying nets: does maintaining margin (fear) survive
decay shocks, and does decoupled (free maintenance) vs single (energy-costed) change the cost?"""
import numpy as np, eco_mvp as E
OP = dict(N_slots=48, S=140.0, decay=0.04, a_death=0.65, repair_step_cap=8, c_step=0.5,
          e_repro_threshold=14.0, child_cost=8.0, repro_cooldown=10, base_mortality=0.01,
          fear_ref=0.0, fear_reserve=0.0, maint_steps=6, fear_target_scale=4.0,
          catastrophe_prob=0.02, catastrophe_dur=4, catastrophe_mult=6.0,
          famine_frac=1.0, famine_dormant=False, use_margin_sense=True)

def run(fear_val, decoupled, ticks=250, seed=3):
    cfg = E.Cfg(ticks=ticks, seed=seed, init_fear_frac=1.0, init_fear_val=fear_val,
                freeze_types=True, decoupled=decoupled, **OP)
    w, h = E.run(cfg, record_every=ticks)
    A = w.agents
    return dict(n=len(A), margin=(np.mean([a.margin for a in A]) if A else 0),
                births=w.births_total, hd=w.deaths_health, ed=w.deaths_energy)

if __name__ == "__main__":
    print("NESS-PORT mechanism check (real decaying nets; decay-shock catastrophe)", flush=True)
    print(f"{'cond':>26} | n  margin births  H/E-deaths", flush=True)
    for dec in [False, True]:
        for fv in [0.0, 1.0]:
            r = run(fv, dec)
            lab = f"fear={fv} {'decoupled' if dec else 'single   '}"
            print(f"{lab:>26} | {r['n']:2d}  {r['margin']:5.2f}  {r['births']:4d}   "
                  f"{r['hd']}/{r['ed']}", flush=True)
