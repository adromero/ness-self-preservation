#!/usr/bin/env python3
"""McNamara-Houston fat-reserve bet-hedge: fear = hoard an ENERGY reserve; catastrophe = FAMINE
(supply cut). Lean (fearless) agents spend energy on offspring and starve in the famine; fat
(fearful) agents survive on reserves. Rate-limited reproduction makes the reserve ~free.
Check: (calm) births>0 & fearful hold a bigger reserve; (famine) fearful starve LESS than fearless.
"""
import numpy as np
import eco_mvp as E

# gentle survival (cheap) + generous income (surplus) + rate limit + fat reserve via fear
ECON = dict(decay=0.02, S=250.0, income_cap=12.0, a_death=0.60, N_slots=96,
            repair_step_cap=10, c_step=0.5, e_repro_threshold=12.0, child_cost=8.0,
            repro_cooldown=15, fear_ref=0.0, fear_reserve=25.0, famine_dormant=True,
            base_mortality=0.015, famine_drain=12.0)
FAMINE = dict(catastrophe_prob=0.006, catastrophe_dur=30, catastrophe_mult=1.0, famine_frac=0.0)


def run_one(fear_val, famine, ticks=500, seed=5):
    kw = dict(FAMINE) if famine else {}
    cfg = E.Cfg(ticks=ticks, seed=seed, init_fear_frac=1.0, init_fear_val=fear_val,
                freeze_types=True, **kw, **ECON)
    w, h = E.run(cfg, record_every=500)
    A = w.agents
    return dict(n=len(A), births=w.births_total,
                e=(np.mean([a.e for a in A]) if A else 0.0),
                hd=w.deaths_health, ed=w.deaths_energy)


if __name__ == "__main__":
    print("FAMINE / FAT-RESERVE DIAGNOSTIC (fear=0 lean vs fear=0.6 fat)", flush=True)
    print(f"{'cond':>16} | n  births  mean_E  deaths(H/E)   <- famine kills via Energy death", flush=True)
    for famine, lab in [(False, "calm"), (True, "famine")]:
        for fv in [0.0, 0.6]:
            r = run_one(fv, famine)
            print(f"fear={fv} {lab:>9} | {r['n']:3d}  {r['births']:4d}   {r['e']:5.1f}   "
                  f"{r['hd']}/{r['ed']}", flush=True)
