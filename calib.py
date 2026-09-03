#!/usr/bin/env python3
"""Find an operating point for the margin-foresight redesign where:
  (1) death is HEALTH-collapse (accuracy < a_death), not energy starvation,
  (2) the population persists (no extinction),
  (3) margin PREDICTS death: agents that die had lower margin than survivors.
Then margin-awareness is decision-relevant and the invasion assay can test its value.
"""
import numpy as np
import eco_mvp as E


def diagnose(a_death, decay, shock_p, seed=5, ticks=160, N=64, S=90):
    cfg = E.Cfg(ticks=ticks, N_slots=N, S=S, seed=seed, decay=decay, a_death=a_death,
                shock_prob=shock_p, shock_mult=4.0)
    w = E.World(cfg)
    died_m = []
    for _ in range(ticks):
        pre = {a.aid: a.margin for a in w.agents}
        w.tick()
        now = {a.aid for a in w.agents}
        for aid, mg in pre.items():
            if aid not in now:
                died_m.append(mg)
    surv = [a.margin for a in w.agents]
    return dict(hd=w.deaths_health, ed=w.deaths_energy, n=len(w.agents),
                acc=(np.mean([a.h for a in w.agents]) if w.agents else float('nan')),
                m_hd=(np.mean(w.hd_marg) if w.hd_marg else float('nan')),
                m_ed=(np.mean(w.ed_marg) if w.ed_marg else float('nan')),
                m_surv=(np.mean(surv) if surv else float('nan')))


if __name__ == "__main__":
    print("a_death decay shk |  Hd  Ed   n  acc | m̄(health-death) m̄(energy-death) m̄(surv)", flush=True)
    for dec in [0.05, 0.07, 0.09]:
        for ad in [0.80, 0.84]:
            for shk in [0.05, 0.12]:
                r = diagnose(ad, dec, shk)
                print(f"  {ad:.2f}  {dec:.2f} {shk:.2f}| {r['hd']:3d} {r['ed']:3d} {r['n']:3d} {r['acc']:.3f}"
                      f" |    {r['m_hd']:6.2f}          {r['m_ed']:6.2f}       {r['m_surv']:6.2f}", flush=True)
