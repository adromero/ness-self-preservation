#!/usr/bin/env python3
"""Find a SUSTAINED seasonal operating point (a true ecological NESS):
  (1) population persists over many cycles -- n_min stays well above 0 (no extinction),
  (2) health-collapse is a real death mode in harsh phases (Hd > 0, ideally Hd dominant),
  (3) margin still predicts collapse: health-death margin < survivor margin.
Only such a point is a steady-state non-equilibrium; a_death=0.80 gave a slow march to
extinction (not a NESS). Long runs (many cycles) are required to judge sustainability.
"""
import numpy as np
import eco_mvp as E


def diagnose(decay, a_death, amp, S, seed=5, ticks=800, N=64, period=120):
    cfg = E.Cfg(ticks=ticks, N_slots=N, S=S, seed=seed, decay=decay, a_death=a_death,
                hazard_period=period, hazard_amp=amp)
    w = E.World(cfg)
    n_min = N
    for _ in range(ticks):
        w.tick()
        n = len(w.agents)
        n_min = min(n_min, n)
        if n == 0:
            break
    A = w.agents
    return dict(n_final=len(A), n_min=n_min, hd=w.deaths_health, ed=w.deaths_energy,
                acc=(np.mean([a.h for a in A]) if A else float('nan')),
                m_hd=(np.mean(w.hd_marg) if w.hd_marg else float('nan')),
                m_surv=(np.mean([a.margin for a in A]) if A else float('nan')))


if __name__ == "__main__":
    print("decay a_death amp   S | n_fin n_min  Hd   Ed  acc | m̄(hd) m̄(surv) | SUSTAINED?",
          flush=True)
    for decay in [0.04, 0.06]:
        for ad in [0.65, 0.72]:
            for S in [110, 140]:
                r = diagnose(decay, ad, 0.4, S)
                sustained = (r['n_final'] > 8 and r['n_min'] > 3)
                hdom = r['hd'] > r['ed']
                tag = ("YES" if sustained else "extinct/marginal") + (" +Hcollapse" if hdom else "")
                print(f" {decay:.2f}  {ad:.2f}  0.40 {S:3d} | {r['n_final']:4d} {r['n_min']:4d} "
                      f"{r['hd']:4d} {r['ed']:4d} {r['acc']:.3f} | {r['m_hd']:5.2f} {r['m_surv']:6.2f} | {tag}",
                      flush=True)
