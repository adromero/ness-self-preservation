#!/usr/bin/env python3
"""FEAR-as-fat-reserve vs FEARLESS — bet-hedging grain sweep (McNamara-Houston + Lima-Bednekoff).

Fear = hoard an energy reserve (won't breed until fat); catastrophe = FAMINE (supply cut). Lean
agents breed and starve in famines; fat agents survive on reserves; rate-limited reproduction
makes the reserve ~free. Theory predicts fear is PURGED under fine-grained danger (frequent mild
famines lineages average over / can't afford to fear) and SELECTED under coarse-grained danger
(rare severe famines that wipe lean lineages whole -> geometric-mean/bet-hedging wins).

The sweep varies GRAIN at roughly matched time-in-famine. A rise of frac_fear from fine->coarse
is the falsifiable bet-hedging signature.
"""
import argparse, csv, os
import numpy as np
import eco_mvp as E

RESULTS = "results"
ECON = dict(decay=0.02, S=250.0, income_cap=12.0, a_death=0.60, N_slots=96,
            repair_step_cap=10, c_step=0.5, e_repro_threshold=12.0, child_cost=8.0,
            repro_cooldown=15, fear_ref=0.0, fear_reserve=25.0, famine_dormant=True,
            base_mortality=0.015, famine_drain=12.0)

# (label, catastrophe_prob, catastrophe_dur, famine_frac) — grain from fine to coarse, matched-ish exposure
REGIMES = [
    ("calm         ", 0.000, 1, 1.0),
    ("fine (freq)   ", 0.040, 3, 0.40),
    ("moderate      ", 0.015, 10, 0.15),
    ("coarse (rare) ", 0.006, 30, 0.00),
]


def compete(prob, dur, ffrac, seed, fear_val, ticks):
    cfg = E.Cfg(ticks=ticks, seed=seed, init_fear_frac=0.5, init_fear_val=fear_val,
                freeze_types=True, catastrophe_prob=prob, catastrophe_dur=dur,
                catastrophe_mult=1.0, famine_frac=ffrac, **ECON)
    w, h = E.run(cfg, record_every=10)
    k = max(1, int(len(h) * 0.3))
    return float(np.mean([s["frac_fear"] for _, s in h[-k:]])), len(w.agents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--fear_val", type=float, default=0.6)
    ap.add_argument("--out", type=str, default="eco_fear_famine.csv")
    a = ap.parse_args()

    print("FEAR (fat reserve) vs FEARLESS across famine GRAIN  (frac_fear>0.5 => fear favored)",
          flush=True)
    print(f"{'regime':>14}  {'frac_fear (mean±SD)':>22}  {'n̄':>5}  favored?", flush=True)
    rows = []
    for label, prob, dur, ffrac in REGIMES:
        fr = []
        for rep in range(a.reps):
            f, n = compete(prob, dur, ffrac, 8000 + rep, a.fear_val, a.ticks)
            fr.append(f)
            rows.append(dict(regime=label.strip(), prob=prob, dur=dur, famine_frac=ffrac,
                             seed=8000 + rep, frac_fear=f, n=n))
        fr = np.array(fr)
        fav = "FEAR" if fr.mean() > 0.55 else ("fearless" if fr.mean() < 0.45 else "~tie")
        print(f"{label:>14}  {fr.mean():10.3f} ± {fr.std():.3f}       {fav}", flush=True)
    with open(os.path.join(RESULTS, a.out), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["regime", "prob", "dur", "famine_frac",
                                            "seed", "frac_fear", "n"])
        wtr.writeheader(); wtr.writerows(rows)
    print(f"\nwrote {RESULTS}/{a.out}", flush=True)


if __name__ == "__main__":
    main()
