#!/usr/bin/env python3
"""C1 information control x actionable-tau environment.

Head-to-head competition of three FROZEN lineages seeded in equal thirds:
  L0        - blind (senses h, e only)
  L1-real   - death-aware: senses true decay-slope + time-to-death tau_hat (pays cost)
  L1-scram  - pays the SAME awareness cost, but those two channels are decorrelated noise

They compete in one shared world (everything identical except whether the extra senses
carry information). Two environments:
  baseline   - terminal_repro OFF (tau_hat is not actionable)
  actionable - terminal_repro ON  (semelparous terminal investment: reproduce-and-die;
               timing it to imminent death needs tau_hat)

Decisive contrast = (frac_real - frac_scram): isolates the VALUE OF INFORMATION at matched
cost. Expect ~0 in baseline (info useless) and >0 in actionable (info usable) IF death-
awareness is adaptive when the environment rewards acting on it.
"""
import argparse, csv, os
import numpy as np
import eco_mvp as E

RESULTS = "results"


def tail_mean(hist, key, frac=0.3):
    k = max(1, int(len(hist) * frac))
    return float(np.mean([s[key] for _, s in hist[-k:]]))


def run_one(env_terminal, seed, lam, ticks, N, S, decay, brood, hidden=0,
            a_death=0.30, shock=0.0, hz_period=0, hz_amp=0.0):
    cfg = E.Cfg(ticks=ticks, N_slots=N, S=S, seed=seed, lam=lam, decay=decay,
                terminal_brood=brood,
                policy_hidden=hidden, a_death=a_death, shock_prob=shock, shock_mult=4.0,
                hazard_period=hz_period, hazard_amp=hz_amp,
                init_mix="compete", freeze_types=True, terminal_repro=env_terminal)
    w, h = E.run(cfg, record_every=10)
    return dict(
        n=len(w.agents),
        L0=tail_mean(h, "frac_L0"), real=tail_mean(h, "frac_L1_real"),
        scram=tail_mean(h, "frac_L1_scram"),
        term_births=w.terminal_births_total,
        semel=w.deaths_semel, health=w.deaths_health,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--S", type=float, default=140.0)
    ap.add_argument("--decay", type=float, default=0.10)
    ap.add_argument("--brood", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--envs", type=str, default="both", help="both|actionable|baseline")
    ap.add_argument("--a_death", type=float, default=0.30)
    ap.add_argument("--shock", type=float, default=0.0)
    ap.add_argument("--hz_period", type=int, default=0)
    ap.add_argument("--hz_amp", type=float, default=0.0)
    ap.add_argument("--out", type=str, default="eco_c1.csv")
    a = ap.parse_args()

    rows = []
    _envs = [(False, "baseline"), (True, "actionable")]
    if a.envs == "actionable": _envs = [(True, "actionable")]
    elif a.envs == "baseline": _envs = [(False, "baseline")]
    for env, name in _envs:
        diffs = []
        print(f"\n=== {name} (terminal_repro={env})  lam={a.lam} N={a.N} S={a.S} decay={a.decay} brood={a.brood} ===", flush=True)
        for rep in range(a.reps):
            r = run_one(env, a.seed0 + rep, a.lam, a.ticks, a.N, a.S, a.decay, a.brood, a.hidden,
                        a.a_death, a.shock, a.hz_period, a.hz_amp)
            d = r["real"] - r["scram"]
            diffs.append(d)
            rows.append(dict(env=name, seed=a.seed0 + rep, **r, real_minus_scram=d))
            print(f"  seed={a.seed0+rep} n={r['n']:3d}  L0={r['L0']:.3f} real={r['real']:.3f} "
                  f"scram={r['scram']:.3f}  (real-scram)={d:+.3f}  term_b={r['term_births']}",
                  flush=True)
        dm, ds = float(np.mean(diffs)), float(np.std(diffs))
        sem = ds / np.sqrt(len(diffs))
        print(f"  >>> {name}: mean(real-scram) = {dm:+.3f} ± {ds:.3f}  (SEM {sem:.3f}, n={len(diffs)})",
              flush=True)

    with open(os.path.join(RESULTS, a.out), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    print(f"\nwrote {RESULTS}/{a.out}")


if __name__ == "__main__":
    main()
