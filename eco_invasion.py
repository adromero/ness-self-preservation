#!/usr/bin/env python3
"""Invasion assay: measure the SELECTION COEFFICIENT of real death-awareness relative to
cost-matched scrambled awareness, directly from the initial frequency dynamics.

Seed a 50/50 real-vs-scram L1 population (no L0), freeze types, run in the actionable
K-regime, and regress logit(n_real / n_scram) on generation number (cumulative deaths /
N_slots). The slope IS s (selection coefficient per generation). This has far lower variance
than fixation-based readouts, so it can resolve a small effect that the compete experiments
drowned in drift.

NULL control (decisive): global scramble=True makes BOTH labelled types get noise senses ->
they are genuinely identical -> the 'real vs scram' label is a neutral marker -> s should be 0.
The gap (test s) - (null s) is the value of information, with the measurement's own noise/bias
differenced out. Run for the MLP brain (can use the info) and the linear brain (cannot).
"""
import argparse, csv, os
import numpy as np
import eco_mvp as E

RESULTS = "results"


def one_invasion(brain, scramble_all, seed, window, decay, N, S, lam, brood, p0=0.5,
                 a_death=0.30, shock=0.0, terminal=True, hz_period=0, hz_amp=0.0):
    cfg = E.Cfg(ticks=window, N_slots=N, S=S, seed=seed, lam=lam, decay=decay,
                terminal_brood=brood, terminal_repro=terminal, policy_hidden=brain,
                a_death=a_death, shock_prob=shock, shock_mult=4.0,
                hazard_period=hz_period, hazard_amp=hz_amp,
                init_fracs=f"0,{p0},{1 - p0}", freeze_types=True, scramble=scramble_all)
    w = E.World(cfg)
    gens, logits = [], []
    for _ in range(window):
        w.tick()
        A = w.agents
        nr = sum(1 for a in A if a.ell == 1 and not a.scrambled)   # 'real' label
        ns = sum(1 for a in A if a.ell == 1 and a.scrambled)       # 'scram' label
        if nr + ns < 4:                       # too few L1 left to estimate a slope
            break
        gens.append(w.deaths_total / N)
        logits.append(np.log((nr + 0.5) / (ns + 0.5)))
    if len(gens) < 6 or (gens[-1] - gens[0]) < 1.0:
        return None                            # not enough generations to fit
    s = float(np.polyfit(gens, logits, 1)[0])  # selection coefficient per generation
    return s


def run_cell(brain, scramble_all, reps, seed0, **kw):
    ss = []
    for rep in range(reps):
        r = one_invasion(brain, scramble_all, seed0 + rep, **kw)
        if r is not None:
            ss.append(r)
    ss = np.array(ss)
    return ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--decay", type=float, default=0.06)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--S", type=float, default=90.0)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--brood", type=int, default=1)
    ap.add_argument("--a_death", type=float, default=0.30)
    ap.add_argument("--shock", type=float, default=0.0)
    ap.add_argument("--terminal", type=int, default=1)
    ap.add_argument("--hz_period", type=int, default=0)
    ap.add_argument("--hz_amp", type=float, default=0.0)
    ap.add_argument("--out", type=str, default="eco_invasion.csv")
    a = ap.parse_args()
    kw = dict(window=a.window, decay=a.decay, N=a.N, S=a.S, lam=a.lam, brood=a.brood,
              a_death=a.a_death, shock=a.shock, terminal=bool(a.terminal),
              hz_period=a.hz_period, hz_amp=a.hz_amp)

    rows = []
    print(f"INVASION ASSAY — s(real vs scram) per generation | actionable K-regime "
          f"decay={a.decay} lam={a.lam} brood={a.brood} window={a.window} reps={a.reps}\n", flush=True)
    print(f"{'brain':7s} {'condition':16s}  {'mean_s':>8s} {'SEM':>7s} {'t':>6s}  n", flush=True)
    cells = {}
    for brain, bname in [(8, "MLP-8"), (0, "linear")]:
        for scr, cname in [(False, "test(real senses)"), (True, "null(both scram)")]:
            ss = run_cell(brain, scr, a.reps, 6000, **kw)
            n = len(ss); mean = ss.mean() if n else float("nan")
            sem = ss.std() / np.sqrt(n) if n else float("nan")
            t = mean / sem if sem > 0 else 0.0
            cells[(bname, cname)] = ss
            print(f"{bname:7s} {cname:16s}  {mean:+8.4f} {sem:7.4f} {t:+6.2f}  {n}", flush=True)
            for v in ss:
                rows.append(dict(brain=bname, condition=cname, s=v))
    # value-of-information = test - null (Welch)
    print("\nValue of information  s_test - s_null  (per generation):", flush=True)
    for bname in ["MLP-8", "linear"]:
        te = cells[(bname, "test(real senses)")]; nu = cells[(bname, "null(both scram)")]
        diff = te.mean() - nu.mean()
        se = np.sqrt(te.var() / len(te) + nu.var() / len(nu))
        t = diff / se if se > 0 else 0.0
        print(f"  {bname:7s}: Δs = {diff:+.4f}  (SE {se:.4f}, t≈{t:+.2f}, "
              f"n_test={len(te)} n_null={len(nu)})", flush=True)

    with open(os.path.join(RESULTS, a.out), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["brain", "condition", "s"])
        wtr.writeheader(); wtr.writerows(rows)
    print(f"\nwrote {RESULTS}/{a.out}", flush=True)


if __name__ == "__main__":
    main()
