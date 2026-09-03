#!/usr/bin/env python3
"""Mechanism test (fast, decisive for information VALUE, no evolution needed).

Give real-L1 and scrambled-L1 the SAME hand-designed policy 'reproduce (semelparate)
when tau_hat says you are about to die', freeze the policy (sigma_w=0), and let them
compete head-to-head under terminal investment. The ONLY difference is whether tau_hat
carries real information. If timing matters, real out-reproduces scrambled.

Run across an r-selected (high decay) and K-selected (low decay) regime.
"""
import numpy as np
import eco_mvp as E

# rows [repair, reproduce, predate, rest] x cols [h_f, e_f, slope_f, tau_f, bias]
HAND = np.array([
    [0, 0, 0, 0.0,  1.0],   # repair: default action
    [0, 0, 0, 5.0, -1.0],   # reproduce/semelparate: fires when tau_f high (near death)
    [0, 0, 0, 0.0, -3.0],   # predate: rare
    [0, 0, 0, 0.0, -3.0],   # rest: rare
], dtype=np.float64)


def mech_test(decay, seed, ticks=250, N=64, S=90):
    cfg = E.Cfg(ticks=ticks, N_slots=N, S=S, seed=seed, lam=0.5, decay=decay,
                init_ell=1, init_scramble_frac=0.5, freeze_types=True,
                terminal_repro=True, terminal_brood=1, sigma_w=0.0)
    w = E.World(cfg)
    for a in w.agents:
        a.W = HAND.copy()
        a.h = E.eval_health(a.model, w.Xpr, w.Ypr)
    for _ in range(ticks):
        w.tick()
        if not w.agents:
            break
    A = w.agents; n = len(A)
    real = sum(1 for a in A if not a.scrambled) / n if n else 0.0
    scram = sum(1 for a in A if a.scrambled) / n if n else 0.0
    return n, real, scram, w.terminal_births_total


if __name__ == "__main__":
    print("HAND-POLICY mechanism test — real vs scram, fixed 'semelparate-when-dying' policy",
          flush=True)
    print("decay  seed   n   real   scram  (real-scram)  term_b", flush=True)
    for decay in [0.10, 0.06]:
        diffs = []
        for seed in [1, 2, 3, 4]:
            n, re, sc, tb = mech_test(decay, seed)
            diffs.append(re - sc)
            print(f"{decay:.2f}   {seed}    {n:3d}  {re:.3f}  {sc:.3f}   {re-sc:+.3f}      {tb}",
                  flush=True)
        print(f"  >>> decay={decay:.2f}: mean(real-scram)={np.mean(diffs):+.3f} "
              f"± {np.std(diffs):.3f} (n={len(diffs)})", flush=True)
