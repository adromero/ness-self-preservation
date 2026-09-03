#!/usr/bin/env python3
"""Regenerate results/report.html from the saved aggregates with corrected
analysis logic — no experiment re-run. (n=2 means mean±std reconstructs the two
replicate values exactly.)"""
import csv, json
from datetime import datetime, timezone

import run_7h as R
import ness_experiment as ne

ss_agg = {}
for row in csv.DictReader(open("results/sweep_agg.csv")):
    ss_agg[(float(row["d"]), int(row["r"]))] = (
        float(row["ss_mean"]), float(row["ss_std"]), int(row["n"]))

sweep_all = {}
for (d, r), (m, s, n) in ss_agg.items():
    sweep_all[(d, r)] = [m - s, m + s] if n == 2 else [m] * max(1, n)

recovery_all = {}
for row in csv.DictReader(open("results/recovery_agg.csv")):
    st = int(row["starve_ticks"]); n = int(row["n"])
    psm, pss = float(row["post_starve_mean"]), float(row["post_starve_std"])
    prm, prs = float(row["post_recovery_mean"]), float(row["post_recovery_std"])
    ps = [psm - pss, psm + pss] if n == 2 else [psm] * max(1, n)
    pr = [prm - prs, prm + prs] if n == 2 else [prm] * max(1, n)
    recovery_all[st] = list(zip(ps, pr))

st = json.load(open("results/status.json"))
healthy_acc, probe, reps, elapsed_h = st["healthy_acc"], tuple(st["probe"]), st["reps"], st["elapsed_h"]

cfg = R.make_cfg(quick=False); cfg.probe_d, cfg.probe_r = probe
an = R.analyze(sweep_all, recovery_all, healthy_acc, cfg, probe)
meta = dict(device=str(ne.DEVICE), reps=reps, elapsed_h=elapsed_h, budget_h=7.0, done=True,
            now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
R.write_report(an, healthy_acc, cfg, probe, meta)
print("rebuilt results/report.html")
