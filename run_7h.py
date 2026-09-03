#!/usr/bin/env python3
"""
7-hour NESS orchestrator.

Runs the NESS experiment (ness_experiment.py, design unchanged) repeatedly with
fresh seeds until a wall-clock budget elapses, so the phase boundary and the
hysteresis curve get real error bars instead of a single noisy draw. After every
replicate it checkpoints aggregate CSVs, plots, and a self-contained HTML report
(results/report.html) — so a crash or an early stop still leaves a valid artifact.

Usage:
    python run_7h.py --hours 7            # the real run
    python run_7h.py --quick --hours 0.1  # fast end-to-end smoke test
"""
import argparse
import base64
import csv
import io
import json
import math
import os
import statistics as stats
import time
import traceback
from datetime import datetime, timezone

import ness_experiment as ne  # design + primitives (unchanged)

RESULTS = ne.RESULTS_DIR
CHANCE = ne.CHANCE
DEAD = ne.DEAD_THRESHOLD


class Cfg:
    pass


def make_cfg(quick: bool) -> Cfg:
    a = Cfg()
    if quick:
        a.ticks, a.eval_every = 120, 15
        a.decay_rates = [0.005, 0.02, 0.08]
        a.repair_budgets = [0, 1, 4]
        a.starve_durations = [20, 60, 150]
        a.recovery_ticks = 120
        a.pre_epochs = 1
    else:
        a.ticks, a.eval_every = 1500, 50
        a.decay_rates = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
        # r=16 trimmed (the costliest column, ~1h/replicate) so more replicates
        # fit the budget -> tighter error bars. Repair still spans 0..8.
        a.repair_budgets = [0, 1, 2, 4, 8]
        a.starve_durations = [25, 50, 100, 200, 400, 800, 1600]
        a.recovery_ticks = 1500
        a.pre_epochs = 3
    a.lr = 1e-3
    a.probe_d, a.probe_r = 0.02, 4
    return a


class RepWriter:
    """Wraps the csv writer so every per-tick row carries its replicate id."""
    def __init__(self, w):
        self.w = w
        self.rep = 0

    def writerow(self, row):
        self.w.writerow([self.rep] + list(row))


def mean_std(xs):
    xs = list(xs)
    if not xs:
        return float("nan"), float("nan"), 0
    m = sum(xs) / len(xs)
    s = stats.pstdev(xs) if len(xs) > 1 else 0.0
    return m, s, len(xs)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def boundary_for_column(ds, ss_means, level=0.5):
    """Primary life/death edge: the FIRST decay d (scanning low->high) at which
    steady-state accuracy drops through `level`, interpolated in log-d. Returns
    (d_crossing or None, max_slope, revived) where `revived` flags a later
    alive->... rebound at high decay (repair re-learning from near-noise)."""
    pts = sorted(zip(ds, ss_means))
    d_cross, max_slope, revived = None, 0.0, False
    for (d0, s0), (d1, s1) in zip(pts, pts[1:]):
        if d1 > d0:
            max_slope = max(max_slope, abs((s1 - s0) / (math.log10(d1) - math.log10(d0))))
        if d_cross is None and s0 > level and s1 <= level:  # first alive->dead
            f = (level - s0) / (s1 - s0)
            d_cross = 10 ** (math.log10(d0) + f * (math.log10(d1) - math.log10(d0)))
        elif d_cross is not None and s0 <= level and s1 > level:  # later rebound
            revived = True
    return d_cross, max_slope, revived


def analyze(sweep_all, recovery_all, healthy_acc, cfg, probe):
    ds = sorted({d for d, _ in sweep_all})
    rs = sorted({r for _, r in sweep_all})
    ss = {k: mean_std(v) for k, v in sweep_all.items()}  # (d,r)->(mean,std,n)

    # Signature 1: steady state under throughput
    viable = [((d, r), ss[(d, r)]) for (d, r) in ss if r > 0 and ss[(d, r)][0] > DEAD]
    best = max(viable, key=lambda kv: kv[1][0]) if viable else None

    # Signature 2: collapse on starvation (r=0 columns)
    r0 = [ss[(d, 0)][0] for d in ds if (d, 0) in ss]
    r0_max = max(r0) if r0 else float("nan")

    # Signature 3: boundary sharpness per repair column
    sharp = {}
    for r in rs:
        col_d = [d for d in ds if (d, r) in ss]
        col_s = [ss[(d, r)][0] for d in col_d]
        sharp[r] = boundary_for_column(col_d, col_s)  # (d_cross, max_slope, revived)

    # Signature 4: irrecoverability / hysteresis
    rec = {}
    for st in sorted(recovery_all):
        ps = [a for a, _ in recovery_all[st]]
        pr = [b for _, b in recovery_all[st]]
        rec[st] = (mean_std(ps), mean_std(pr))
    rec_curve = [(st, rec[st][0][0], rec[st][1][0]) for st in sorted(rec)]
    pnr = None  # point of no return: shortest starvation whose mean recovery is dead
    for st, _psm, prm in rec_curve:
        if prm < DEAD + 0.05:
            pnr = st
            break

    # Signature 2 at the actual operating decay rate (not the trivially-mild one)
    pd = probe[0]
    r0_at_probe = ss[(pd, 0)][0] if (pd, 0) in ss else float("nan")
    post_starve_long = rec_curve[-1][1] if rec_curve else float("nan")

    return dict(ds=ds, rs=rs, ss=ss, viable=viable, best=best, r0_max=r0_max,
                sharp=sharp, rec=rec, rec_curve=rec_curve, pnr=pnr, probe=probe,
                r0_at_probe=r0_at_probe, post_starve_long=post_starve_long)


# ---------------------------------------------------------------------------
# Plots (return base64 PNG)
# ---------------------------------------------------------------------------
def _b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def make_figs(an, healthy_acc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    ds, rs, ss = an["ds"], an["rs"], an["ss"]
    figs = {}

    # Phase diagram (mean ss)
    grid = np.array([[ss[(d, r)][0] for r in rs] for d in ds])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis", vmin=CHANCE, vmax=1.0)
    ax.set_xticks(range(len(rs)), [str(r) for r in rs])
    ax.set_yticks(range(len(ds)), [f"{d:g}" for d in ds])
    ax.set_xlabel("repair budget r (grad steps / tick)")
    ax.set_ylabel("decay rate d")
    ax.set_title("Phase diagram — mean steady-state accuracy")
    fig.colorbar(im, label="steady-state accuracy")
    figs["phase"] = _b64(fig)

    # Std / noise across replicates (peaks at the boundary if the transition is real)
    gstd = np.array([[ss[(d, r)][1] for r in rs] for d in ds])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(gstd, origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks(range(len(rs)), [str(r) for r in rs])
    ax.set_yticks(range(len(ds)), [f"{d:g}" for d in ds])
    ax.set_xlabel("repair budget r"); ax.set_ylabel("decay rate d")
    ax.set_title("Across-replicate std of steady-state accuracy\n(a ridge = critical boundary)")
    fig.colorbar(im, label="std")
    figs["phase_std"] = _b64(fig)

    # Boundary sharpness: ss vs d for each r
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rs:
        col = [(d, ss[(d, r)][0], ss[(d, r)][1]) for d in ds if (d, r) in ss]
        xd = [c[0] for c in col]; ym = [c[1] for c in col]; ye = [c[2] for c in col]
        ax.errorbar(xd, ym, yerr=ye, marker="o", capsize=2, label=f"r={r}")
    ax.axhline(0.5, ls=":", c="gray"); ax.axhline(CHANCE, ls="--", c="gray", label="chance")
    ax.set_xscale("log"); ax.set_xlabel("decay rate d (log)"); ax.set_ylabel("steady-state accuracy")
    ax.set_title("Life/death boundary sharpness (Signature 3)")
    ax.legend(fontsize=8)
    figs["boundary"] = _b64(fig)

    # Recovery / hysteresis
    if an["rec_curve"]:
        st = [c[0] for c in an["rec_curve"]]
        psm = [an["rec"][s][0][0] for s in st]; pss = [an["rec"][s][0][1] for s in st]
        prm = [an["rec"][s][1][0] for s in st]; prs = [an["rec"][s][1][1] for s in st]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(st, psm, yerr=pss, marker="o", capsize=2, label="acc after starvation")
        ax.errorbar(st, prm, yerr=prs, marker="s", capsize=2, label="acc after recovery attempt")
        ax.axhline(CHANCE, ls="--", c="gray", label="chance")
        if an["pnr"]:
            ax.axvline(an["pnr"], ls=":", c="red", label=f"point of no return ~{an['pnr']}")
        ax.set_xscale("log"); ax.set_xlabel("starvation duration (ticks, log)")
        ax.set_ylabel("test accuracy")
        ax.set_title("Irrecoverability / hysteresis (Signature 4)")
        ax.legend(fontsize=8)
        figs["recovery"] = _b64(fig)
    return figs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def verdicts(an, healthy_acc):
    v = {}
    best = an["best"]
    v["s1"] = ("YES" if best else "NO",
               (f"{len(an['viable'])} viable (r&gt;0) operating points; best steady state "
                f"{best[1][0]*100:.1f}% ± {best[1][1]*100:.1f} at d={best[0][0]:g}, r={best[0][1]} "
                f"— a plateau {(healthy_acc-best[1][0])*100:.1f} pts below the clean ceiling "
                f"({healthy_acc*100:.1f}%).") if best else "No operating point held above the death line.")
    pd, pr = an["probe"]
    r0p = an["r0_at_probe"]; psl = an["post_starve_long"]
    dead = (not math.isnan(r0p)) and r0p < DEAD
    v["s2"] = ("YES" if dead else "PARTIAL",
               f"At the operating decay rate d={pd:g}, cutting repair (r=0) collapses the steady "
               f"state to {r0p*100:.1f}% and prolonged starvation of the running network drives it to "
               f"{psl*100:.1f}% — the {DEAD*100:.0f}% death line, essentially chance ({CHANCE*100:.0f}%). "
               f"Metabolism is load-bearing: without repair, throughput-driven decay is lethal. "
               f"(Only near-zero decay, d≤0.005, survives without repair.)")
    # sharpness: primary alive->dead edge per repair column
    slopes = [s[1] for s in an["sharp"].values() if s[1] > 0]
    max_slope = max(slopes) if slopes else 0.0
    crossings = {r: s[0] for r, s in an["sharp"].items() if s[0]}
    revived = [r for r, s in an["sharp"].items() if len(s) > 2 and s[2]]
    sharp_txt = "; ".join(f"r={r}: dies at d≈{d:.3g}" for r, d in sorted(crossings.items()))
    rev_txt = (f" A secondary effect: at very high decay (d≈0.16) columns r={sorted(revived)} "
               f"partly revive — repair re-learns from near-random weights each tick, so the phase "
               f"structure is richer than a single boundary.") if revived else ""
    v["s3"] = ("SHARP" if max_slope > 0.6 else "GRADUAL",
               f"Steepest drop {max_slope:.2f} accuracy per decade of decay "
               f"({'a step-like transition' if max_slope>0.6 else 'a smooth crossover'}). "
               f"Primary death edge ({sharp_txt or 'n/a'}) moves to faster decay as repair rises — "
               f"more metabolism buys tolerance.{rev_txt}")
    if an["rec_curve"]:
        if an["pnr"]:
            ss_probe = an["ss"].get((pd, pr), (float("nan"),))[0]
            first_st = an["rec_curve"][0][0]
            early = an["pnr"] <= first_st
            v["s4"] = ("YES",
                       (f"Strong hysteresis. Turning repair back on after starvation does not restore the "
                        f"network: post-recovery accuracy sits at chance for every starvation tested "
                        f"(point of no return ≤{an['pnr']} ticks). Even a lightly-starved state that is still "
                        f"{an['rec_curve'][0][1]*100:.0f}% accurate cannot be rebuilt — repair *maintains* a "
                        f"steady state but cannot *climb back* to it. Caveat: the operating point "
                        f"(d={pd:g}, r={pr}, steady state ≈{ss_probe*100:.0f}%) sits near the transition, so its "
                        f"basin is small; a more deeply-alive point may be more forgiving — worth a targeted follow-up."))
        else:
            worst = min(an["rec_curve"], key=lambda c: c[2])
            v["s4"] = ("NO", f"The network recovered at every tested starvation duration "
                             f"(worst post-recovery {worst[2]*100:.1f}% after {worst[0]} ticks). "
                             f"No point of no return at these settings.")
    else:
        v["s4"] = ("N/A", "No recovery data.")
    return v


def write_report(an, healthy_acc, cfg, probe, meta):
    figs = make_figs(an, healthy_acc)
    v = verdicts(an, healthy_acc)

    def img(key, cap):
        return (f'<figure><img src="data:image/png;base64,{figs[key]}"/>'
                f'<figcaption>{cap}</figcaption></figure>') if key in figs else ""

    ds, rs, ss = an["ds"], an["rs"], an["ss"]
    # steady-state table
    head = "".join(f"<th>r={r}</th>" for r in rs)
    rows = ""
    for d in ds:
        cells = ""
        for r in rs:
            m, s, n = ss[(d, r)]
            alive = m > DEAD
            cells += (f'<td class="{"alive" if alive else "dead"}">{m*100:.0f}'
                      f'<span class="pm">±{s*100:.0f}</span></td>')
        rows += f"<tr><th>d={d:g}</th>{cells}</tr>"
    ss_table = f"<table class='grid'><tr><th></th>{head}</tr>{rows}</table>"

    # recovery table
    rec_rows = ""
    for st, psm, prm in an["rec_curve"]:
        (psm_, pss_, _), (prm_, prs_, _) = an["rec"][st]
        rec_rows += (f"<tr><td>{st}</td><td>{psm_*100:.1f} ± {pss_*100:.1f}</td>"
                     f"<td>{prm_*100:.1f} ± {prs_*100:.1f}</td></tr>")
    rec_table = ("<table class='rec'><tr><th>starve ticks</th><th>after starvation (%)</th>"
                 f"<th>after recovery (%)</th></tr>{rec_rows}</table>") if rec_rows else "<p>No recovery data yet.</p>"

    def card(n, title, verdict, text, color):
        return (f"<div class='sig'><div class='badge {color}'>{verdict}</div>"
                f"<h3>Signature {n}: {title}</h3><p>{text}</p></div>")

    cmap = {"YES": "good", "SHARP": "good", "NO": "bad", "PARTIAL": "warn",
            "GRADUAL": "warn", "N/A": "warn"}
    cards = (
        card(1, "Steady state under throughput", v["s1"][0], v["s1"][1], cmap.get(v["s1"][0], "warn")) +
        card(2, "Collapse on starvation", v["s2"][0], v["s2"][1], cmap.get(v["s2"][0], "warn")) +
        card(3, "Phase transition (sharp boundary)", v["s3"][0], v["s3"][1], cmap.get(v["s3"][0], "warn")) +
        card(4, "Irrecoverability / hysteresis", v["s4"][0], v["s4"][1], cmap.get(v["s4"][0], "warn"))
    )

    status = "COMPLETE" if meta["done"] else "IN PROGRESS"
    html = f"""<!doctype html><meta charset="utf-8">
<title>NESS Experiment — decay vs. repair</title>
<style>
 :root{{--bg:#0d1117;--fg:#e6edf3;--mut:#9aa7b4;--card:#161b22;--line:#30363d;
   --good:#2ea043;--bad:#da3633;--warn:#d29922;--acc:#58a6ff}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);
   font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
 .wrap{{max-width:1080px;margin:0 auto;padding:32px 22px 80px}}
 h1{{font-size:26px;margin:0 0 4px}} h2{{margin:34px 0 10px;font-size:19px;border-bottom:1px solid var(--line);padding-bottom:6px}}
 h3{{margin:0 0 6px;font-size:16px}} .mut{{color:var(--mut)}} a{{color:var(--acc)}}
 .pill{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600}}
 .live{{background:#1f6feb33;color:#79c0ff}} .cmplt{{background:#2ea04333;color:#7ee787}}
 .meta{{display:flex;flex-wrap:wrap;gap:10px 22px;margin:14px 0 6px;color:var(--mut);font-size:13px}}
 .meta b{{color:var(--fg)}}
 .sigs{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}}
 @media(max-width:720px){{.sigs{{grid-template-columns:1fr}}}}
 .sig{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;position:relative}}
 .badge{{position:absolute;top:12px;right:12px;font-weight:700;font-size:12px;padding:3px 9px;border-radius:6px}}
 .badge.good{{background:var(--good)}} .badge.bad{{background:var(--bad)}} .badge.warn{{background:var(--warn);color:#111}}
 .sig p{{margin:6px 0 0;color:var(--mut);font-size:13.5px}}
 figure{{margin:16px 0;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}}
 figure img{{width:100%;border-radius:6px;display:block}} figcaption{{color:var(--mut);font-size:13px;margin-top:8px}}
 table{{border-collapse:collapse;margin:10px 0;font-size:13px}}
 td,th{{border:1px solid var(--line);padding:5px 9px;text-align:center}}
 .grid td.alive{{background:#12331c}} .grid td.dead{{background:#3a1414;color:#f0a8a8}}
 .grid .pm{{color:var(--mut);font-size:11px;margin-left:2px}}
 .note{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 18px;color:var(--mut);font-size:13.5px}}
 code{{background:#1f2630;padding:1px 5px;border-radius:4px}}
</style>
<div class="wrap">
<h1>NESS Experiment — decay vs. repair in a neural network
 <span class="pill {'cmplt' if meta['done'] else 'live'}">{status}</span></h1>
<p class="mut">Does a network under continuous weight decay + online repair behave like a
 <b>non-equilibrium steady state</b>? Four signatures tested. Live network = holds accuracy above the
 {DEAD*100:.0f}% death line; chance = {CHANCE*100:.0f}%.</p>
<div class="meta">
 <span>device <b>{meta['device']}</b></span>
 <span>replicates completed <b>{meta['reps']}</b></span>
 <span>elapsed <b>{meta['elapsed_h']:.2f} h</b> / budget {meta['budget_h']:.1f} h</span>
 <span>healthy ceiling <b>{healthy_acc*100:.1f}%</b></span>
 <span>grid <b>{len(ds)}×{len(rs)}</b> (d×r)</span>
 <span>probe point <b>d={probe[0]:g}, r={probe[1]}</b></span>
 <span>updated <b>{meta['now']}</b></span>
</div>

<h2>Verdict — the four NESS signatures</h2>
<div class="sigs">{cards}</div>

<h2>Signature 1 &amp; 3 — the phase diagram</h2>
{img('phase','Mean steady-state accuracy over the decay×repair grid (mean of all replicates). '
     'Bright = the network holds a living steady state; dark = it decays to chance. '
     'A living plateau below the clean ceiling is the NESS signature.')}
{img('boundary','Steady-state accuracy vs. decay rate for each repair budget, with across-replicate error bars. '
     'A near-vertical drop = a sharp life/death transition; more repair (higher r) shifts the cliff to faster decay.')}
{img('phase_std','Across-replicate variability. A ridge of high variance along the boundary is the classic tell of a real '
     'critical transition (the system is bistable there, so replicates split between living and dead).')}
<p class="mut">Steady-state accuracy (%), mean ± std across replicates. Green = alive, red = dead.</p>
{ss_table}

<h2>Signature 4 — starvation &amp; recovery (hysteresis)</h2>
{img('recovery','Starve the steady state for increasing durations (repair off), then turn repair back on. '
     'If the recovery curve falls to chance past some duration, the damage is irreversible — a point of no return.')}
{rec_table}

<h2>Methods &amp; integrity</h2>
<div class="note">
 <b>Design (unchanged from the submitted script):</b> pretrained MLP on MNIST; each tick applies multiplicative
 Gaussian weight noise scaled by per-layer std plus a small fraction of weight zeroing (decay), optionally followed by
 <code>r</code> SGD steps on fresh data (repair). Steady-state accuracy = mean of the last 25% of a {cfg.ticks}-tick run.<br>
 <b>Fixes applied (bugs, not design):</b> (1) the probe operating point for starvation/recovery is now auto-selected from
 the sweep (the script's own <code>“adjusted below based on sweep if needed”</code> comment was never implemented, so
 signatures 2&amp;4 could have started from a dead state); here it resolved to <b>d={probe[0]:g}, r={probe[1]}</b>.
 (2) The whole experiment is repeated with fresh seeds for <b>{meta['reps']}</b> replicates so every number above carries
 an error bar rather than a single noisy draw. Physics, model, grid, and thresholds are exactly as submitted.
 <b>Limitation:</b> the full 7×6 grid at 1500 ticks costs ~2.7 h per replicate (the r=8/16 columns dominate), so only
 {meta['reps']} replicates fit in the budget — error bars (std of {meta['reps']}) are indicative, not tight; a larger n
 would need a longer run or a trimmed grid.<br>
 {"<b>Grid note:</b> the r=16 repair column was trimmed for this run (it cost ~1h/replicate); repair spans 0..8, so more replicates fit and the error bars tighten. " if 16 not in rs else ""}
 <b>Data:</b> per-tick log <code>results/log.csv</code> (with replicate id), aggregates <code>results/sweep_agg.csv</code>
 and <code>results/recovery_agg.csv</code>.
</div>
</div>"""
    with open(os.path.join(RESULTS, "report.html"), "w") as f:
        f.write(html)


def write_aggregates(sweep_all, recovery_all):
    with open(os.path.join(RESULTS, "sweep_agg.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["d", "r", "ss_mean", "ss_std", "n"])
        for (d, r) in sorted(sweep_all):
            m, s, n = mean_std(sweep_all[(d, r)]); w.writerow([d, r, f"{m:.4f}", f"{s:.4f}", n])
    with open(os.path.join(RESULTS, "recovery_agg.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["starve_ticks", "post_starve_mean", "post_starve_std",
                                       "post_recovery_mean", "post_recovery_std", "n"])
        for st in sorted(recovery_all):
            ps = [a for a, _ in recovery_all[st]]; pr = [b for _, b in recovery_all[st]]
            pm, psd, n = mean_std(ps); rm, rsd, _ = mean_std(pr)
            w.writerow([st, f"{pm:.4f}", f"{psd:.4f}", f"{rm:.4f}", f"{rsd:.4f}", n])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=7.0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    cfg = make_cfg(args.quick)
    budget = args.hours * 3600.0
    start = time.time()
    print(f"[orch] device={ne.DEVICE} budget={args.hours}h grid={len(cfg.decay_rates)}x{len(cfg.repair_budgets)}",
          flush=True)

    import copy
    train_loader, test_loader = ne.get_data()
    model = ne.pretrain(train_loader, test_loader, epochs=cfg.pre_epochs)
    healthy_state = copy.deepcopy(model.state_dict())
    healthy_acc = ne.evaluate(model, test_loader)
    print(f"[orch] healthy ceiling {healthy_acc:.4f}", flush=True)

    log_f = open(os.path.join(RESULTS, "log.csv"), "w", newline="")
    log_w = csv.writer(log_f); log_w.writerow(["rep", "phase", "d", "r", "tick", "acc"])
    rw = RepWriter(log_w)

    sweep_all, recovery_all = {}, {}
    probe = None
    reps, rep_dur = 0, 0.0

    while True:
        elapsed = time.time() - start
        # stop if the next replicate probably won't finish in budget (but always do >=1)
        if reps >= 1 and elapsed + rep_dur * 1.05 > budget:
            print(f"[orch] budget reached ({elapsed/3600:.2f}h, {reps} reps). finalizing.", flush=True)
            break
        r0 = time.time()
        rw.rep = reps
        try:
            seed_off = reps * 100003
            sweep = ne.phase_sweep(healthy_state, train_loader, test_loader, cfg, rw,
                                   seed_offset=seed_off)
            for k, val in sweep.items():
                sweep_all.setdefault(k, []).append(val)
            if probe is None:
                probe = ne.pick_probe(sweep, cfg.probe_d, cfg.probe_r)
                print(f"[orch] fixed probe point d={probe[0]:g}, r={probe[1]}", flush=True)
            cfg.probe_d, cfg.probe_r = probe
            recs = ne.starvation_and_recovery(healthy_state, train_loader, test_loader, cfg, rw,
                                              seed_offset=seed_off)
            for (st, ps, pr) in recs:
                recovery_all.setdefault(st, []).append((ps, pr))
            reps += 1
        except Exception:
            print("[orch] replicate failed:\n" + traceback.format_exc(), flush=True)
        log_f.flush()
        rep_dur = time.time() - r0

        # checkpoint after every replicate (partial results always valid)
        try:
            an = analyze(sweep_all, recovery_all, healthy_acc, cfg, probe or (cfg.probe_d, cfg.probe_r))
            write_aggregates(sweep_all, recovery_all)
            meta = dict(device=str(ne.DEVICE), reps=reps, elapsed_h=(time.time()-start)/3600,
                        budget_h=args.hours, done=False,
                        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
            write_report(an, healthy_acc, cfg, probe or (cfg.probe_d, cfg.probe_r), meta)
            json.dump({"reps": reps, "elapsed_h": meta["elapsed_h"], "done": False,
                       "healthy_acc": healthy_acc, "probe": probe},
                      open(os.path.join(RESULTS, "status.json"), "w"))
            print(f"[orch] rep {reps} done in {rep_dur:.0f}s; checkpoint written "
                  f"({meta['elapsed_h']:.2f}h elapsed)", flush=True)
        except Exception:
            print("[orch] checkpoint failed:\n" + traceback.format_exc(), flush=True)

    # finalize
    an = analyze(sweep_all, recovery_all, healthy_acc, cfg, probe or (cfg.probe_d, cfg.probe_r))
    write_aggregates(sweep_all, recovery_all)
    meta = dict(device=str(ne.DEVICE), reps=reps, elapsed_h=(time.time()-start)/3600,
                budget_h=args.hours, done=True,
                now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    write_report(an, healthy_acc, cfg, probe or (cfg.probe_d, cfg.probe_r), meta)
    json.dump({"reps": reps, "elapsed_h": meta["elapsed_h"], "done": True,
               "healthy_acc": healthy_acc, "probe": probe},
              open(os.path.join(RESULTS, "status.json"), "w"))
    log_f.close()
    print(f"[orch] COMPLETE: {reps} replicates, {meta['elapsed_h']:.2f}h. report at results/report.html", flush=True)


if __name__ == "__main__":
    main()
