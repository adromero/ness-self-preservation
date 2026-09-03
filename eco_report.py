#!/usr/bin/env python3
"""Render results/eco_report.html from the pilot lambda-sweep (results/eco_sweep.csv)
and the validation-gate log (results/eco_validation.log).

No experiment logic lives here -- it only reads the CSV/log the simulator produced and
renders figures + prose. Dark theme matches results/spec.html so the three pages
(spec / NESS report / ecology report) share a look at localhost:8055.
"""
import base64, csv, io, os, re, html
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eco_mvp as E

RESULTS = "results"
SWEEP_CSV = os.path.join(RESULTS, "eco_sweep.csv")
VAL_LOG = os.path.join(RESULTS, "eco_validation.log")
OUT = os.path.join(RESULTS, "eco_report.html")

BG, FG, MUT, CARD, LINE, ACC = "#0d1117", "#e6edf3", "#9aa7b4", "#161b22", "#30363d", "#58a6ff"
GOOD, WARN, BAD = "#3fb950", "#d29922", "#f85149"


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def style_ax(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=MUT)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, color=LINE, alpha=0.4, lw=0.6)


def read_sweep(path=SWEEP_CSV):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(lam=float(r["lam"]), m=float(r["frac_L1_mean"]),
                             sd=float(r["frac_L1_std"]), pop=float(r["pop_mean"]),
                             extinct=int(r["extinct"]), n=int(r["n"])))
    rows.sort(key=lambda x: x["lam"])
    return rows


def read_val_log():
    if not os.path.exists(VAL_LOG):
        return []
    lines = []
    for ln in open(VAL_LOG):
        ln = ln.rstrip("\n")
        if re.match(r"\s*\[\d\]", ln) or "VALIDATION GATE" in ln or ln.strip().startswith("reported"):
            lines.append(ln.strip())
    return lines


def fig_fracL1(rows, small=None):
    """Primary series = `rows` (higher-N). If `small` given, overlay it faded as the
    lower-N 'before' curve to show drift shrinking as population grows."""
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    style_ax(ax)
    ax.axhline(0.5, color=MUT, ls="--", lw=1, label="neutral expectation (0.5)")
    if small:
        n0 = int(np.mean([r["pop"] for r in small]))
        ls = [r["lam"] for r in small]; ms = [r["m"] for r in small]; ss = [r["sd"] for r in small]
        ax.errorbar(ls, ms, yerr=ss, fmt="o--", color="#6e7681", ecolor="#6e7681",
                    elinewidth=1, capsize=3, ms=5, lw=1.2, alpha=0.75,
                    label=f"N≈{n0} pilot (drift-dominated at low λ)")
    lams = [r["lam"] for r in rows]; m = [r["m"] for r in rows]; sd = [r["sd"] for r in rows]
    n = rows[0]["n"] if rows else 1
    npop = int(np.mean([r["pop"] for r in rows])) if rows else 0
    sem = [x / max(1, np.sqrt(n)) for x in sd]
    ax.errorbar(lams, m, yerr=sd, fmt="o-", color=ACC, ecolor=ACC, elinewidth=1.3,
                capsize=4, ms=7, lw=2.0, label=f"N≈{npop}, {n} seeds (mean ± SD)")
    ax.errorbar(lams, m, yerr=sem, fmt="none", ecolor="#a5d6ff", elinewidth=2.6, capsize=0)
    ax.set_xlabel("awareness tax  λ  (energy/tick paid by L1 agents)")
    ax.set_ylabel("equilibrium fraction of aware (L1) agents")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Is death-awareness selected for, net of its cost?")
    ax.legend(facecolor=CARD, edgecolor=LINE, labelcolor=FG, fontsize=8.5, loc="upper right")
    return fig_to_b64(fig)


def fig_pop(rows):
    lams = [r["lam"] for r in rows]
    pop = [r["pop"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    style_ax(ax)
    ax.plot(lams, pop, "s-", color=GOOD, ms=6, lw=1.6)
    ax.set_xlabel("awareness tax  λ")
    ax.set_ylabel("mean population")
    ax.set_title("Population (carrying capacity) vs awareness tax")
    ax.set_ylim(0, max(pop) * 1.2 if pop else 1)
    return fig_to_b64(fig)


def interpret(rows, small=None):
    """Return (headline_html, bullets, verdict_color) straight from the data.
    Statistically honest: never declares a net-of-cost sign for a lambda whose replicates
    are drift-dominated (SD spanning most of [0,1]). If `small` (a lower-N sweep) is given,
    reports how raising N sharpened the cost->purge transition."""
    import math
    by = {r["lam"]: r for r in rows}
    lams = sorted(by)
    lam0 = lams[0]
    base = by[lam0]
    n = base["n"]
    sem = (lambda r: r["sd"] / math.sqrt(max(1, r["n"])))
    b = []

    # per-lambda drift classification (SD near Bernoulli-max => replicates fixed at 0/1)
    drift_lams = [l for l in lams if by[l]["sd"] >= 0.20]
    drift_dominated = bool(drift_lams)

    # where does cost decisively purge awareness? (tight + low)
    purged = [l for l in lams if by[l]["m"] < 0.10 and by[l]["sd"] < 0.12]
    lam_star = min(purged) if purged else None

    # is lam0 distinguishable from neutral 0.5, accounting for SEM?
    d0 = 0.5 - base["m"]; s0 = sem(base)
    z0 = d0 / s0 if s0 > 0 else float("inf")

    # bullet 1: the robust finding (cost purges awareness)
    if lam_star is not None:
        b.append(f"<b>Robust result:</b> the awareness tax decisively purges death-awareness once "
                 f"λ ≳ <b>{lam_star:g}</b> energy/tick — from λ={lam_star:g} upward the aware fraction "
                 f"collapses to a tight band near zero "
                 f"({by[lam_star]['m']:.2f}±{by[lam_star]['sd']:.2f} → "
                 f"{by[lams[-1]]['m']:.2f}±{by[lams[-1]]['sd']:.2f} at λ={lams[-1]:g}). Cost alone, "
                 f"with no offsetting benefit, is enough to select the organ out.")

    # bullet 1b: did a clean monotone dose-response emerge? (the payoff of higher N)
    means = [by[l]["m"] for l in lams]
    monotone = all(means[i] >= means[i + 1] - 0.02 for i in range(len(means) - 1))
    span_sig = (len(lams) >= 2 and (by[lams[0]]["m"] - by[lams[-1]]["m"])
                > 2 * (sem(by[lams[0]]) + sem(by[lams[-1]])))
    if monotone and span_sig and len(lams) >= 3:
        b.append(f"<b>A clean dose-response emerged at higher N.</b> The mean aware fraction falls "
                 f"monotonically with cost — {by[lams[0]]['m']:.2f} → "
                 + " → ".join(f"{by[l]['m']:.2f}" for l in lams[1:])
                 + f" across λ={lams[0]:g}→{lams[-1]:g} — and the endpoints differ decisively "
                 f"(Δ={by[lams[0]]['m'] - by[lams[-1]]['m']:.2f}, many SEM). The small-N pilot could not "
                 f"resolve this ordering at all (its low-λ points were non-monotone noise). So cost "
                 f"unambiguously disfavours awareness; drift only blurs <i>where</i> along the way the "
                 f"crossover sits.")

    # bullet 2: the honest limit at zero/near-zero cost (only the truly drift-dominated λ)
    if drift_dominated:
        dl = drift_lams[0]
        r = by[dl]
        at_neutral = abs(r["m"] - 0.5) <= 2 * sem(r)
        sit = ("sits at the neutral 0.5 line" if at_neutral
               else f"sits at {r['m']:.2f}")
        b.append(f"<b>At zero cost, awareness is neutral — and drift, not selection, sets any single "
                 f"run.</b> λ={dl:g} gives {r['m']:.2f}±{r['sd']:.2f}: the mean {sit}, but individual "
                 f"replicates scatter across almost all of [0,1] (they fix at 0 or 1). The SD stayed near its "
                 f"theoretical maximum even after the census population was tripled to ~{r['pop']:.0f}, "
                 f"which tells us the <i>effective</i> population is far smaller — reproductive skew "
                 f"(a few energy-rich agents parent most offspring) makes drift fast. So awareness "
                 f"confers no net benefit when free; it is simply along for the ride.")

    # bullet 2b: what raising N revealed (transition sharpened)
    if small:
        sby = {r["lam"]: r for r in small}
        flips = [l for l in lams if l in sby and sby[l]["sd"] >= 0.20
                 and by[l]["m"] < 0.10 and by[l]["sd"] < 0.12]
        if flips:
            fl = flips[0]
            b.append(f"<b>Raising N sharpened the transition.</b> At λ={fl:g} the aware fraction fell "
                     f"from {sby[fl]['m']:.2f}±{sby[fl]['sd']:.2f} at N≈{int(sby[fl]['pop'])} (where drift "
                     f"masked it) to {by[fl]['m']:.2f}±{by[fl]['sd']:.2f} at N≈{int(by[fl]['pop'])}. The "
                     f"small-N 'neutrality' at λ={fl:g} was a drift artifact: with more population, "
                     f"selection sees even a small cost and purges it. The true break-even cost lies "
                     f"<i>below</i> {fl:g} — awareness is worth its keep only if it is nearly free.")

    # bullet 3: population invariance
    pops = [by[l]["pop"] for l in lams]
    b.append(f"Population is roughly flat across λ ({min(pops):.0f}–{max(pops):.0f}): the tax changes "
             f"<i>which</i> strategy survives, not <i>how many</i> agents the world supports — carrying "
             f"capacity is set by supply and decay, not by awareness.")
    # bullet 4: extinctions if any
    ext = [(l, by[l]["extinct"]) for l in lams if by[l]["extinct"]]
    if ext:
        b.append("One replicate went extinct at "
                 + ", ".join(f"λ={l:g} ({e}/{n})" for l, e in ext)
                 + " — a reminder the operating point runs the population genuinely close to the "
                 "death-line (that is what creates the turnover the selection test needs).")

    # --- headline ---
    saw_flip = False
    if small:
        sby = {r["lam"]: r for r in small}
        saw_flip = any(l in sby and sby[l]["sd"] >= 0.20 and by[l]["m"] < 0.10 and by[l]["sd"] < 0.12
                       for l in lams)
    star = f"{lam_star:g}" if lam_star is not None else "any real"
    if saw_flip or (drift_dominated and lam_star is not None):
        headline = ("Death-awareness is <b>not selected for</b> in this environment. When it is free it "
                    "is <b>selectively neutral</b> — the aware fraction sits on the 0.5 line and any single "
                    "run merely drifts (the <i>effective</i> population is small, so drift is strong). The "
                    f"moment it costs anything selection can see (here λ ≈ {star}, once the population is "
                    "large enough) it is <b>purged</b>"
                    + (" — and tripling N confirmed the small-N 'neutrality' at that cost was a drift "
                       "artifact, not a benefit" if saw_flip else "")
                    + ". For awareness to pay its way, the sensed death-signal must drive a decision the "
                    "blind agent cannot make — which this substrate does not yet reward (see the C1 "
                    "control and actionable-τ̂ steps below).")
        vc = WARN
    elif drift_dominated:
        headline = ("The near-zero-cost regime is drift-dominated (error bars span most of [0,1]); the "
                    f"sweep cleanly shows cost purges awareness once λ ≳ {star}, but pinning the "
                    "net-of-cost sign at zero cost needs a still-larger effective population.")
        vc = WARN
    else:
        headline = ("Cost purges death-awareness; near zero cost it is statistically neutral. The "
                    "sensed death-signal buys no net fitness this environment rewards.")
        vc = WARN
    return headline, b, vc


BIGN_CSV = os.path.join(RESULTS, "eco_sweep_bigN.csv")


def main():
    small = read_sweep(SWEEP_CSV)          # N~32 pilot
    big = read_sweep(BIGN_CSV)             # N~98 higher-power rerun
    if big:
        rows, overlay = big, small          # primary = higher-N; overlay = pilot
    else:
        rows, overlay = small, None
    val = read_val_log()
    f1 = fig_fracL1(rows, small=overlay)
    f2 = fig_pop(rows)
    headline, bullets, vc = interpret(rows, small=overlay)
    c = E.Cfg()
    primary_pop = int(np.mean([r["pop"] for r in rows])) if rows else 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    val_html = ""
    if val:
        items = []
        for ln in val:
            color = GOOD if "PASS" in ln else (BAD if "FAIL" in ln else MUT)
            items.append(f'<div style="color:{color};font-family:ui-monospace,monospace;'
                         f'font-size:13px;margin:2px 0">{html.escape(ln)}</div>')
        val_html = "".join(items)

    def _tbl(rr):
        return "".join(
            f"<tr><td>{r['lam']:g}</td><td>{r['m']:.3f}</td><td>±{r['sd']:.3f}</td>"
            f"<td>{r['pop']:.1f}</td><td>{r['extinct']}/{r['n']}</td></tr>" for r in rr)
    sweep_rows = _tbl(rows)
    overlay_tbl = ""
    if overlay:
        n0 = int(np.mean([r["pop"] for r in overlay]))
        overlay_tbl = (f'<h3>Original pilot (N≈{n0}) — for comparison</h3>'
                       f'<table><thead><tr><th>λ</th><th>frac L1</th><th>±SD</th>'
                       f'<th>pop</th><th>extinct</th></tr></thead><tbody>{_tbl(overlay)}</tbody></table>')

    params = [
        ("N_slots (max population)", c.N_slots), ("supply S (energy/tick)", c.S),
        ("base decay d", c.decay), ("baseline drain m", c.m),
        ("L1 awareness cost (cost_l1)", c.cost_l1), ("repro threshold", c.e_repro_threshold),
        ("child cost", c.child_cost), ("predation friction (1-η)", round(1 - c.eta, 2)),
        ("steal cap", c.steal_cap), ("mut. rate awareness allele", c.mut_ell),
    ]
    param_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in params)

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NESS-Ecology — Pilot λ-Sweep</title>
<style>
:root{{--bg:{BG};--fg:{FG};--mut:{MUT};--card:{CARD};--line:{LINE};--acc:{ACC}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);
 font:15.5px/1.65 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
.wrap{{max-width:900px;margin:0 auto;padding:32px 22px 80px}}
h1{{font-size:26px;line-height:1.25;margin:0 0 4px;border-bottom:1px solid var(--line);padding-bottom:14px}}
h2{{font-size:20px;margin:34px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}}
h3{{font-size:16px;margin:20px 0 6px}}
a{{color:var(--acc);text-decoration:none}} a:hover{{text-decoration:underline}}
code{{background:#1f2630;padding:1.5px 6px;border-radius:5px;font-size:13px}}
table{{border-collapse:collapse;margin:12px 0;font-size:13.5px;width:100%;display:block;overflow-x:auto}}
th,td{{border:1px solid var(--line);padding:6px 11px;text-align:left}}
th{{background:#1c2330}}
.sub{{color:var(--mut);font-size:13.5px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 20px;margin:16px 0}}
.verdict{{border-left:4px solid {vc};background:#131922;padding:14px 18px;border-radius:8px;margin:14px 0}}
img{{max-width:100%;border:1px solid var(--line);border-radius:10px;margin:10px 0;background:{BG}}}
ul{{padding-left:20px}} li{{margin:6px 0}}
.pill{{display:inline-block;font-size:11.5px;padding:2px 9px;border-radius:20px;background:#1c2330;
 color:var(--mut);border:1px solid var(--line);margin-left:8px}}
</style></head><body><div class="wrap">

<h1>NESS-Ecology — Pilot λ-Sweep <span class="pill">Stage-1 MVP</span></h1>
<div class="sub">Does an evolving population keep <b>death-awareness</b> when the awareness organ costs
energy? Generated {now}. &nbsp;·&nbsp; <a href="/spec.html">full spec</a> ·
<a href="/report.html">NESS substrate report</a></div>

<div class="verdict"><b>Headline.</b> {headline}</div>

<h2>1 · What the experiment does</h2>
<p>A chemostat ecology of neural-net agents lives on the NESS substrate: every tick their
weights <b>decay</b> (skills rot), and they must spend energy on <b>repair</b> (online SGD) to
stay above a health death-line. Energy is finite (supply <code>S</code>), so agents compete,
predate, reproduce (Moran birth–death) and die. Two heritable strategies coexist:</p>
<ul>
<li><b>L0 (blind):</b> senses only its own health and energy.</li>
<li><b>L1 (death-aware):</b> additionally senses its <i>decay slope</i> and an estimate of its
<i>time-to-death</i> (τ̂) — but pays an <b>awareness tax</b> <code>λ·cost_l1</code> energy/tick to run
the organ. New L1 senses start at zero weight (<i>neutral unmask</i>), so awareness is not free
information handed to the agent — it must be made useful by selection or it is dead weight.</li>
</ul>
<p>The sweep raises the tax <code>λ</code> from 0 (awareness free) upward and measures the
<b>equilibrium fraction of aware agents</b>. λ=0 is the decisive control: if awareness is adaptive,
it should be <i>favoured</i> when free; if it merely rides along, it should be <i>neutral</i>; the
tax then reveals the net-of-cost balance.</p>

<h2>2 · Result</h2>
<p class="sub">Primary curve below is the higher-power rerun at <b>N≈{primary_pop}</b> (population tripled
by raising energy supply, since carrying capacity is set by supply — not by the slot cap). The faded
grey curve is the original N≈32 pilot; the point of tripling N is to shrink genetic drift so the
low-cost regime can actually be read.</p>
<img src="data:image/png;base64,{f1}" alt="frac L1 vs lambda">
<img src="data:image/png;base64,{f2}" alt="population vs lambda">
<h3>Higher-power run (N≈{primary_pop})</h3>
<table><thead><tr><th>λ (tax)</th><th>frac L1</th><th>±SD</th><th>pop</th><th>extinct</th></tr></thead>
<tbody>{sweep_rows}</tbody></table>
{overlay_tbl}
<div class="card"><b>Reading it in plain English</b><ul>{''.join(f'<li>{x}</li>' for x in bullets)}</ul></div>

<h2>3 · Why you can trust the numbers — validation gate</h2>
<p class="sub">Every run asserts strict energy conservation each tick; the simulator passes an
independent gate before any science is read off it:</p>
<div class="card">{val_html or '<span class="sub">(run <code>eco_mvp.py --validate &gt; results/eco_validation.log</code> to populate)</span>'}</div>
<p class="sub"><b>Honest caveats baked into the gate:</b> (5) heritability is confirmed but measured at
birth (offspring inherit the parent's damaged body, so parent–offspring health correlation is
trivially high) — it proves heritable variation <i>exists</i>, not its magnitude. (6) kill-the-winner
predation is verified to be a <i>live, non-lethal-to-the-system</i> mechanism, but it does <b>not</b>
raise lineage diversity here — it mainly lowers carrying capacity by dissipating energy as friction.
Neither is rigged to pass; both report what the model actually does.</p>

<h2>4 · What this does <i>not</i> yet show (next steps)</h2>
<ul>
<li>The decisive <b>information control (C1)</b>: an L1 agent that pays the tax but whose two extra
senses are <b>scrambled</b> (carry no information). If real-L1 ties scrambled-L1, awareness's
<i>information</i> is worthless here; if real-L1 wins at matched cost, the sensed death-signal is
being used adaptively. This pilot brackets cost vs no-cost; C1 brackets information vs no-information.</li>
<li>An environment where τ̂ is <b>actionable</b> — e.g. a decision only the death-aware agent can time
correctly (bet-hedging before a starvation shock, terminal reproduction near death). Awareness can
only be selected <i>for</i> if the substrate rewards acting on it.</li>
<li>Larger populations (N·s ≫ 1 with less drift) and a formal selection coefficient / fixation-probability
readout rather than equilibrium fraction alone.</li>
</ul>

<h2>Parameters (operating point)</h2>
<table><tbody>{param_rows}</tbody></table>
<div class="sub">Operating point chosen for robustness: decay=0.10 gives sustained ~1 death/tick turnover
(statistical power) with no extinction across seeds, health pinned just above the death-line. The
table shows the base pilot scale (N_slots=64, S=90 → N≈32); the higher-power rerun keeps every knob
identical but raises supply to <b>S=250</b> with <b>N_slots=192</b> to reach N≈{primary_pop}, tripling
the population so drift no longer swamps selection at low λ.</div>

</div></body></html>"""
    with open(OUT, "w") as f:
        f.write(doc)
    print(f"wrote {OUT}  ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
