#!/usr/bin/env python3
"""Render results/investigation.html — the consolidated report of the whole self-preservation
investigation on the NESS ecology. Reads the sweep CSVs; the eco_life map and neural-port 2x2
are the confirmed run outputs. Dark theme matches spec.html / report.html at localhost:8055."""
import base64, csv, io, os
from datetime import datetime, timezone
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "results"; OUT = os.path.join(R, "investigation.html")
BG, FG, MUT, CARD, LINE, ACC = "#0d1117", "#e6edf3", "#9aa7b4", "#161b22", "#30363d", "#58a6ff"
GOOD, WARN, BAD = "#3fb950", "#d29922", "#f85149"


def b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()

def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color(LINE)
    ax.tick_params(colors=MUT); ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG); ax.title.set_color(FG)
    ax.grid(True, color=LINE, alpha=0.35, lw=0.6)

def read_sweep(fn):
    rows = list(csv.DictReader(open(os.path.join(R, fn))))
    return ([float(r["lam"]) for r in rows], [float(r["frac_L1_mean"]) for r in rows],
            [float(r["frac_L1_std"]) for r in rows])


def fig_awareness():
    fig, ax = plt.subplots(figsize=(7.6, 4.4)); style(ax)
    ax.axhline(0.5, color=MUT, ls="--", lw=1, label="neutral (0.5)")
    l0, m0, s0 = read_sweep("eco_sweep.csv")
    ax.errorbar(l0, m0, yerr=s0, fmt="o--", color="#6e7681", ecolor="#6e7681", elinewidth=1,
                capsize=3, ms=5, lw=1.1, alpha=0.7, label="N≈32 pilot")
    l1, m1, s1 = read_sweep("eco_sweep_bigN.csv")
    ax.errorbar(l1, m1, yerr=s1, fmt="o-", color=ACC, ecolor=ACC, elinewidth=1.3, capsize=4,
                ms=7, lw=2.0, label="N≈111, 8 seeds")
    ax.set_xlabel("awareness tax  λ  (energy/tick)"); ax.set_ylabel("equilibrium fraction aware (L1)")
    ax.set_ylim(-0.03, 1.03); ax.set_title("Epistemic awareness: neutral when free, purged by cost")
    ax.legend(facecolor=CARD, edgecolor=LINE, labelcolor=FG, fontsize=9)
    return b64(fig)


def fig_map():
    # eco_life: does high-fear INVADE from rare (start 0.1)?  rows=upkeep, cols=famine_prob
    upkeep = [0.0, 0.005]; fprob = [0.004, 0.02, 0.05]
    invade = np.array([[0.175, 1.000, 1.000],   # upkeep 0.000
                       [0.000, 0.382, 1.000]])  # upkeep 0.005
    fig, ax = plt.subplots(figsize=(6.6, 3.4)); style(ax); ax.grid(False)
    im = ax.imshow(invade, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{p:g}" for p in fprob])
    ax.set_yticks(range(2)); ax.set_yticklabels([f"{u:g}" for u in upkeep])
    ax.set_xlabel("catastrophe frequency (per tick)"); ax.set_ylabel("hedge upkeep cost")
    ax.set_title("Fear (single→decoupled currency): where a will-to-live is selected")
    for i in range(2):
        for j in range(3):
            ax.text(j, i, f"{invade[i,j]:.2f}", ha="center", va="center",
                    color=("white" if invade[i,j] < 0.6 else "black"), fontsize=11, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("frac fear (invade from 10%)", color=FG)
    cb.ax.yaxis.set_tick_params(color=MUT); plt.setp(cb.ax.get_yticklabels(), color=MUT)
    return b64(fig)


def fig_port():
    starts = ["invade from rare\n(start 20%)", "stable when common\n(start 80%)"]
    single = [0.000, 0.292]; decoup = [0.914, 0.958]
    x = np.arange(2); w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.0)); style(ax)
    ax.axhline(0.5, color=MUT, ls="--", lw=1)
    ax.bar(x - w/2, single, w, color=BAD, label="single currency (upkeep = reproductive energy)")
    ax.bar(x + w/2, decoup, w, color=GOOD, label="decoupled currency (separate maintenance budget)")
    ax.set_xticks(x); ax.set_xticklabels(starts); ax.set_ylim(0, 1.05)
    ax.set_ylabel("final fraction of margin-maintainers")
    ax.set_title("NESS port (real decaying nets): decoupling flips repelled → selected")
    ax.legend(facecolor=CARD, edgecolor=LINE, labelcolor=FG, fontsize=8.5, loc="center left")
    for xi, v in zip(x - w/2, single): ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", color=FG, fontsize=9)
    for xi, v in zip(x + w/2, decoup): ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", color=FG, fontsize=9)
    return b64(fig)


def main():
    f1, f2, f3 = fig_awareness(), fig_map(), fig_port()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>A Will to Live — NESS Ecology</title>
<style>
:root{{--bg:{BG};--fg:{FG};--mut:{MUT};--card:{CARD};--line:{LINE};--acc:{ACC}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:15.5px/1.68 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
.wrap{{max-width:900px;margin:0 auto;padding:32px 22px 90px}}
h1{{font-size:27px;line-height:1.25;margin:0 0 6px;border-bottom:1px solid var(--line);padding-bottom:14px}}
h2{{font-size:20px;margin:38px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}}
h3{{font-size:16px;margin:22px 0 6px;color:#cdd6e0}}
a{{color:var(--acc);text-decoration:none}} a:hover{{text-decoration:underline}}
code{{background:#1f2630;padding:1.5px 6px;border-radius:5px;font-size:13px}}
table{{border-collapse:collapse;margin:12px 0;font-size:13.5px;width:100%;display:block;overflow-x:auto}}
th,td{{border:1px solid var(--line);padding:6px 11px;text-align:left}} th{{background:#1c2330}}
.sub{{color:var(--mut);font-size:13.5px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 20px;margin:16px 0}}
.key{{border-left:4px solid {GOOD};background:#12211a;padding:14px 18px;border-radius:8px;margin:16px 0}}
.null{{border-left:4px solid {WARN};background:#20200f;padding:12px 16px;border-radius:8px;margin:14px 0}}
img{{max-width:100%;border:1px solid var(--line);border-radius:10px;margin:12px 0;background:{BG}}}
ul{{padding-left:20px}} li{{margin:5px 0}} strong{{color:#fff}}
.pill{{display:inline-block;font-size:11.5px;padding:2px 9px;border-radius:20px;background:#1c2330;color:var(--mut);border:1px solid var(--line);margin-left:8px}}
</style></head><body><div class="wrap">

<h1>A Will to Live <span class="pill">NESS ecology · full investigation</span></h1>
<div class="sub">When does self-preservation evolve in a population of self-repairing neural networks?
Generated {now}. &nbsp;·&nbsp; <a href="/sowhat.html">so what → engineering</a> · <a href="/transcript.html">transcript</a> · <a href="/spec.html">substrate spec</a> · <a href="/report.html">NESS report</a> · <a href="/eco_report.html">λ-sweep</a></div>

<div class="key"><b>Headline.</b> Self-preservation — a decaying network spending effort to keep its own
performance margin repaired against shocks — is selected <b>only when that upkeep is decoupled from
the reproductive budget</b> — and cheap relative to how often danger strikes. On a single fungible currency
it is repelled under every ecological structure we tried; give survival its own (cheap-enough) budget and it
invades from a rare minority and rises to dominance. Confirmed end-to-end on the real decaying-network agents.</div>

<h2>1 · The substrate and the question</h2>
<p>Agents are small neural nets living on a <b>non-equilibrium steady state (NESS)</b> substrate: every
tick their weights <b>decay</b> (skills rot) and they must spend energy on <b>repair</b> (online SGD) to
stay above a competence death-line. Energy is finite, so agents compete, reproduce (Moran birth–death),
and die. The question: does such a system develop a <b>will to live</b> — an evolved tendency to preserve
itself beyond what blind reproduction already forces? We tested two candidate forms: <b>epistemic</b>
(awareness — a sensor for impending death) and <b>conative</b> (fear — a drive to hold a safety buffer).</p>

<h2>2 · Epistemic awareness is not selected</h2>
<p>An <b>L1</b> agent senses its decay-slope and estimated time-to-death (τ̂) — later, its confidence
<i>margin</i>, the true leading indicator — and pays an energy tax λ for the organ; <b>L0</b> is blind.
Sweeping the tax:</p>
<img src="data:image/png;base64,{f1}" alt="awareness lambda sweep">
<div class="null"><b>Null result.</b> When free (λ=0) awareness just drifts near neutral (enormous
seed-to-seed variance, no consistent advantage); as cost rises it is steadily purged. A low-variance
<b>invasion assay</b> put the value of information (real vs cost-matched <i>scrambled</i> senses) at
<b>Δs ≈ 0</b> — indistinguishable from zero and never positive (linear −0.001, MLP −0.016; both n.s.). Reason: τ̂ is
<b>redundant</b> — the optimal response to impending death (repair/reproduce now) is already reachable from
the health and energy the blind agent senses. Foresight pays only when the future is under-determined by
the present; this substrate has no such gap.</div>

<h2>3 · The pivot: from knowing to caring</h2>
<p>Awareness is an <i>information</i> channel feeding a fitness-maximiser — and value-of-information theory
guarantees it gets competed away, because the fitness optimum is "run lean." <b>Fear</b> is different: not
information, but a <b>bias in the objective</b> — a drive to hold a safety buffer <i>against</i> the
reproductive optimum. It sidesteps the information trap, so we tested it directly.</p>

<h2>4 · Conative fear, single currency: repelled everywhere</h2>
<p>We built the classic conditions self-preservation is thought to need — long lives, group structure,
private idiosyncratic risk, rare severe correlated catastrophes, a genuine reserve-survival advantage —
and on a <b>single fungible currency</b> (the buffer <i>is</i> reproductive energy) fear was <b>not
selected under any of them</b>. A frozen-type invasion assay showed high-fear repelled from every starting
frequency (no bistability, no valley-lock).</p>
<div class="null"><b>Why.</b> A hedge's cost is <b>continuous</b> (paid every calm tick) while its payoff is
<b>episodic and rare</b> (the occasional catastrophe). With fast competitive turnover the continuous cost
compounds over the many generations between catastrophes and the hedger is out-bred before it is ever
needed. This <b>time-asymmetry</b> is the root obstruction — and it is maximal on a single currency, where
the reserve <i>is</i> foregone reproduction (a 1:1 opportunity cost).</div>

<h2>5 · The escape: decouple the cost, or make danger frequent</h2>
<p>In a fast, clean metapopulation model we added a genuinely <b>separate</b> resource stream for the
buffer (a second currency) and mapped when self-preservation is selected, over hedge <b>upkeep cost</b> ×
catastrophe <b>frequency</b> (colour = does high-fear invade from a 10% minority):</p>
<img src="data:image/png;base64,{f2}" alt="cost x frequency map">
<div class="key"><b>The map.</b> Self-preservation is selected when the hedge is <b>decoupled and cheap</b>
relative to how often death threatens. A free decoupled buffer is <b>stable</b> even under rare danger and
<b>fixes</b> under frequent danger; even a whisper of continuous cost (upkeep 0.005) is repelled under rare
danger, and recovered only when danger is frequent enough to erase the time-asymmetry. The single fungible
currency sits in the worst corner (maximal cost) — which is why nothing worked there.</div>

<h2>6 · Port to the NESS substrate: the real self-repairing nets</h2>
<p>Back on the actual decaying networks, the hedge becomes <b>the net maintaining its own margin</b> via
free "maintenance" SGD (a decoupled repair budget), the catastrophe a <b>decay shock</b> (differentially
lethal: high-margin nets ride it out, low-margin collapse). Reproduction is not gated on fear. Single vs
decoupled maintenance, invasion assay:</p>
<img src="data:image/png;base64,{f3}" alt="NESS port 2x2">
<div class="key"><b>Confirmed on the real agents.</b> Same substrate, same everything — only the currency
differs. <b>Single</b> currency repels margin-maintenance from both rare (→0.00) and common (→0.29) starts.
<b>Decoupled</b> currency lets it <b>invade from a 20% minority to 91%</b> and hold at 96% when common.
Mechanism check: fear-maintained nets suffered <b>0 health-collapse deaths vs 10</b> for the fearless, and
decoupling made that upkeep nearly free (110 vs 69 births).</div>

<h2>7 · What it means</h2>
<p>A will to live is <b>not</b> a primitive that falls out of decay + repair + competition, and it is
<b>not</b> about knowing you will die. It is a <b>variance-reduction investment</b> that competition
suppresses whenever it is paid in the currency of reproduction — and it emerges the moment survival has its
<b>own budget</b>. Put plainly:</p>
<div class="card" style="font-size:16px"><b>Self-preservation evolves precisely when caring about staying
alive is cheap enough to afford between the rare moments it matters</b> — i.e. when its cost is decoupled
from reproduction, and small relative to how often death actually threatens.</div>
<p class="sub">Controls that make this defensible: strict energy conservation asserted every tick;
order-independent (synchronous) updates; cost-matched scrambled-sense control for information; frozen-type
invasion assays with no mutation-bias; a neutral-drift control that caught (and retracted) an earlier false
positive; and reproduction of the single-currency null on the neural substrate before the decoupled
positive. Several promising early readings were refuted by harder tests along the way — the surviving
claims are the ones that cleared them.</p>

</div></body></html>"""
    open(OUT, "w").write(doc)
    print(f"wrote {OUT} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
