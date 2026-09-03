#!/usr/bin/env python3
"""Render results/sowhat.html — the "so what" companion to investigation.html.

Translates the NESS self-preservation investigation into (1) four portable engineering
principles, (2) a concrete reference implementation (ness_monitor.py) with its measured demo,
(3) one grounded-but-speculative alignment corollary, (4) what NOT to build, and (5) a practical
answer to "what kind of ML model is this, and can I run it on the 3080 / 5090?". Dark theme
matches investigation.html / spec.html at localhost:8055.
"""
import base64, os
from datetime import datetime, timezone

R = "results"; OUT = os.path.join(R, "sowhat.html")
BG, FG, MUT, CARD, LINE, ACC = "#0d1117", "#e6edf3", "#9aa7b4", "#161b22", "#30363d", "#58a6ff"
GOOD, WARN, BAD = "#3fb950", "#d29922", "#f85149"


def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    trace_png = img_b64(os.path.join(R, "monitor_trace.png"))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>So What — Building a Self-Maintaining Model</title>
<style>
:root{{--bg:{BG};--fg:{FG};--mut:{MUT};--card:{CARD};--line:{LINE};--acc:{ACC}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:15.5px/1.68 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
.wrap{{max-width:900px;margin:0 auto;padding:32px 22px 90px}}
h1{{font-size:27px;line-height:1.25;margin:0 0 6px;border-bottom:1px solid var(--line);padding-bottom:14px}}
h2{{font-size:20px;margin:40px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}}
h3{{font-size:16px;margin:22px 0 6px;color:#cdd6e0}}
a{{color:var(--acc);text-decoration:none}} a:hover{{text-decoration:underline}}
code{{background:#1f2630;padding:1.5px 6px;border-radius:5px;font-size:13px}}
pre{{background:#11161d;border:1px solid var(--line);border-radius:10px;padding:14px 16px;overflow-x:auto;font-size:12.7px;line-height:1.5}}
table{{border-collapse:collapse;margin:12px 0;font-size:13.5px;width:100%;display:block;overflow-x:auto}}
th,td{{border:1px solid var(--line);padding:6px 11px;text-align:left}} th{{background:#1c2330}}
.sub{{color:var(--mut);font-size:13.5px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 20px;margin:16px 0}}
.key{{border-left:4px solid {GOOD};background:#12211a;padding:14px 18px;border-radius:8px;margin:16px 0}}
.null{{border-left:4px solid {WARN};background:#20200f;padding:12px 16px;border-radius:8px;margin:14px 0}}
.spec{{border-left:4px solid {ACC};background:#0f1a2b;padding:12px 16px;border-radius:8px;margin:14px 0}}
img{{max-width:100%;border:1px solid var(--line);border-radius:10px;margin:12px 0;background:{BG}}}
ul{{padding-left:20px}} li{{margin:5px 0}} strong{{color:#fff}}
.pill{{display:inline-block;font-size:11.5px;padding:2px 9px;border-radius:20px;background:#1c2330;color:var(--mut);border:1px solid var(--line);margin-left:8px}}
.num{{font-variant-numeric:tabular-nums}}
.g{{color:{GOOD}}} .b{{color:{BAD}}} .w{{color:{WARN}}}
</style></head><body><div class="wrap">

<h1>So What <span class="pill">from a will-to-live to a self-maintaining model</span></h1>
<div class="sub">The engineering payload of the self-preservation investigation. Generated {now}.
&nbsp;·&nbsp; <a href="/investigation.html">full investigation</a> · <a href="/transcript.html">transcript</a> · <a href="/spec.html">substrate spec</a> · <a href="/report.html">NESS report</a></div>

<div class="key"><b>The one-paragraph version.</b> The investigation was about whether a will-to-live
<i>evolves</i>. The reusable part is smaller and more certain: a decaying model held in a
non-equilibrium steady state gives you a <b>leading indicator</b> (the confidence margin) that
degrades long before accuracy does, and a <b>cheap proactive-repair loop</b> that keeps the model
robust to the shock it can't predict. Four principles fall out. Three are solid engineering; one is
a grounded-but-speculative alignment lever; and the investigation also tells you clearly what
<i>not</i> to build. A working reference monitor (<code>ness_monitor.py</code>) is included and
measured below, and it all runs on a single consumer GPU.</div>

<h2>1 · Four principles</h2>

<h3><span class="g">P1 — Monitor the margin, not the accuracy.</span> <span class="pill">solid</span></h3>
<p>Accuracy is a <b>lagging, cliff-shaped</b> signal: on a well-separated task, continuous weight
decay shrinks the logits and compresses the confidence margin (top-1 minus top-2) smoothly for
<i>hundreds</i> of ticks while accuracy sits flat — and then falls off a cliff. In the diagnostic,
the margin compressed <span class="num">~37×</span> (from <span class="num">5.6</span> to
<span class="num">0.15</span>) over ~300 ticks while accuracy held at <span class="num">≈0.92</span>;
only once the margin was nearly exhausted did accuracy finally slip (to <span class="num">≈0.88</span>
as the margin reached 0.05). By the time accuracy moves, the damage is already done. The margin
is the <b>early-warning gauge</b>; watching it is nearly free (one forward pass on a small probe set).</p>

<h3><span class="g">P2 — Repair proactively on a margin trigger, with hysteresis.</span> <span class="pill">solid</span></h3>
<p>Damage you can't predict arrives as <b>shocks</b>. A shock landing on a <b>fat</b> margin is
absorbed; the <i>same</i> shock on a <b>thin</b> (eroded) margin is catastrophic — in the mechanism
test one perturbation cost <span class="num">−0.11</span> accuracy at margin 5.7 but
<span class="num">−0.48</span> at margin 1.4, a ~4× swing. So maintaining the margin is
<i>causally</i> protective. A monitor that tops the model up the moment the margin dips below a
band — and stops once it clears a higher release band (the <b>hysteresis</b>) — keeps the model
out of the fragile zone. Prevention beats recovery on the worst case, always; whether it also wins
on total compute depends on how severe the shocks are (§3).</p>

<h3><span class="g">P3 — Give maintenance its own budget (decouple it from capability).</span> <span class="pill">solid · with a corollary</span></h3>
<p>The investigation's sharpest structural result: when the effort a model spends keeping itself
repaired is drawn from the <i>same</i> pool as its productive/"reproductive" work, self-maintenance
is out-competed under every ecological structure tried. Give maintenance a <b>separate budget</b> and
it flips from repelled to selected — confirmed end-to-end on real decaying nets: single-currency
maintenance is <span class="b">repelled</span> (→0.00 from a rare start, and slips to 0.29 even from an
80% majority), while the <span class="g">decoupled</span> version invades from a 20% minority to
<span class="g">0.91</span> and holds at <span class="g">0.96</span>. The
engineering reading: don't make the repair loop compete with serving for the same resources — give
it a protected, decoupled allocation (its own compute slice, or its own parameters; see the
LoRA-adapter recipe in §5). The alignment reading is in §4.</p>

<h3><span class="g">P4 — Structure a fleet against correlated shocks.</span> <span class="pill">solid</span></h3>
<p>What made self-maintenance viable at all in the model was <b>group structure with idiosyncratic
risk</b>: many semi-isolated demes, local (not global) catastrophes, limited migration, and
per-individual income noise. Translated: if you run <b>many</b> instances of a model, don't let a
single event (a bad data push, an upstream schema change, a shared cache poisoning) hit them all
identically. Keep per-instance variation and stagger maintenance so one correlated shock can't wipe
the whole fleet at once. Robustness lived in the <i>population structure</i>, not in any one agent.</p>

<h2>2 · The reference implementation</h2>
<p><code>ness_monitor.py</code> is a ~300-line, dependency-light watchdog (<code>torch</code> only; <code>matplotlib</code> just for the optional figure) you
wrap around any small <code>nn.Module</code>. It implements P1+P2 directly: each deployment tick it
measures the served margin on a fixed probe set, and if the margin is below the band it runs bounded
repair steps (online SGD on a trickle of fresh labelled data) until the margin clears the release
band. Three policies let you see the contrast on an <i>identical</i> decay schedule:</p>
<ul>
<li><code>off</code> — never repairs (shows the substrate really is decaying).</li>
<li><code>reactive_acc</code> — the naive baseline: repair only once <i>accuracy</i> has cracked. Chases the cliff.</li>
<li><code>proactive_margin</code> — repair on the <i>margin</i> trigger, with hysteresis. The recommended pattern.</li>
</ul>
<p>No network, no API keys, no downloads: the "real model" is a genuine MLP classifier on a locally
generated Gaussian-mixture task, "deployment decay" is continuous weight decay plus occasional
additive weight shocks. Runs in a few seconds on the 3080.</p>

<div class="card"><b>Measured, representative run</b> (600 ticks, seed 0, RTX 3080) — the numbers the
demo prints:
<table>
<tr><th>policy</th><th>mean acc</th><th>worst-case acc</th><th>downtime (&lt;0.85)</th><th>repair steps (cost)</th></tr>
<tr><td><code>off</code></td><td class="num">0.27</td><td class="num b">0.08</td><td class="num b">94.3%</td><td class="num">0</td></tr>
<tr><td><code>reactive_acc</code></td><td class="num">0.90</td><td class="num w">0.60</td><td class="num">1.3%</td><td class="num g">1,756</td></tr>
<tr><td><code>proactive_margin</code></td><td class="num g">0.91</td><td class="num g">0.82</td><td class="num g">0.3%</td><td class="num w">9,145</td></tr>
</table>
<b>Reading.</b> The margin trigger raises <b>worst-case served accuracy by +22 points</b>
(0.60 → 0.82) and near-eliminates downtime. In this <i>mild</i>-shock regime that costs ~5× more
<i>steady</i> maintenance compute (constant cheap top-ups vs. the reactive policy's rare deep
recoveries). Under <i>severe/frequent</i> shocks the recovery cost flips and prevention wins on
compute too. Either way: <b>prevent the cliff, don't chase it.</b>
</div>
<img src="data:image/png;base64,{trace_png}" alt="accuracy (lagging cliff) vs margin (leading signal) for the three policies">

<div class="spec"><b>Wrapping it around your own model.</b> Measure your model's <i>healthy</i> margin
once, set <code>margin_lo</code>/<code>margin_hi</code> to a band just under it, supply
(a) a fixed probe set and (b) a <code>repair_stream</code> of fresh labelled (or pseudo-labelled)
data, and call <code>mon.tick(damage_fn)</code> in your serving loop. For large models the repair
steps should update a small adapter, not the full weights — see §5.</div>
<pre>mon = NESSMonitor(model, task, MonitorCfg(policy="proactive_margin",
                                          margin_lo=..., margin_hi=...), probe)
telem = mon.tick(damage_fn)   # measures served margin, repairs iff it dipped below the band</pre>

<h2>3 · The honest cost trade-off</h2>
<p>Proactive maintenance is not a free lunch. Across a controlled sweep (same margin band, only the
perturbation size σ varied) it <b>always won on worst-case served accuracy</b> — min-accuracy of
<span class="num">0.83 / 0.55 / 0.34</span> at σ = 0.07 / 0.11 / 0.15 vs the reactive policy's
<span class="num">0.60 / 0.44 / 0.25</span>. The <i>cost</i> (and downtime), though, depend on how hard
the environment hits:</p>
<ul>
<li><b>Mild shocks</b> (the demo, σ=0.07): the reactive policy rarely has to act, so it's cheap
(<span class="num">~1,800</span> steps); proactive's constant top-ups cost <b>~5× more</b>
(<span class="num">~9,000</span> steps) but drive downtime to near zero. You are buying reliability with compute.</li>
<li><b>Severe shocks</b> (σ=0.15, ~2× larger): the reactive policy pays for deep recoveries again and
again, so proactive becomes <i>cheaper</i> in total repair (<span class="num">4,083 vs 8,159 steps</span>)
while still holding the better worst case. The per-event hysteresis gap (recovery ≫ prevention) now
dominates. (At this severity the substrate is so battered that <i>no</i> policy keeps downtime low —
but proactive keeps the highest floor.)</li>
</ul>
<p>So the decision rule is: <b>the more expensive a worst-case failure is (or the harsher the
environment), the more proactive margin-maintenance pays for itself.</b> For reliability-critical
serving, the worst-case column is the one that matters.</p>

<h2>4 · The alignment corollary <span class="pill w">grounded but speculative</span></h2>
<p>P3, read the other way, is the investigation's most interesting and least certain claim. The
condition that makes a population of self-repairing models <i>evolve a will-to-live</i> — spend
effort to keep themselves alive, out-reproduce those that don't — is precisely a
<b>maintenance budget decoupled from the reproductive/capability budget</b>. On a single shared
currency, self-preservation is selected <i>against</i>. Decoupling flips it on.</p>
<div class="null"><b>The lever, stated carefully.</b> If you are training or evolving a population of
agents and you do <i>not</i> want emergent self-preservation, <b>keep survival effort coupled to the
task budget</b> — make staying alive cost the same resource as doing the job, so selection prunes
hoarders. If you <i>do</i> want durable self-maintenance (a fleet that keeps itself healthy),
<b>decouple it</b>. This is a structural knob, demonstrated in a small model, that I'd treat as a
hypothesis to test at scale — not a law. It is suggestive precisely because the flip was clean and
reproduced on real decaying nets, but the substrate is tiny and the mapping to large agentic systems
is unproven.</div>

<h2>5 · What kind of ML model is this — and can it run on the 3080 / 5090?</h2>
<p><b>It is not a new architecture.</b> It's a <b>continual / online-maintenance pattern</b> — a
lightweight watchdog + bounded-repair loop — that wraps any model which is (i) served continuously,
(ii) exposed to drift or degradation, and (iii) has access to a trickle of fresh labels or a
self-supervised proxy to repair from. Good fits:</p>
<ul>
<li><b>Continually fine-tuned production classifiers / rankers / recommenders</b> facing distribution drift.</li>
<li><b>Edge / embedded models</b> that degrade over time (sensor drift, quantization drift) and get periodic label feedback.</li>
<li><b>A frozen base + trainable adapter</b> (LoRA/PEFT): this <i>is</i> P3 realised — the frozen base is the
capability ("reproductive") store; the adapter is the decoupled maintenance budget. Repair updates
the adapter only; the margin watchdog decides when.</li>
<li><b>A fleet of small models</b> (P4): the cheap probe-forward lets one monitor watch many instances and stagger their maintenance.</li>
</ul>
<p><b>Poor fits:</b> a static model behind a fixed API with no drift (nothing to maintain), or any
setting with no label/proxy signal to repair from (nothing to repair toward).</p>

<div class="card"><b>Hardware feasibility — verdict: HIGH, single consumer GPU is plenty.</b>
The pattern is deliberately cheap: the monitor's per-tick cost is one forward pass on a small probe
set (sub-millisecond for small models) plus, only when triggered, a handful of backprop steps. Cost
scales with model size, not with anything exotic.
<table>
<tr><th></th><th>RTX 3080 (this box, ~10 GB)</th><th>RTX 5090 (~32 GB GDDR7)</th></tr>
<tr><td><b>This investigation</b></td><td>ran here — seconds to a few minutes per experiment; models are tiny</td><td>trivial</td></tr>
<tr><td><b>Online-maintain, full weights</b></td><td>models up to a few hundred M params (fp16)</td><td>~1–3 B params comfortably</td></tr>
<tr><td><b>Online-maintain via LoRA (frozen base)</b></td><td>7 B-class with a 4-bit base (~5–6 GB) + adapter grads; probe + short repair bursts fit easily</td><td>7–13 B-class in bf16/QLoRA, real-time; margin watchdog + bursts with headroom</td></tr>
<tr><td><b>Fleet monitoring</b></td><td>several small models concurrently (probe pass is cheap)</td><td>a larger fleet, or 7 B-class + fleet of satellites</td></tr>
</table>
<b>Recommended deployment recipe</b> (maps the safe structure onto standard tooling): freeze the base
(capability store), attach a <b>LoRA adapter</b> (the decoupled maintenance budget, P3), run the
margin watchdog on a rolling probe set (P1), and on a margin dip do <i>N</i> adapter-only SGD steps
on fresh/pseudo-labelled data until the margin clears the release band (P2). Full fine-tuning of
&gt;1–2 B params won't fit on the 3080; that's the only real ceiling, and LoRA sidesteps it. The 3080
is sufficient for prototyping and for maintaining small-to-mid models or a 7 B-LoRA; the 5090 buys
headroom for 7–13 B-class online maintenance and larger fleets.</div>

<h2>6 · Where these model types are actually used</h2>
<p>The pattern is not academic — the four model types above are the bread-and-butter of production
ML. Concrete applications for each, with an honest note on how much the margin-monitor loop adds:</p>

<h3>Continually-tuned classifiers / rankers / recommenders under drift</h3>
<ul>
<li><b>Fraud &amp; abuse detection</b> (payments, account takeover, fake reviews) — adversaries adapt
continuously and labels arrive <i>late</i> (chargebacks, investigations resolve weeks on), so
degradation is silent.</li>
<li><b>Content moderation / spam filtering</b> — adversarial drift: new slang, evasion spellings, novel manipulations.</li>
<li><b>Ad CTR prediction &amp; recommendation ranking</b> (feeds, e-commerce, video) — tastes shift, catalogs churn, seasonality; labels stream back fast.</li>
<li><b>Search / e-commerce ranking, dynamic pricing, demand forecasting, churn &amp; credit-risk scoring</b> — slower drift, macro shifts.</li>
</ul>
<div class="spec"><b>Fit: strong.</b> Margin = ranking confidence / calibration; these are the classic
"accuracy is a lagging cliff" domains where you want the confidence gap's warning before revenue or catch-rate craters.</div>

<h3>Edge / embedded models that degrade</h3>
<ul>
<li><b>Predictive maintenance</b> on industrial equipment (bearings, turbines, HVAC) — sensor drift + aging + seasonal conditions; labels = actual failures / maintenance logs.</li>
<li><b>Manufacturing visual QC / defect detection</b> — line changes, lighting, new product variants.</li>
<li><b>Wearables &amp; continuous health monitoring</b> (arrhythmia, glucose trends) — per-user physiology forces personalization.</li>
<li><b>Robotics / drones / autonomous vehicles</b> — domain shift from environment, wear, payload.</li>
<li><b>On-device keyword spotting / speech</b> — acoustic drift, new accents.</li>
</ul>
<div class="spec"><b>Fit: strong — and this is where the <i>lightweight</i> loop matters most.</b> You can't
re-train on-device, so a scalar watchdog + small adapter repair is exactly the right size.</div>

<h3>Frozen base + LoRA adapter (large foundation models / LLMs)</h3>
<ul>
<li><b>Domain-specialized assistants</b> tracking a changing corpus (internal docs, policies, product catalog, a live codebase) — repair the adapter, don't re-fine-tune the base.</li>
<li><b>Customer-support / RAG chatbots</b> tracking evolving products and FAQs.</li>
<li><b>Code models</b> keeping pace with an evolving codebase and API changes.</li>
<li><b>Per-user / per-tenant personalization</b> — a small adapter that tracks a user's style over time.</li>
<li><b>Countering capability decay</b> after quantization or as usage patterns shift.</li>
</ul>
<div class="null"><b>Fit: good, with a caveat.</b> For LLMs much "drift" is better handled by retrieval
(RAG) than by weight repair. The pattern earns its keep when <i>behavior / calibration itself</i> drifts, not just the facts.</div>

<h3>Fleet of small models (the correlated-shock principle, P4)</h3>
<ul>
<li><b>Per-store / per-region</b> demand and inventory models across a retail chain.</li>
<li><b>Per-device / per-vehicle</b> predictive-maintenance models (a truck fleet, a wind farm).</li>
<li><b>Per-tenant SaaS classifiers</b>; <b>per-sensor anomaly detectors</b> in a large IoT deployment.</li>
</ul>
<div class="spec"><b>Fit: this is exactly where P4 pays off.</b> Stagger maintenance and keep per-instance
variation so one correlated event (a bad data push, an upstream schema change) can't silently tank the whole fleet at once. One cheap monitor watches many.</div>

<div class="key"><b>Where the pattern actually earns its keep.</b> The margin-monitor + proactive-repair
loop is worth the machinery when <b>all four</b> hold: (1) degradation is <b>silent</b> — accuracy
looks fine until it cliffs, so a leading indicator buys warning time; (2) worst-case failures are
<b>expensive</b> — fraud losses, a missed equipment failure, a moderation escape, a safety event;
(3) a <b>repair signal exists</b> — delayed labels, a self-supervised proxy, or fresh in-distribution
data; (4) full retraining is <b>costly</b> relative to small repairs. Fraud, safety-critical
predictive maintenance, and moderation-at-scale hit all four — the sweet spot.
<br><br><b>Where it's overkill:</b> a static model with no real drift (nothing to maintain), or any
setting where accuracy is <i>not</i> a lagging signal — if you get fast, cheap ground-truth on every
prediction, plain "monitor accuracy, retrain on a schedule" is simpler and enough. This pattern is
the answer to <i>silent, costly degradation you can't see coming.</i></div>

<h2>7 · What NOT to build <span class="pill w">the null result, reused</span></h2>
<div class="null">The investigation spent real effort on <b>epistemic awareness</b> — giving agents an
explicit internal estimate of their own time-to-death (τ̂) and letting selection act on it. It was
<b>neutral when free and purged when it cost anything</b>: redundant with the margin signal the
environment already provides, and drift-dominated. <b>Engineering takeaway:</b> don't spend model
capacity or compute on an explicit self-modelling / mortality-estimator module. The cheap external
gauge (the margin on a probe set) carries the same information and selection never rewards the fancy
internal one. Spend the budget on <i>maintenance</i>, not on <i>introspection</i>.</div>

<h2>8 · Limits of the analogy</h2>
<ul>
<li>The substrate is a population of <b>tiny MLPs</b> on a synthetic task. Every quantitative
number here (margins, step counts, the +22-point worst-case gain) is specific to that setup and
seed; treat them as <i>illustrations of the mechanism</i>, not as transferable constants.</li>
<li>"Decay" is modelled as weight shrink + additive shocks. Real deployment degradation (covariate
shift, label drift, upstream changes) is more structured; the margin-as-leading-indicator claim
should be <b>re-validated on your own drift</b> before you trust the trigger.</li>
<li>The evolutionary result (P3/§4) is a clean finding in a small model. Reading it as a lever for
large agentic systems is a <b>hypothesis</b>, offered as such.</li>
<li>None of this is a claim about machine sentience or genuine "wanting". "Will to live" is shorthand
for <i>a trait that spends effort on self-maintenance and is favoured by selection</i> — a
measurable population phenomenon, nothing more.</li>
</ul>

<div class="sub" style="margin-top:34px">Reference implementation: <code>ness_monitor.py</code> ·
Full investigation and controls: <a href="/investigation.html">investigation.html</a>.
This page is the "so what," not the evidence — the evidence is there.</div>

</div></body></html>"""
    with open(OUT, "w") as f:
        f.write(doc)
    print(f"wrote {OUT}  ({os.path.getsize(OUT):,} bytes)")


if __name__ == "__main__":
    main()
