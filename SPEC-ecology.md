# NESS-Ecology MVP — Specification

**One-line question:** *In a population competing for finite repair-energy under irreversible death, does an agent that models its own death (represents its time-to-die and conditions behavior on it) out-reproduce one that doesn't — net of the cost of carrying that model?*

This spec turns that question into a runnable, defensible experiment. Every mechanism below is concrete (a formula or a rule), every known failure mode is named with its fix, and the controls that separate a real result from an artifact are mandatory, not optional. It is built on the NESS substrate we already validated (sharp life/death boundary + irreversible collapse) and on four literature reviews (selection dynamics, policy/genome representation, resource-and-predation ecology, awareness-and-its-cost) whose key sources are cited inline and listed at the end.

**Scope of the claim (stated once, load-bearing throughout).** "Awareness" here is strictly **functional/representational** — Block's *access* sense: an internal representation of state (including the agent's own predicted death) that is *poised for use in decision and control* [Block; Kaelbling–Littman–Cassandra POMDP]. We make **no claim about phenomenal experience** (qualia, "what it is like"). "Death-awareness" means precisely: *the agent maintains a self-model of its own predicted termination (an estimated time-to-death τ̂) and acts on it.* That is measurable and non-mystical. This distinction is what keeps the experiment science rather than fantasy.

---

## 0. What the research *forces* (the anti-fantasy backbone)

Six constraints are not design taste — violate any one and the result is meaningless. They drive everything after.

1. **Fitness must be emergent, never assigned.** If we score agents by a hand-written function of awareness, we've assumed the conclusion (the classic ALife tautology, "you get what you select for"). Fitness = realized survival + reproduction in the ecology, full stop. This is why we use an Avida/Polyworld-style birth–death ecology, not a generational GA with a scalar objective. [Ofria–Wilke Avida; Yaeger Polyworld]

2. **Awareness's *only* advantage may be information — never capacity.** By the Blackwell/Good **value-of-information** theorem, costless extra observation can only help, so awareness would trivially ratchet to maximum and stop being a trait under selection. Three independent reviews converged on the same triad to prevent this:
   - **Fixed decision architecture across levels** — higher awareness changes *which inputs are visible*, never the parameter count, nonlinearity, or output head. (policy review)
   - **Neutral unmasking** — when a mutation raises awareness and exposes a new sense, its policy weight starts at **0**, so raising awareness is *behavior-neutral at birth* and must be *discovered* to pay. (policy review, borrowing NEAT's zero-weight new-connection trick)
   - **A real, unavoidable cost** — see constraint 5. (awareness-cost review)
   This is the **capacity confound**, flagged by the selection review as "the deepest threat," and the **scrambled-information control** (§10) is what proves we beat it.

3. **There must be a decision whose optimum depends on τ̂.** Value of information is zero if the best action is the same whatever the signal says [Howard VOI]. So the ecology **must** contain an allocation choice — *repair now vs. invest in reproduction vs. attack vs. rest* — whose optimum genuinely shifts with the agent's predicted time-to-death. Without it, death-awareness can never pay and the experiment is a guaranteed null by construction. (awareness-cost review — *the single biggest way to accidentally get a null*.)

4. **A single shared pool + single death axis ⇒ monoculture, unless predation.** With one limiting resource, **R\*/competitive-exclusion theory** predicts the lowest-R\* genotype (the most repair-efficient converter) drives everyone extinct — the "trivially fixate on one strategy" degeneracy. The fix with the most direct theoretical warrant is **kill-the-winner predation** (negative frequency-dependence: attacks fall hardest on the most abundant/richest type), which sustains diversity and produces bounded predator–prey cycles [Thingstad KtW; Holling-II; Tilman R\*]. Predation is therefore **not a flourish — it is the diversity engine**, and it doubles as a rich τ̂-dependent decision context.

5. **The cost of awareness must be real and hard to game.** Grounded in the "expensive-brain / expensive-tissue" and rational-inattention literatures [Isler–van Schaik; Sims; Landauer]. The **whole scientific content is whether awareness pays *after* subtracting its cost** — so the cost is a first-class, swept parameter, and the answer to "net of cost?" is literally *the cost level at which selection on awareness flips sign*.

6. **Small N ⇒ drift can masquerade as selection.** At N·s ≲ 1 a trait is effectively neutral [Ohta nearly-neutral]. So: measure the selection coefficient s, require **N·s ≫ 1** (≈ s ≥ 0.08 at N=64), run a **neutral-marker control** (a gene the policy never reads), replicate ≥ 30 independent populations, and show the signal **strengthens with N** (drift wouldn't).

---

## 1. The substrate (from NESS — unchanged)

Each agent's "body" is an MLP (784→256→256→10) pretrained on MNIST (clean ≈ 97%). Two forces act per tick (validated dynamics: sharp boundary, irreversible collapse):

- **Decay** `decay_step(model, d)`: to each weight tensor add Gaussian noise scaled by the tensor's own std, `w += N(0,1)·std(w)·d`, then zero a fraction `~d/10` of weights (bit-death). Damage accumulates.
- **Repair** `repair(model, r)`: `r` SGD steps (lr fixed) on fresh MNIST minibatches — undoes damage, costs energy.
- **Health** `h_i ∈ [0,1]` = accuracy on a small held-out probe set (cheap, ~3 batches).
- **Death**: `h_i < a_death` ⇒ **permanent removal**. Irreversible (empirically: below the line, repair can't rebuild — we make it literal by deleting the agent).

**Critical inherited constraint (Lamarckian leakage).** Within-life repair must be **bounded** (small `r`), or every agent relearns to ~97% regardless of inheritance, there is no heritable fitness variance, and "evolution" degenerates into within-life learning. Before trusting any selection result we **assert parent–offspring fitness correlation > 0** (heritability check).

---

## 2. The agent = body + awareness-organ + genome

| Part | Symbol | Role |
|---|---|---|
| Body (MLP weights) | `θ_i` | the substrate that decays; determines health `h_i` |
| Energy store | `e_i ≥ 0` | currency; spent on repair/reproduction/attack; 0 ⇒ starvation death |
| Awareness organ integrity | `A_i ∈ [0,1]` | must be maintained or observations degrade (`σ_obs ∝ 1/A_i`) — the cost mechanism |
| **Genome (heritable, mutable):** | | |
| awareness level | `ℓ ∈ {0,1,2,3}` | ordinal allele; mutates ±1 rarely (μ_a≈0.02) |
| policy weights | `W ∈ ℝ^{4×(F+1)}` | masked linear-softmax controller (~ (F+1)·4 params) |
| ecological traits | `κ_rep, s_str, ρ_dec` | repair-efficiency, fight-strength, decay-resistance (small Gaussian mutation) |
| neutral marker | `μ_tag` | **control gene the policy never reads** (drift baseline) |

Everything except `θ_i` (the pretrained body, copied at birth) and `A_i` (state) is inherited with mutation.

---

## 3. Sensing — awareness-gated observation (the ordinal ladder)

Awareness is a **POMDP observation function** knob: level `ℓ` unlocks a *strict superset* of features and a non-decreasing lookahead horizon `H_ℓ`. Features are **zero-mean normalized** so an unlocked-but-unused input doesn't shift the decision by default.

| ℓ | Observation vector `o^(ℓ)` (each ⊇ the one above) | Horizon | Grounding |
|---|---|---|---|
| **L0 reactive** | `[ h_i, e_i ]` | 0 | interoception [Barrett–Simmons] |
| **L1 death-aware** ★ | L0 ⊕ `[ ĥ'_i , τ̂_i ]` | ≥1 | predictive interoception / allostasis [Seth] |
| **L2 social** | L1 ⊕ nearest-k neighbor `[h, e, predator-flag]` | ≥H1 | richer O over multi-agent state |
| **L3 world-model** | L2 ⊕ features from short rollouts of a learned `f(o,a)→ô` | >H2 | world models [Ha–Schmidhuber] |

where `ĥ'_i` = EMA of the health decay slope, and the **death-awareness signal**
`τ̂_i = h_i / max(ε, −ĥ'_i)` = **estimated ticks-to-death**.

**L1 is the privileged rung**: the smallest level that explicitly represents its own predicted end and can act on it. The cleanest publishable headline is *"L1 is selected over both L0 and higher levels, net of cost."*

**Organ coupling:** when `A_i < 1`, every observed feature gets noise `σ_obs = σ0 / A_i` — i.e., a neglected awareness organ makes the senses unreliable. This is what makes awareness genuinely *costly to keep* (§7), not just costly to have.

---

## 4. Deciding — the policy (VOI-satisfying allocation)

Per tick each living agent maps its (masked) observation to an **energy allocation** — the decision constraint 3 requires:

```
logits = W · (mask_ℓ ⊙ o) + b                      # W fixed shape 4×(F+1), same across all ℓ
p = softmax(logits)                                 # fractions over the 4 actions
alloc = p · e_i_spendable        over {REPAIR, REPRODUCE_INVEST, PREDATE, REST}
```

- **REPAIR**: convert energy → accuracy recovery (diminishing returns, §6).
- **REPRODUCE_INVEST**: bank toward the reproduction threshold.
- **PREDATE**: fund an attack on a neighbor (§5).
- **REST**: hold energy (hedge).

The optimum split genuinely depends on `τ̂`: an agent that predicts imminent death should pour into REPAIR (or make a terminal reproduction bet); one that predicts safety should invest in REPRODUCE or opportunistic PREDATE. That τ̂-dependence is *where death-awareness earns its keep* — and exactly what the scrambled control removes.

**Representation choice:** masked **linear-softmax** (interpretable, ~20 params, no hidden-unit permutation symmetry ⇒ crossover-safe, smoothest landscape, and it keeps awareness a clean orthogonal gene) [policy review]. Upgrade to a tiny MLP only if the linear policy demonstrably can't express a needed behavior (Stage 2).

---

## 5. The world tick — deterministic, conservative, synchronous (Jacobi)

All decisions computed from the **frozen** start-of-tick state; deltas applied to a copy; committed at end. Fixed phase order (documented because it is behaviorally load-bearing):

```
1. REPLENISH   pool += S                              # the ONLY energy source
2. DECAY       h_i -= δ_i ;  A_i -= δ0·κ^ℓ_i          # damage (a debt), organ wear
3. PREDATION   (two-phase; see §5.1)
4. ALLOCATE    fair-share income, then agents spend per §4 (REPAIR uses §6 curve)
5. UPKEEP      organ maintenance drawn from the same pool (§7)
6. DEATH       h_i < a_death OR e_i ≤ 0 ⇒ remove; released energy → pool (carcass recycle)
7. REPRODUCE   fill freed slots via inheritance handoff (§8)
```

### 5.1 Predation (the diversity engine — kill-the-winner, bounded)

Two-phase to avoid double-spend/negative-energy bugs:
- **Phase A (intents, from frozen state):** each PREDATE-funding agent picks a target — biased to the **richest eligible neighbor** (kill-the-winner) — and skips **retaliation-vulnerable** targets (stronger neighbors). Pays attack cost `c_a` up front (→ dissipated), win or lose.
- **Phase B (resolution):** gather all attackers per victim; contest each with **Tullock** odds `P(win)=s_att/(s_att+s_def)` (seeded RNG); winners extract `min(cap, e_victim)` **capped total**; attacker gains `η·extracted` (η<1), victim loses `extracted`, `(1−η)·extracted` → **dissipated (friction)**. Invariant: `Σ extracted_from_v ≤ e_victim`.

**Four bounding levers** (why it can't degenerate to all-predator collapse or free-predator dominance):
| lever | default | effect |
|---|---|---|
| transfer efficiency `η` | **0.6** (<1 mandatory) | all-predator world is a net energy sink → starves → caps predator fraction (**primary bound**) |
| attack cost `c_a` | ≈ `η·cap·0.3` | predation pays only vs. *rich* victims → self-limiting, winner-biased |
| steal cap `cap` | ≈ 20–30% of a rich store | no one-kill jackpot → repeated attacks → prey/mutation can respond |
| contest + retaliation skip | Tullock; skip stronger | no guaranteed winner; no suicidal attacks |

### 5.2 Conservation invariant (asserted every tick — the core bug gate)

```
Total   = pool + Σ e_i
Dissipated = Σ baseline_drain + Σ repair_work + Σ organ_upkeep
           + Σ attack_costs + Σ (1−η)·extracted
assert | Total(t+1) − (Total(t) + S − Dissipated) | < 1e-9   # energy created ONLY by S, destroyed ONLY by named sinks
```

---

## 6. Energy economy & carrying capacity (chemostat, self-regulating)

It's a **chemostat** (one shared pool replenished at rate `S`), not a spatial Sugarscape. **Do not tune a target N — set the supply `S`; the population self-regulates** by negative density dependence to
`N* ≈ S / (m + r̄)` (m = baseline drain, r̄ = mean repair spend) [Tilman; Sugarscape K].

- **Fair-share allocator** `income_i = min(bid_i, pool/N_live)` (IFD-like, *stabilizing*). A merit-weighted allocator amplifies the R\* winner and accelerates monoculture — start fair-share; it's a primary dynamics knob.
- **REPAIR curve (diminishing returns, Holling-II-like):** `Δh_i = H_max · spend/(spend + h_half)` — prevents one agent from monopolizing repair.
- **Tuning targets for non-degenerate dynamics:** set `S` so `N* ≈ 0.6–0.8·N_slots`; set `δ` so the **median agent spends ~40–70% of its fair-share income on repair** (survivable but selective); `a_death` just below the equilibrium accuracy band.

---

## 7. The cost of awareness (hybrid; the load-bearing parameter)

Primary = **endogenous organ maintenance** (least gameable, reuses our own physics): the awareness organ decays `Ȧ = −δ0·κ^ℓ` and must be repaired from the **same finite pool**; underfunding raises `σ_obs = σ0/A_i`. You cannot get the benefit without paying continuously, and you cannot switch the cost off without losing the capability [expensive-brain]. Floor = **per-bit information tax** `λ_bit·(#extra observation dims provisioned)` — enforces "information is never free" [Sims; Landauer] and makes the scrambled control cost-matched automatically.

```
Cost_i(tick) = repair needed to hold A_i against δ0·κ^ℓ_i        (from the pool)
             + λ_bit · (extra dims unlocked at level ℓ_i)
```

A **single global cost-scale λ** multiplies `δ0` and `λ_bit` jointly. **This is the swept knob** — see §11.

*(Charge provisioned channel capacity / #dims, not realized mutual information — principled target, tractable proxy, and un-smugglable. Judgment call flagged.)*

---

## 8. Evolution / selection (Moran overlapping birth–death)

- **No generations replaced en masse.** One elementary step = an energy-triggered **birth** (an agent above the reproduction threshold spawns) and, at the population cap, a corresponding **death** (Moran replacement). Death is otherwise endogenous (§5.6). Define **1 "generation" = N_slots birth–death events** and state it (sets the units of s and fixation time).
- **Reproduction = inheritance handoff (conserved):** parent pays `child_cost` from its store into the offspring; child inherits `θ` (copied body), genome with mutation, into a freed slot only.
- **Mutation:** awareness `ℓ` ±1 at rate μ_a≈0.02 (reflecting at bounds) with **neutral unmask** (new sense's weight = 0); policy weights per-gene Gaussian `N(0,σ²)`, σ≈0.01; ecological traits Gaussian; neutral marker mutates at the **same rate as ℓ**. **Mutation-only** (no crossover) for the MVP — removes the competing-conventions failure class and is fine at ~20 params.
- **Reproducibility:** seed every stochastic step; keep a per-agent **seed-lineage log** (init seed + mutation seeds) for exact ancestry replay.

---

## 9. Measurement — *proving* selection (not eyeballing)

A defensible claim needs **agreement across ≥2 direct instruments + the neutral control**:

1. **Fixation probability** of an introduced higher-ℓ mutant vs the **neutral null 1/N** (invert Moran `ρ=(1−1/r)/(1−1/rᴺ)` to recover `s`). Binary ⇒ needs many seeds.
2. **Price selection covariance** `Cov(w_i, ℓ_i)/w̄` per generation (w = realized offspring); >0 ⇒ selection favors awareness that generation; test vs 0 **and** vs the neutral marker's covariance.
3. **Selection coefficient** `ŝ` from the slope of `logit(freq of high-ℓ)` vs generation; require CI excluding 0 and `N·ŝ ≳ 1`.
4. **Invasion / PIP** (heuristic at small N): pairwise invasibility of `ℓ′` into resident `ℓ`; the singular strategy `ℓ*` and its stability.

**Decision rule — "awareness selected at cost λ" iff all hold:** fixation prob significantly > 1/N and > neutral marker; Price covariance > 0 (vs 0 and vs marker); ŝ > 0 with CI excluding 0; signal **strengthens with N**. **Not selected** when awareness is statistically indistinguishable from the neutral marker.

**The "net of cost" answer is the sign of the selection gradient as a function of λ, and the located `λ*` where it flips.**

---

## 10. Controls (mandatory — each disarms a specific artifact)

| # | Control | Prediction if the result is real |
|---|---|---|
| C1 | **Scrambled, cost-matched** — same input channels, same organ cost, but the death-relevant signal is temporally permuted / another agent's τ̂ (zero task info) [Hewitt–Liang control task] | **Must NOT be selected** over L0. If it *is*, the "benefit" was capacity/noise/accounting, not information — result void. **This is the decisive control.** |
| C2 | **Free awareness (λ=0)** smoke test | awareness **must** win; if not even free, the info isn't beneficial and the premise fails |
| C3 | **Neutral marker** (gene policy never reads, same mutation rate) | fixation ≈ 1/N, Cov(w,tag) ≈ 0 — the empirical drift null to test against |
| C4 | **No-selection / drift** (flat fitness) | ℓ random-walks; any directional trend under selection is then attributable to selection |
| C5 | **N-scaling** (32/64/128) | selection signal grows with N; drift shrinks |
| C6 | **Decision-knockout** (force automatic repair) | VOI→0, awareness collapses to neutral — confirms the effect is decision-mediated |
| C7 | **Lesion** (ablate an evolved high-ℓ agent's extra inputs at test) | fitness/behavior degrades — proves it *uses* what it pays for |

C1–C5 are in the MVP; C6–C7 are confirmatory (Stage 2).

---

## 11. The experiment

1. **Smoke (C2):** λ=0 — confirm awareness wins when free.
2. **Cost sweep:** log-grid λ (~10 points over 10⁻³–10¹), ≥30 replicate populations each, run to evolutionary stationarity (≥ 3–10×`(1/s)·ln(2N)` generations). Record the **evolutionarily stable / modal ℓ** and the selection gradient's sign. Expect three regimes: **trivial-high** (λ→0, top level fixes), **trivial-low** (λ→∞, L0 fixes), and an **interior window** where an intermediate level (ideally **L1**) is stable — *the interesting regime*.
3. **Headline result (defensible to a skeptic):** in the interior window, **L1 is selected over both L0 (dies knowing less) and higher levels (pay for horizon they can't cash in), while the scrambled cost-matched control C1 is NOT selected.** ⇒ death-awareness is selected *net of its cost*, and it's the **information**, not architecture or accounting.
4. **(Stage 2) directional map:** 2-D λ × environment-predictability (decay volatility / predator lethality). The result that truly survives review is *directional*: harsher / more-predictable environments push `ℓ*` **up**. A single point proves nothing; a sensible monotone surface does.

---

## 12. Validation / "bug-free" gate (run before any science)

1. **Conservation:** with S=0, reproduction off, sinks off ⇒ `Total` constant to 1e-9; sinks on ⇒ ledger balances every tick (§5.2).
2. **No-negative / no-double-spend:** `e_i≥0`, `pool≥0`, `Σ income ≤ pool_before`, `Σ extracted_from_v ≤ e_victim` — asserted every tick.
3. **Order-independence:** shuffle agent iteration order under fixed RNG ⇒ identical trajectory (catches read-then-write leaks).
4. **Population stability:** 10k ticks ⇒ `N(t)` settles near `N*` with bounded CV; never 0 (extinction) nor pinned at `N_slots` (saturation). Treat extinctions as **censored**, reported, never silently averaged over.
5. **Heritability:** parent–offspring fitness correlation > 0 (else Lamarckian leakage — no variance to select on).
6. **Non-degeneracy (the design's own predation test):** **predation OFF ⇒ trait diversity collapses to the lowest-R\* genotype** (competitive-exclusion prediction confirmed); **predation ON ⇒ diversity sustained + bounded predator–prey oscillation** (kill-the-winner confirmed). If ON and OFF look identical, predation isn't working (η too high / cap too low / targeting not winner-biased).

---

## 13. Parameters (defaults; ★ = expect to sweep/tune)

| Param | Default | Param | Default |
|---|---|---|---|
| `N_slots` | 64 (sweep 32/64/128) | `η` transfer eff. ★ | 0.6 |
| `S` supply/tick ★ | set `N*≈0.7·N_slots` | `c_a` attack cost ★ | ≈η·cap·0.3 |
| `m` baseline drain | ~5–10% fair-share income | `cap` steal cap ★ | ~25% rich store |
| `δ` decay/tick ★ | repairable by 40–70% income | contest | Tullock s/(s+s) |
| `a_death` | just below equil. band | `H_max,h_half` repair curve | clear diminishing returns |
| awareness levels (MVP) | **L0, L1** (add L2/L3 Stage 2) | `μ_a` (ℓ mutation) | 0.02, ±1 |
| policy σ | 0.01 | global cost-scale `λ` ★ | **log-sweep 10⁻³–10¹** |
| replicates | ≥30 populations / condition | run length | ≥3–10×(1/s)ln(2N) gens |

---

## 14. MVP cut vs. growth path (keeping it *minimal* viable)

**MVP (Stage 1) — the smallest thing that answers the core question defensibly:**
- Substrate + chemostat + **predation ON** (needed for diversity *and* as a τ̂-relevant decision context).
- Awareness **L0 vs L1 only** (binary-ish) — the cleanest possible "is death-awareness selected?"
- Masked linear-softmax policy over {repair, reproduce, predate, rest}.
- Hybrid cost with global λ; **λ-sweep** = the net-of-cost answer.
- Moran birth–death, N=64; mutation-only.
- Controls **C1–C5**; validation §12 all six.

**Deferred (Stage 2+):** L2 social + L3 world-model awareness; the 2-D λ×environment map; PIP/ESS full treatment; tiny-MLP policy; controls C6–C7. These *extend* the intent (the full ladder, richer awareness, arms-race dynamics); none is needed to answer "is death-awareness (L1 over L0) selected, net of a real cost, because of the information (C1)?"

---

## 15. Compute feasibility (RTX 3080)

N=64 tiny MLPs, bounded within-life repair, health on a ~3-batch probe — *cheaper per tick than the NESS sweep we already ran* (adaptive repair, small N). A full λ-sweep with ≥30 replicates is an afternoon-to-overnight run, not an epic. It bolts onto the existing `ness_experiment.py` decay/repair primitives.

---

## 16. What this does and does NOT establish

- **Does:** whether *mortality-modeling as a heritable trait* is favored by selection in a competitive, predator-ridden, finite-resource world with irreversible death — net of a principled cost, with the information (not capacity) shown to be the cause (C1).
- **Does NOT:** touch phenomenal experience. A high-ℓ agent *behaves as if it knows death is real*; nothing here shows it *experiences* anything. Same wall as before — kept explicit so the result can't be over-read.

---

## 17. Open judgment calls (flagged, with the recommended default)

1. **Whether L2/L3 sit on the same scalar "awareness axis" as L0/L1** — defensible only as "information + horizon," not "amount of mind." (Recommend: keep, defend as such; MVP uses only L0/L1 so it's moot until Stage 2.)
2. **Cost currency** — organ-maintenance (recommended, least gameable) vs. simple per-level tax (simpler, coarser). (Recommend: organ-maintenance + info floor.)
3. **Allocator** — fair-share (stabilizing, recommended) vs. merit-weighted (amplifies R\* winner).
4. **Death-check placement** — decay→predation→repair→death (recommended) means predation can pre-empt a repair; A/B it.
5. **Charge provisioned dims vs realized MI** — provisioned (tractable, un-smuggleable) recommended; note the gap to a reviewer.
6. **Do not bake in that L1 wins** — that it *emerges* is the result.

---

## References (key; full lists in the four research briefs)
Selection & dynamics: Moran 1958; Kimura 1962 / Haldane 1927 (fixation); Price 1970/72; Maynard Smith 1982, Geritz et al. 1998 (ESS/adaptive dynamics); Nowak 2006; Ohta 1973 (nearly-neutral); Lenski LTEE (neutral marker). ALife platforms: Ray (Tierra); Ofria–Wilke (Avida); Yaeger (Polyworld); Epstein–Axtell (Sugarscape). Ecology: Tilman (R\*); Gause (competitive exclusion); Thingstad / Maslov–Sneppen (kill-the-winner); Holling (functional response); Fretwell (IFD); Hutchinson (paradox of the plankton); Maynard Smith (Hawk–Dove). Policy/genome: Stanley–Miikkulainen (NEAT); Salimans et al. (ES); Such et al. (deep neuroevolution); Cuccu et al. (six neurons); Blackwell 1953 / Good (value of information). Awareness & cost: Kaelbling–Littman–Cassandra (POMDP); Block (access vs phenomenal); Seth (interoceptive predictive coding); Ha–Schmidhuber (world models); Howard 1966 (VOI); Sims (rational inattention); Tishby–Pereira–Bialek (information bottleneck); Landauer; Isler–van Schaik (expensive brain); Hewitt–Liang 2019 (control tasks).
