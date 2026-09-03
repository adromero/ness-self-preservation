#!/usr/bin/env python3
"""
NESS-Ecology MVP (Stage 1) — implements SPEC-ecology.md.

A chemostat + Moran birth-death ecology on the NESS substrate. Agents (MLPs under
decay+repair) compete for one finite energy pool, die permanently, reproduce with
heritable+mutable traits, and prey on each other (kill-the-winner, bounded). The
trait under selection is an ordinal awareness allele L0 (reactive) vs L1
(death-aware: senses its decay slope + estimated time-to-death tau_hat).

Question: does L1 out-reproduce L0 NET OF A COST (global scale `lam`)?
The answer is the sign of selection on awareness as a function of lam.

Everything is synchronous (Jacobi: decisions from the frozen post-decay snapshot,
applied atomically), energy is conserved to 1e-9 every tick (only source = supply S,
only sinks = named dissipation), and the whole thing is seed-deterministic.

Modes:
  python eco_mvp.py --validate                 # the bug-free gate (fast)
  python eco_mvp.py --run --lam 1.0 --ticks 2000
  python eco_mvp.py --sweep                     # pilot lam-sweep -> results/eco_sweep.csv
"""
from __future__ import annotations
import argparse, copy, csv, json, math, os, random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

import ness_experiment as ne  # MLP + decay_step (validated substrate)

DEVICE = ne.DEVICE
RESULTS = ne.RESULTS_DIR
CHANCE = 0.10


# ---------------------------------------------------------------------------
# Config (defaults from SPEC-ecology.md §13)
# ---------------------------------------------------------------------------
@dataclass
class Cfg:
    N_slots: int = 64
    ticks: int = 2000
    seed: int = 0
    # substrate
    decay: float = 0.10            # base per-tick decay d (operating point: robust, no extinction, ~1 death/tick)
    repair_batch: int = 256
    repair_step_cap: int = 6       # bounded within-life repair (anti-Lamarck)
    a_death: float = 0.30          # health death line
    probe_n: int = 512
    # energy economy
    S: float = 90.0                # supply/tick (sets carrying capacity; N*~0.7 slots)
    m: float = 0.5                 # baseline drain / agent / tick
    e_init: float = 10.0
    e_repro_threshold: float = 18.0
    child_cost: float = 9.0
    income_cap: float = 8.0        # max an agent can draw from its fair share/tick
    c_step: float = 0.5            # energy per SGD repair step (sink)
    repair_gain: float = 0.06      # health recovered per effective SGD step (Holling via cap)
    # predation (bounded: eta<1, cost, cap, contest, retaliation)
    predation: bool = True
    eta: float = 0.6               # transfer efficiency (<1 mandatory)
    c_attack: float = 0.8          # attack cost (paid win/lose, dissipated)
    steal_cap: float = 4.0         # COMBATLIMIT
    # awareness cost (swept via lam): per-tick energy tax for L1
    lam: float = 1.0
    cost_l1: float = 1.2           # base L1 awareness cost (organ+info), scaled by lam
    obs_noise0: float = 0.15       # sigma when organ underfunded (A<1)
    # evolution
    mut_ell: float = 0.02          # prob awareness allele flips (+/-1) per birth
    sigma_w: float = 0.08          # policy-weight Gaussian mutation
    sigma_trait: float = 0.03      # ecological-trait mutation
    sigma_tag: float = 0.05        # neutral lineage-marker drift (per birth)
    # controls
    scramble: bool = False         # C1 (global): all L1 senses decorrelated from truth
    init_scramble_frac: float = 0.0  # C1 (per-agent): fraction of initial L1 that are scrambled
    freeze_types: bool = False     # lock ell & scrambled (pure-lineage competition)
    terminal_repro: bool = False   # actionable-tau: allow terminal investment near death
    terminal_cost: float = 4.0     # energy per terminal offspring (< child_cost)
    terminal_health_factor: float = 1.0  # terminal offspring health = this x parent health
    terminal_brood: int = 2        # offspring from one terminal (semelparous) event
    no_selection: bool = False     # C4: parent chosen uniformly (drift), not by energy
    init_ell: int = -1             # -1 = random 0/1; else fix all agents at this level
    init_mix: str = ""             # "compete" -> seed equal thirds L0 / L1-real / L1-scram
    init_fracs: str = ""           # "l0,real,scram" fractions -> exact deterministic seeding
    pretrain_epochs: int = 1
    policy_hidden: int = 0         # 0 = linear-softmax brain; >0 = one tanh hidden layer of this width
    use_margin_sense: bool = True  # L1 senses the margin leading-indicator (vs old redundant accuracy-slope)
    shock_prob: float = 0.0        # B: per-agent prob of a decay SHOCK each tick (0 = off)
    shock_mult: float = 3.0        # decay multiplier during a shock
    hazard_period: int = 0         # ecological NESS: seasonal decay cycle length in ticks (0 = stationary)
    hazard_amp: float = 0.0        # amplitude of the seasonal decay modulation (fraction of base decay)
    catastrophe_prob: float = 0.0  # per-tick prob of a CORRELATED catastrophe hitting ALL agents at once
    catastrophe_mult: float = 6.0  # decay multiplier during a catastrophe (severity -> bet-hedging variance)
    catastrophe_dur: int = 1       # ticks a catastrophe PULSE lasts (rare+brief+severe = coarse-grained)
    repro_cooldown: int = 0        # rate-limited reproduction: refractory ticks after a birth (0 = unlimited)
    fear_reserve: float = 0.0      # energy a fearful agent hoards above the repro threshold (fat reserve)
    famine_frac: float = 1.0       # supply multiplier during a catastrophe (0 = total famine -> starvation)
    famine_dormant: bool = False   # during a famine: no decay/repair/breeding -> survival = reserve vs duration (threshold-lethal)
    base_mortality: float = 0.0    # type-neutral per-tick random death -> continuous turnover (sub-saturation)
    maint_steps: int = 0           # NESS-port: free maintenance-repair steps/tick toward fear-target margin (0=off)
    fear_target_scale: float = 4.0 # target margin an agent maintains = fear * this
    decoupled: bool = False        # maintenance repair FREE (separate budget) vs energy-costed (single currency)
    famine_drain: float = 1.0      # baseline-drain multiplier during a dormancy famine (starvation cost -> demand)
    fear_ref: float = 2.0          # margin below which fear activates (crude danger cue)
    init_fear_frac: float = 0.0    # fraction seeded fearful (rest fearless) for the fear competition
    init_fear_val: float = 0.0     # fear gain of the fearful lineage


F_DIM = 4  # features: [h, e, slope, tau_hat]; L0 masks the last two
N_ACT = 4  # actions: [REPAIR, REPRODUCE, PREDATE, REST]


# ---------------------------------------------------------------------------
# Data (preloaded to GPU for speed; same batches/statistics as NESS)
# ---------------------------------------------------------------------------
def load_gpu_data(cfg: Cfg):
    from torchvision import datasets, transforms
    tfm = transforms.ToTensor()
    tr = datasets.MNIST("data", train=True, download=True, transform=tfm)
    te = datasets.MNIST("data", train=False, download=True, transform=tfm)
    Xtr = tr.data.float().div(255.0).view(-1, 784).to(DEVICE)
    Ytr = tr.targets.to(DEVICE)
    g = torch.Generator().manual_seed(cfg.seed)
    idx = torch.randperm(len(te.data), generator=g)[: cfg.probe_n]
    Xpr = te.data[idx].float().div(255.0).view(-1, 784).to(DEVICE)
    Ypr = te.targets[idx].to(DEVICE)
    return Xtr, Ytr, Xpr, Ypr


def pretrain_body(Xtr, Ytr, Xpr, Ypr, cfg: Cfg):
    torch.manual_seed(cfg.seed)
    m = ne.MLP().to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    n = Xtr.shape[0]
    for _ in range(cfg.pretrain_epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n - 256, 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            F.cross_entropy(m(Xtr[b]), Ytr[b]).backward()
            opt.step()
    return m.state_dict()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class Agent:
    __slots__ = ("model", "opt", "e", "A", "ell", "W", "kappa_rep", "rho_dec",
                 "s_str", "tag", "h", "ema_slope", "age", "aid", "gen", "rng", "scrambled",
                 "margin", "margin_slope", "fear", "repro_cd")

    def __init__(self, state, ell, W, kappa_rep, rho_dec, s_str, tag, e, aid, base_seed,
                 scrambled=False, fear=0.0):
        self.model = ne.MLP().to(DEVICE)
        self.model.load_state_dict(state)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.e = float(e)
        self.A = 1.0
        self.ell = int(ell)
        self.scrambled = bool(scrambled)   # C1: L1 senses present+costed but decorrelated
        self.fear = float(fear)            # heritable caution drive (conative, not epistemic)
        self.repro_cd = 0                  # reproduction refractory countdown (rate limit)
        self.W = W                    # np array (4, F_DIM+1)
        self.kappa_rep = float(kappa_rep)
        self.rho_dec = float(rho_dec)
        self.s_str = float(s_str)
        self.tag = float(tag)
        self.h = 0.0
        self.ema_slope = 0.0
        self.margin = 0.0        # confidence margin (leading indicator of the accuracy cliff)
        self.margin_slope = 0.0  # ema of margin change (negative = eroding toward collapse)
        self.age = 0
        self.aid = aid
        # per-agent RNGs keyed by id -> trajectory is independent of iteration order
        self.gen = torch.Generator(device=DEVICE.type).manual_seed((base_seed * 1_000_003 + aid) & 0x7FFFFFFF)
        self.rng = np.random.default_rng(base_seed * 7919 + aid)


def eval_health(model, Xpr, Ypr):
    model.eval()
    with torch.no_grad():
        return (model(Xpr).argmax(1) == Ypr).float().mean().item()


def eval_health_margin(model, Xpr, Ypr):
    """Accuracy (h = death variable) AND confidence margin (leading indicator) in one pass.
    Margin = mean(top1_logit - top2_logit): erodes ~30-40 decay steps BEFORE accuracy cliffs,
    so it is genuine subclinical foresight, provably not a function of accuracy."""
    model.eval()
    with torch.no_grad():
        lg = model(Xpr)
        acc = (lg.argmax(1) == Ypr).float().mean().item()
        t2 = lg.topk(2, dim=1).values
        margin = (t2[:, 0] - t2[:, 1]).mean().item()
    return acc, margin


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------
class World:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.gen = torch.Generator(device=DEVICE.type).manual_seed(cfg.seed)
        self.Xtr, self.Ytr, self.Xpr, self.Ypr = load_gpu_data(cfg)
        self.body0 = pretrain_body(self.Xtr, self.Ytr, self.Xpr, self.Ypr, cfg)
        self.pool = 0.0
        self.t = 0
        self.next_id = 0
        self.agents: list[Agent] = []
        _mix = [(0, False), (1, False), (1, True)]  # L0, L1-real, L1-scram
        if cfg.init_fear_frac > 0.0:               # fear competition: fearless vs fearful (all L0)
            n_fear = round(cfg.init_fear_frac * cfg.N_slots)
            fears = [cfg.init_fear_val] * n_fear + [0.0] * (cfg.N_slots - n_fear)
            for fv in fears:
                self.agents.append(self._spawn_random(force_ell=0, force_fear=fv))
        elif cfg.init_fracs:                           # exact seeding for invasion assays
            fl0, freal, fscram = (float(x) for x in cfg.init_fracs.split(","))
            n_l0 = round(fl0 * cfg.N_slots); n_real = round(freal * cfg.N_slots)
            n_scram = cfg.N_slots - n_l0 - n_real
            seed_types = ([(0, False)] * n_l0 + [(1, False)] * n_real + [(1, True)] * n_scram)
            for el, sc in seed_types:
                self.agents.append(self._spawn_random(force_ell=el, force_scram=sc))
        else:
            for i in range(cfg.N_slots):
                if cfg.init_mix == "compete":
                    el, sc = _mix[i % 3]
                    self.agents.append(self._spawn_random(force_ell=el, force_scram=sc))
                else:
                    self.agents.append(self._spawn_random())
        for a in self.agents:
            a.h, a.margin = eval_health_margin(a.model, self.Xpr, self.Ypr)
        # ledger bookkeeping
        self.dissipated_last = 0.0
        self.deaths_total = 0; self.births_total = 0
        self.deaths_last = 0; self.births_last = 0
        self.attacks_total = 0; self.steal_total = 0.0
        self.terminal_births_total = 0
        self.deaths_health = 0; self.deaths_energy = 0; self.deaths_semel = 0
        self.hd_marg = []; self.ed_marg = []   # margins at moment of health/energy death
        self.terminal_dying = set()
        self._scram_feats = {}
        self.catastrophe_last = False
        self._cat_timer = 0
        self.supply_last = cfg.S

    # ---- genome helpers ----
    def _rand_W(self):
        H = self.cfg.policy_hidden
        if H <= 0:
            return self.rng.normal(0, 0.5, size=(N_ACT, F_DIM + 1)).astype(np.float64)
        # small MLP: F_DIM -> H (tanh) -> N_ACT ; biases start at 0 (output biases learn drift)
        W1 = self.rng.normal(0, 0.5, size=(H, F_DIM)).astype(np.float64)
        b1 = np.zeros(H, dtype=np.float64)
        W2 = self.rng.normal(0, 0.5, size=(N_ACT, H)).astype(np.float64)
        b2 = np.zeros(N_ACT, dtype=np.float64)
        return (W1, b1, W2, b2)

    # ---- policy helpers (linear ndarray OR mlp tuple) ----
    def _copy_mutate_policy(self, W, sigma):
        if isinstance(W, tuple):
            return tuple(x + self.rng.normal(0, sigma, size=x.shape) for x in W)
        return W + self.rng.normal(0, sigma, size=W.shape)

    @staticmethod
    def _unmask_senses(W):
        # neutral unmask: zero the weights fed by the slope(2)/tau_hat(3) input channels
        if isinstance(W, tuple):
            W[0][:, 2] = 0.0; W[0][:, 3] = 0.0
        else:
            W[:, 2] = 0.0; W[:, 3] = 0.0

    def _spawn_random(self, force_ell=None, force_scram=None, force_fear=None):
        c = self.cfg
        if force_ell is not None:
            ell = int(force_ell)
            scrambled = bool(force_scram)
        else:
            ell = (int(self.rng.integers(0, 2)) if c.init_ell < 0 else c.init_ell)
            scrambled = (ell == 1 and self.rng.random() < c.init_scramble_frac)
        fear0 = float(force_fear) if force_fear is not None else 0.0
        a = Agent(self.body0, ell, self._rand_W(),
                  kappa_rep=1.0, rho_dec=0.0, s_str=1.0,
                  tag=float(self.rng.normal()), e=c.e_init, aid=self.next_id,
                  base_seed=c.seed, scrambled=scrambled, fear=fear0)
        self.next_id += 1
        return a

    def _child(self, parent: Agent, kind="normal"):
        c = self.cfg
        W = self._copy_mutate_policy(parent.W, c.sigma_w)
        ell = parent.ell
        if (not c.freeze_types) and self.rng.random() < c.mut_ell:
            ell = int(np.clip(ell + self.rng.choice([-1, 1]), 0, 1))
            # NEUTRAL UNMASK: newly exposed L1 senses start at zero weight
            if ell == 1 and parent.ell == 0:
                self._unmask_senses(W)
        scrambled = parent.scrambled   # fixed lineage label (inherited, never mutates)
        fear = parent.fear
        if not c.freeze_types:
            fear = float(np.clip(parent.fear + self.rng.normal(0, c.sigma_trait), 0.0, 1.0))
        kappa = float(np.clip(parent.kappa_rep + self.rng.normal(0, c.sigma_trait), 0.5, 2.0))
        rho = float(np.clip(parent.rho_dec + self.rng.normal(0, c.sigma_trait), 0.0, 0.6))
        s = float(np.clip(parent.s_str + self.rng.normal(0, c.sigma_trait), 0.3, 2.5))
        tag = parent.tag + self.rng.normal(0, c.sigma_tag)   # neutral marker: drifts every birth
        e0 = c.terminal_cost if kind == "terminal" else c.child_cost
        ch = Agent(parent.model.state_dict(), ell, W, kappa, rho, s, tag,
                   e=e0, aid=self.next_id, base_seed=c.seed, scrambled=scrambled, fear=fear)
        ch.h = parent.h  # inherits the (damaged) body, so heritable fitness variance exists
        ch.margin = parent.margin; ch.margin_slope = parent.margin_slope
        if kind == "terminal":
            ch.h = parent.h * c.terminal_health_factor   # runt: lower starting health -> lower EV
        ch.ema_slope = parent.ema_slope
        self.next_id += 1
        return ch

    # ---- observation + policy ----
    def _real_l1_feats(self, a: Agent):
        """Death-awareness features. With use_margin_sense (default) these are the MARGIN
        leading-indicator (level + trend) that erodes before the accuracy cliff -> genuine
        foresight. Otherwise the old accuracy-slope/tau_hat (shown to be redundant with h)."""
        if self.cfg.use_margin_sense:
            lvl = (a.margin - 3.5) / 3.5        # normalized margin (neg = danger zone)
            trend = a.margin_slope * 3.0         # margin trend (neg = eroding toward collapse)
            return lvl, trend
        slope_f = a.ema_slope * 10.0
        tau = a.h / max(1e-3, -a.ema_slope) if a.ema_slope < 0 else 999.0
        tau_f = math.tanh(50.0 / max(1.0, tau)) - 0.5   # high when near death
        return slope_f, tau_f

    def _obs(self, a: Agent):
        c = self.cfg
        h_f = a.h - 0.5
        e_f = (a.e - c.e_repro_threshold) / c.e_repro_threshold
        if a.ell >= 1:
            if c.scramble or a.scrambled:
                # C1 control: another L1 agent's REAL (slope, tau) feats, assigned by a
                # tick-seeded permutation (precomputed in tick). This is distribution-EXACT
                # (same values, just reassigned) and decorrelated from THIS agent's state,
                # so cost + channels are matched but the senses carry no self-information.
                slope_f, tau_f = self._scram_feats.get(a.aid) or self._real_l1_feats(a)
            else:
                slope_f, tau_f = self._real_l1_feats(a)
            # organ underfunding -> observation noise
            noise = c.obs_noise0 * (1.0 / max(1e-3, a.A) - 1.0)
            if noise > 0:
                slope_f += float(a.rng.normal(0, noise))
                tau_f += float(a.rng.normal(0, noise))
            obs = np.array([h_f, e_f, slope_f, tau_f])
            mask = np.array([1.0, 1, 1, 1])
        else:
            obs = np.array([h_f, e_f, 0.0, 0.0])
            mask = np.array([1.0, 1, 0, 0])
        return obs * mask

    def _decide(self, a: Agent):
        obs = self._obs(a)                   # 4-dim masked features
        W = a.W
        if isinstance(W, tuple):
            W1, b1, W2, b2 = W
            logits = W2 @ np.tanh(W1 @ obs + b1) + b2
        else:
            logits = W @ np.append(obs, 1.0)  # linear + bias
        z = logits - logits.max()
        p = np.exp(z); p /= p.sum()
        return p  # [REPAIR, REPRODUCE, PREDATE, REST]

    # ---- the tick ----
    def tick(self, order=None):
        c = self.cfg
        n = len(self.agents)
        if n == 0:
            return
        # honor iteration order (used by the order-independence check); per-agent
        # RNGs are keyed by agent id, so a correct tick is order-invariant in energy.
        A = self.agents if order is None else [self.agents[i] for i in order]
        dissipated = 0.0
        self.deaths_last = 0; self.births_last = 0
        self.terminal_dying = set()
        self.t += 1

        # CORRELATED, rare, BRIEF, severe pulse hitting ALL agents at once (coarse-grained variance)
        catastrophe = False
        if c.catastrophe_prob > 0.0:
            if self._cat_timer > 0:
                catastrophe = True; self._cat_timer -= 1
            else:
                crng = np.random.default_rng((c.seed * 99991 + self.t * 7 + 3) & 0x7FFFFFFF)
                if crng.random() < c.catastrophe_prob:
                    catastrophe = True; self._cat_timer = c.catastrophe_dur - 1
        self.catastrophe_last = catastrophe

        # 1. REPLENISH (a catastrophe can be a FAMINE: supply cut -> agents live off reserves)
        self.supply_last = c.S * (c.famine_frac if catastrophe else 1.0)
        self.pool += self.supply_last

        # 2. DECAY (bodies + organ wear) — recompute health
        env_mult = 1.0
        if c.hazard_period > 0:      # ecological NESS: shared, autocorrelated seasonal hazard
            env_mult = 1.0 + c.hazard_amp * math.sin(2.0 * math.pi * self.t / c.hazard_period)
        dormant = catastrophe and c.famine_dormant   # famine dormancy: pure reserve-survival test
        for a in A:
            d_eff = 0.0 if dormant else c.decay * (1.0 - a.rho_dec) * env_mult
            if catastrophe and not dormant:
                d_eff *= c.catastrophe_mult
            if c.shock_prob > 0.0:   # B: occasional decay shock (order-independent per-agent draw)
                srng = np.random.default_rng((c.seed * 911 + self.t * 7919 + a.aid) & 0x7FFFFFFF)
                if srng.random() < c.shock_prob:
                    d_eff *= c.shock_mult
            ne.decay_step(a.model, d_eff, a.gen)
            new_h, new_margin = eval_health_margin(a.model, self.Xpr, self.Ypr)
            slope = new_h - a.h
            a.ema_slope = 0.5 * a.ema_slope + 0.5 * slope
            a.h = new_h
            m_slope = new_margin - a.margin
            a.margin_slope = 0.5 * a.margin_slope + 0.5 * m_slope
            a.margin = new_margin

        # C1 scramble surrogates: give each scrambled L1 agent ANOTHER L1 agent's real
        # (slope, tau) feats via a tick-seeded permutation. Distribution-exact + decorrelated
        # + order-independent (computed from aid-sorted state, deterministic seed).
        self._scram_feats = {}
        if c.scramble or any(a.scrambled for a in A):
            l1 = sorted((x for x in A if x.ell >= 1), key=lambda z: z.aid)
            scr = [x for x in l1 if (c.scramble or x.scrambled)]
            if l1 and scr:
                pool = [self._real_l1_feats(x) for x in l1]
                srng = np.random.default_rng((c.seed * 2654435761 + self.t * 2246822519) & 0x7FFFFFFF)
                perm = srng.permutation(len(pool))
                for i, x in enumerate(scr):
                    self._scram_feats[x.aid] = pool[int(perm[i % len(perm)])]

        # frozen snapshot for Jacobi decisions
        acts = {a.aid: self._decide(a) for a in A}
        e0 = {a.aid: a.e for a in A}
        h0 = {a.aid: a.h for a in A}

        # 3. PREDATION (two-phase, kill-the-winner, bounded)
        if c.predation and n >= 2:
            # phase A: intents from frozen state
            attackers = {}  # victim_aid -> list of attacker Agents
            for a in A:
                if acts[a.aid][2] < 0.34:      # not choosing PREDATE strongly
                    continue
                if e0[a.aid] < c.c_attack:
                    continue
                # pick richest ELIGIBLE neighbor (kill-the-winner), skip retaliation-risky.
                # deterministic tie-break by aid -> target choice is order-independent even
                # when frozen energies tie (e.g. all-equal on the first tick).
                best, best_key = None, None
                for b in A:
                    if b.aid == a.aid:
                        continue
                    if b.s_str > a.s_str:      # retaliation skip
                        continue
                    k = (e0[b.aid], -b.aid)    # highest energy, then lowest aid
                    if best_key is None or k > best_key:
                        best, best_key = b, k
                if best is not None:
                    a.e -= c.c_attack; dissipated += c.c_attack  # cost paid win/lose
                    self.attacks_total += 1
                    attackers.setdefault(best.aid, []).append(a)
            # phase B: resolve, cap total extraction at victim energy
            for vaid, atks in attackers.items():
                victim = next(x for x in A if x.aid == vaid)
                pot = min(c.steal_cap, e0[vaid])
                # contest per victim+tick RNG -> order-independent resolution.
                # resolve attackers in a canonical (aid) order so each gets a fixed RNG
                # draw regardless of iteration order.
                vrng = np.random.default_rng((self.cfg.seed * 15485863 + self.t * 40503 + vaid) & 0x7FFFFFFF)
                winners = [x for x in sorted(atks, key=lambda z: z.aid)
                           if vrng.random() < x.s_str / (x.s_str + victim.s_str)]
                if not winners:
                    continue
                extract = min(pot, victim.e)          # cannot take more than victim holds now
                if extract <= 0:
                    continue
                victim.e -= extract
                per = extract / len(winners)
                for w in winners:
                    w.e += c.eta * per
                self.steal_total += c.eta * extract
                dissipated += (1.0 - c.eta) * extract  # friction

        # 4. ALLOCATION (fair-share income) + SPEND (repair/reproduce/rest)
        live = [a for a in A]
        share = self.pool / max(1, len(live))
        for a in A:
            income = min(c.income_cap, share)
            income = min(income, self.pool)
            a.e += income; self.pool -= income
        # spend per policy split (from frozen acts)
        births = []
        for a in A:
            p = acts[a.aid]
            spend_repair = 0.0 if dormant else p[0] * a.e
            # FEAR (conative): a reflexive caution drive, NOT an information channel. When the
            # crude danger cue fires (margin below fear_ref) a fearful agent diverts extra energy
            # to repair BEYOND the policy's fitness-greedy choice -- sacrificing reproduction for
            # survival. fear=0 is fearless (pure policy). Selection tunes the gain.
            if a.fear > 0.0 and a.margin < c.fear_ref:
                danger = (c.fear_ref - a.margin) / c.fear_ref
                spend_repair = spend_repair + a.fear * danger * (a.e - spend_repair)
            # repair: energy -> bounded SGD steps -> health recovery (real substrate)
            k = min(c.repair_step_cap, int(spend_repair / c.c_step))
            if k > 0:
                a.model.train()
                for _ in range(k):
                    bi = torch.randint(0, self.Xtr.shape[0], (c.repair_batch,),
                                       generator=a.gen, device=DEVICE)
                    a.opt.zero_grad()
                    F.cross_entropy(a.model(self.Xtr[bi]), self.Ytr[bi]).backward()
                    a.opt.step()
                cost = k * c.c_step
                a.e -= cost; dissipated += cost
                new_h, new_m = eval_health_margin(a.model, self.Xpr, self.Ypr)
                a.h = min(1.0, new_h)
                a.margin = new_m   # repair rebuilds margin (cheap while healthy; costly after collapse)
            # MAINTENANCE repair toward the fear-target margin (the ported self-preservation hedge):
            # free (decoupled 'maintenance currency') or energy-costed (single currency).
            if (not dormant) and c.maint_steps > 0 and a.margin < a.fear * c.fear_target_scale:
                a.model.train()
                for _ in range(c.maint_steps):
                    bi = torch.randint(0, self.Xtr.shape[0], (c.repair_batch,), generator=a.gen, device=DEVICE)
                    a.opt.zero_grad()
                    F.cross_entropy(a.model(self.Xtr[bi]), self.Ytr[bi]).backward()
                    a.opt.step()
                if not c.decoupled:
                    cost = min(c.maint_steps * c.c_step, a.e); a.e -= cost; dissipated += cost
                nh, nm = eval_health_margin(a.model, self.Xpr, self.Ypr)
                a.h = min(1.0, nh); a.margin = nm
            # reproduce: normal if rich enough; else a TERMINAL 'runt' bet (if enabled).
            # Timing the terminal bet well needs tau_hat -> this is the actionable-tau lever.
            wants_repro = p[1] >= max(p[0], p[2], p[3])
            slot_free = len(A) + len(births) < c.N_slots
            repro_gate = c.e_repro_threshold + a.fear * c.fear_reserve   # fear -> hoard a fat reserve
            if wants_repro and slot_free and a.repro_cd <= 0 and not dormant:
                if a.e >= repro_gate:
                    a.e -= c.child_cost
                    births.append((a, "normal", c.child_cost))
                    a.repro_cd = c.repro_cooldown
                elif c.terminal_repro and a.e >= c.terminal_cost:
                    # SEMELPAROUS terminal investment: dump energy into a brood, then die.
                    # Timing this to low tau_hat (imminent death) is the actionable-tau lever;
                    # doing it while you would have survived forfeits your whole future.
                    brood = min(c.terminal_brood, int(a.e / c.terminal_cost))
                    for _ in range(brood):
                        a.e -= c.terminal_cost
                        births.append((a, "terminal", c.terminal_cost))
                    self.terminal_dying.add(a.aid)
            # REST/leftover: energy stays in a.e

        # 5. UPKEEP (awareness cost — the swept tax) + baseline drain
        for a in A:
            need = c.lam * (c.cost_l1 if a.ell >= 1 else 0.0)
            paid = min(need, a.e)
            a.e -= paid; dissipated += paid
            a.A = 1.0 if need <= 1e-9 else float(np.clip(paid / need, 0.0, 1.0))
            drain_rate = c.m * (c.famine_drain if (self.catastrophe_last and c.famine_dormant) else 1.0)
            drain = min(drain_rate, a.e)
            a.e -= drain; dissipated += drain
            a.age += 1
            if a.repro_cd > 0:
                a.repro_cd -= 1

        # 6. DEATH (permanent; carcass energy -> pool = conserved)
        survivors = []
        for a in A:
            died_random = False
            if c.base_mortality > 0.0:
                mrng = np.random.default_rng((c.seed * 1299709 + self.t * 131 + a.aid) & 0x7FFFFFFF)
                died_random = mrng.random() < c.base_mortality
            if a.h < c.a_death or a.e <= 0.0 or a.aid in self.terminal_dying or died_random:
                self.pool += max(0.0, a.e)       # recycle remaining energy
                self.deaths_last += 1
                if a.aid in self.terminal_dying:
                    self.deaths_semel += 1       # died BY reproducing (semelparity)
                elif a.e <= 0.0:
                    self.deaths_energy += 1; self.ed_marg.append(a.margin)
                elif a.h < c.a_death:
                    self.deaths_health += 1; self.hd_marg.append(a.margin)
            else:
                survivors.append(a)
        self.agents = survivors

        # 7. REPRODUCTION (Moran fill; per-birth cost pre-paid into 'in-flight' energy).
        #    births = list of (parent, kind, cost); each realized child is born with e0=cost.
        inflight = sum(cost for (_, _, cost) in births)
        realized_cost = 0.0
        slots = births
        if c.no_selection:  # drift control: random parents keep the same (kind,cost) slots
            k = len(births)
            rp = list(self.rng.choice(survivors, size=min(k, len(survivors)), replace=False)) \
                if survivors and k else []
            slots = [(rp[i], births[i][1], births[i][2]) for i in range(len(rp))]
        for parent, kind, cost in slots:
            if len(self.agents) >= c.N_slots:
                break
            if kind != "terminal" and parent not in self.agents:
                continue                         # normal birth: skip if parent died this tick
            self.agents.append(self._child(parent, kind))   # child born with e0=cost
            self.births_last += 1
            if kind == "terminal":
                self.terminal_births_total += 1
            realized_cost += cost
        # any pre-paid cost whose birth did not realize returns to the pool (conserved)
        self.pool += inflight - realized_cost

        self.deaths_total += self.deaths_last
        self.births_total += self.births_last
        self.dissipated_last = dissipated

    def total_energy(self):
        return self.pool + sum(a.e for a in self.agents)

    def snapshot(self):
        A = self.agents
        n = len(A)
        ell1 = sum(1 for a in A if a.ell == 1)
        n_real = sum(1 for a in A if a.ell == 1 and not a.scrambled)
        n_scram = sum(1 for a in A if a.ell == 1 and a.scrambled)
        n_l0 = sum(1 for a in A if a.ell == 0)
        return dict(n=n, frac_L1=(ell1 / n if n else 0.0),
                    frac_L1_real=(n_real / n if n else 0.0),
                    frac_L1_scram=(n_scram / n if n else 0.0),
                    frac_L0=(n_l0 / n if n else 0.0),
                    n_real=n_real, n_scram=n_scram, n_l0=n_l0,
                    n_fear=sum(1 for a in A if a.fear > 1e-9),
                    n_fearless=sum(1 for a in A if a.fear <= 1e-9),
                    frac_fear=(sum(1 for a in A if a.fear > 1e-9) / n if n else 0.0),
                    mean_fear=(np.mean([a.fear for a in A]) if n else 0.0),
                    mean_h=(np.mean([a.h for a in A]) if n else 0.0),
                    mean_e=(np.mean([a.e for a in A]) if n else 0.0),
                    total_energy=self.total_energy(),
                    mean_tag=(np.mean([a.tag for a in A]) if n else 0.0),
                    deaths=self.deaths_last, births=self.births_last,
                    deaths_total=self.deaths_total, births_total=self.births_total)


# ---------------------------------------------------------------------------
# Runs / metrics
# ---------------------------------------------------------------------------
def run(cfg: Cfg, record_every=10, assert_conservation=True):
    w = World(cfg)
    hist = []
    for t in range(cfg.ticks):
        pre = w.total_energy()
        w.tick()
        if assert_conservation:
            expected = pre + w.supply_last - w.dissipated_last
            err = abs(w.total_energy() - expected)
            assert err < 1e-6, f"CONSERVATION VIOLATION at t={t}: err={err:.3e}"
        assert w.pool >= -1e-9 and all(a.e >= -1e-9 for a in w.agents), f"negative energy at t={t}"
        if len(w.agents) == 0:
            hist.append((t, w.snapshot())); break
        if t % record_every == 0 or t == cfg.ticks - 1:
            hist.append((t, w.snapshot()))
    return w, hist


def _final_fracL1(hist):
    return hist[-1][1]["frac_L1"] if hist else float("nan")


def _tail_fracL1(hist, frac=0.3):
    """Mean frac_L1 over the last `frac` of recorded snapshots (robust equilibrium
    estimate; single-snapshot frac_L1 is noisy at small N)."""
    if not hist:
        return float("nan")
    k = max(1, int(len(hist) * frac))
    return float(np.mean([s["frac_L1"] for _, s in hist[-k:]]))


# ---------------------------------------------------------------------------
# Validation gate (SPEC §12)
# ---------------------------------------------------------------------------
def validate():
    print("=== NESS-Ecology validation gate ===")
    ok = True

    # 1. strict conservation on a normal run (asserted inside run())
    c = Cfg(ticks=120, N_slots=32, seed=1)
    w, h = run(c, record_every=20)
    print(f"[1] conservation: PASS (120 ticks, no violation)  final pop={h[-1][1]['n']}")

    # 2. no-negative / no-double-spend also asserted inside run()
    print("[2] no-negative energy / pool: PASS (asserted every tick)")

    # 3. order-independence: same seed + same-length tick, different agent iteration
    #    order -> identical total energy and identical per-agent energies (no double-spend).
    def one_tick(order_seed):
        ww = World(Cfg(N_slots=24, seed=7))
        order = list(np.random.default_rng(order_seed).permutation(len(ww.agents)))
        ww.tick(order=order)
        return round(ww.total_energy(), 6), tuple(sorted(round(a.e, 6) for a in ww.agents))
    tot1, es1 = one_tick(11)
    tot2, es2 = one_tick(999)
    same = (tot1 == tot2) and (es1 == es2)
    print(f"[3] order-independence: {'PASS' if same else 'FAIL'}  "
          f"(total {tot1} vs {tot2}; per-agent match={es1 == es2})")
    ok &= same

    # 4. population stability near carrying capacity, no extinction/saturation
    c = Cfg(ticks=800, N_slots=64, seed=2)
    w, h = run(c, record_every=20)
    tail = [s["n"] for _, s in h[len(h) // 2:]]
    stable = (min(tail) > 3) and (max(tail) <= c.N_slots) and (np.mean(tail) > 8)
    print(f"[4] population stability: {'PASS' if stable else 'FAIL'}  "
          f"tail pop mean={np.mean(tail):.1f} min={min(tail)} max={max(tail)} (cap {c.N_slots})")
    ok &= stable

    # 5. heritability: parent-offspring fitness (health) correlation > 0
    c = Cfg(ticks=200, N_slots=48, seed=3)
    w = World(c)
    pairs = []
    orig_child = w._child
    def spy_child(parent, kind="normal"):
        ch = orig_child(parent, kind)
        pairs.append((parent.h, ch.h))
        return ch
    w._child = spy_child
    for _ in range(c.ticks):
        w.tick()
        if not w.agents:
            break
    if len(pairs) > 20:
        ph, chh = zip(*pairs)
        r = float(np.corrcoef(ph, chh)[0, 1])
        herit = r > 0.1
    else:
        r, herit = float("nan"), False
    print(f"[5] heritability (parent-offspring health r): {'PASS' if herit else 'FAIL'}  r={r:.2f} (n={len(pairs)})")
    ok &= herit

    # 6. predation is a LIVE, non-lethal-to-the-system mechanism.
    #    Correctness bar (asserted): with predation ON the ecology (a) stays viable
    #    (no extinction) and (b) the predation mechanism actually fires (attacks occur
    #    and energy is stolen) -- i.e. it is engaged, not silently dead code.
    #    Whether kill-the-winner predation *raises* trait diversity is an empirical
    #    question reported (not asserted) below, using a neutral lineage marker (tag).
    def tag_eff_number(agents, bins=20):
        vals = [a.tag for a in agents]
        if len(vals) < 2: return 1.0
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9: return 1.0
        idx = np.clip(((np.array(vals) - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
        _, cnt = np.unique(idx, return_counts=True); pr = cnt / cnt.sum()
        return float(np.exp(-(pr * np.log(pr)).sum()))
    w_on, _ = run(Cfg(ticks=400, N_slots=64, seed=4, predation=True), record_every=100)
    w_off, _ = run(Cfg(ticks=400, N_slots=64, seed=4, predation=False), record_every=100)
    on_viable = len(w_on.agents) > 3
    engaged = (w_on.attacks_total > 0) and (w_on.steal_total > 0.0)
    p6 = on_viable and engaged
    eff_on = tag_eff_number(w_on.agents); eff_off = tag_eff_number(w_off.agents)
    print(f"[6] predation live & viable: {'PASS' if p6 else 'FAIL'}  "
          f"(ON n={len(w_on.agents)} attacks={w_on.attacks_total} stolen={w_on.steal_total:.1f}; "
          f"OFF n={len(w_off.agents)})")
    print(f"    reported lineage diversity (tag eff-#): ON={eff_on:.2f}  OFF={eff_off:.2f}  "
          f"(empirical, not a pass/fail gate)")
    ok &= p6

    print(f"\nVALIDATION GATE: {'ALL CORE CHECKS PASS' if ok else 'SOME CHECKS FAILED — fix before science'}")
    return ok


# ---------------------------------------------------------------------------
# Pilot lam-sweep (the net-of-cost question)
# ---------------------------------------------------------------------------
def sweep(lams, reps, ticks, N, out_csv, S=None):
    """Pilot lambda-sweep: equilibrium L1 (awareness) fraction vs the awareness tax.

    lam=0 is the decisive control -- awareness is FREE, so any drop of frac_L1 below
    the neutral 0.5 there means the extra senses are intrinsically non-adaptive (net),
    independent of cost. Rising lam then adds the tax. frac_L1 is tail-averaged over the
    last 30%% of each run to damp small-N snapshot noise; each cell reports mean+/-std
    across `reps` independent seeds."""
    rows = []
    for lam in lams:
        finals, pops, extinct = [], [], 0
        for rep in range(reps):
            c = Cfg(lam=lam, ticks=ticks, N_slots=N, seed=1000 + rep)
            if S is not None:
                c.S = S
            w, h = run(c, record_every=10)
            fl = _tail_fracL1(h, 0.3); finals.append(fl)
            popn = len(w.agents); pops.append(popn)
            if popn == 0:
                extinct += 1
            print(f"  lam={lam:<6g} rep={rep} tail frac_L1={fl:.3f} pop={popn} "
                  f"deaths={w.deaths_total} births={w.births_total}", flush=True)
        m = float(np.mean(finals)); sd = float(np.std(finals))
        pm = float(np.mean(pops))
        rows.append((lam, m, sd, pm, extinct, reps))
        print(f"lam={lam:<6g}  frac_L1={m:.3f} +/- {sd:.3f}  pop~{pm:.1f}  "
              f"extinct={extinct}/{reps}  (n={reps})", flush=True)
    with open(out_csv, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["lam", "frac_L1_mean", "frac_L1_std", "pop_mean", "extinct", "n"])
        wtr.writerows([(l, f"{m:.4f}", f"{sd:.4f}", f"{pm:.2f}", ex, n)
                       for l, m, sd, pm, ex, n in rows])
    print(f"\nwrote {out_csv}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--S", type=float, default=None,
                    help="energy supply/tick (sets carrying capacity); default keeps Cfg.S")
    ap.add_argument("--out", type=str, default="eco_sweep.csv",
                    help="output CSV basename under results/")
    ap.add_argument("--lams", type=str, default="0,0.25,0.5,1.0,2.0",
                    help="comma-separated awareness-tax values for --sweep")
    a = ap.parse_args()
    print(f"device={DEVICE}")
    if a.validate:
        validate()
    elif a.run:
        w, h = run(Cfg(lam=a.lam, ticks=a.ticks, N_slots=a.N))
        print(json.dumps(h[-1][1], indent=2))
    elif a.sweep:
        lams = [float(x) for x in a.lams.split(",")]
        sw_ticks = a.ticks if a.ticks != 2000 else 400   # pilot default
        print(f"sweep: lams={lams} reps={a.reps} ticks={sw_ticks} N_slots={a.N} "
              f"S={a.S if a.S is not None else Cfg().S} -> results/{a.out}")
        sweep(lams, reps=a.reps, ticks=sw_ticks, N=a.N, S=a.S,
              out_csv=os.path.join(RESULTS, a.out))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
