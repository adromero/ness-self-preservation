#!/usr/bin/env python3
"""Does self-preservation evolve with THREE escapes but NO separate safety currency?

Minimal metapopulation bet-hedging model (CPU, fast) isolating the selection question the
neural NESS substrate could not resolve. A single fungible currency (energy): the hedge
(hoarding a reserve) trades 1:1 against reproduction, exactly as in the failed NESS runs.
The three escapes I argued the NESS class structurally lacks are toggleable:

  1. slow_life   : long lives, slow breeding -> FEW decisive draws per lineage (geometric selection)
  2. groups      : demes with LOCAL famines + limited migration -> lineage/group-level selection
  3. private     : per-agent idiosyncratic income shocks -> CONTINUOUS individual risk (payoff
                   matched in time to the continuous hedging cost, breaking the time-asymmetry)

FEAR is a heritable reserve-hoarding trait (won't breed until energy >= threshold + fear*scale),
mutating from 0. If mean fear evolves UP under (1&2&3), self-preservation is selected WITHOUT a
safety currency -> the currency is not needed. Ablations remove one escape at a time to see which
are necessary. The NESS-like control (all escapes off) should keep fear ~0.

Energy is conserved each tick (source = supply; sole sink = metabolism) and every stochastic
draw is keyed by (agent id, tick) so the tick is order-independent.
"""
import argparse
import numpy as np

# ---------------------------------------------------------------------------
class Cfg:
    def __init__(self, **kw):
        self.G = 24            # demes
        self.K = 14            # capacity per deme  (N = G*K = 336)
        self.S = 6.0           # supply per deme per tick
        self.income_cap = 1.2
        self.metab = 0.5       # baseline metabolism / tick
        self.repro_threshold = 8.0
        self.child_cost = 6.0
        self.repro_cooldown = 25     # slow breeding (slow life-history)
        self.base_mortality = 0.002  # low -> long lives
        self.max_age = 4000
        self.fear_reserve = 20.0     # energy hoarded per unit fear (single currency!)
        self.mut_fear = 0.04
        self.fear_max = 1.5
        self.init_fear = 0.0         # seed all agents at this fear (test valley-lock vs disfavored)
        self.init_frac_hi = -1.0     # >=0: invasion mode -> per-deme fraction seeded at fear_hi, rest fear_lo
        self.fear_lo = 0.1
        self.fear_hi = 1.2
        self.safety_currency = False  # DECOUPLED hedge: separate famine-buffer store; breeding NOT fear-gated
        self.safety_max = 20.0        # buffer capacity per unit fear
        self.safety_build = 1.0       # max energy->safety transfer / tick (one-time build)
        self.safety_upkeep = 0.005    # tiny energy upkeep per unit stored safety / tick
        self.safety_supply = 3.0      # SEPARATE per-deme supply feeding ONLY the safety store (the 2nd currency)
        # escapes (toggleable)
        self.slow_life = True        # if False: fast turnover (short lives, cheap breeding)
        self.groups = True           # if False: 1 well-mixed deme
        self.private = True          # if False: no idiosyncratic income shock
        self.income_cv = 0.8         # private income shock size (when private on)
        # famine (correlated within a deme; local when groups on, global when off)
        self.famine_prob = 0.004     # per-deme per-tick onset prob
        self.famine_dur = 35
        self.migrate_prob = 0.01
        self.ticks = 4000
        self.seed = 0
        for k, v in kw.items():
            setattr(self, k, v)
        if not self.groups:
            self.G, self.K = 1, self.G * self.K   # collapse to one big well-mixed deme
        if not self.slow_life:
            self.base_mortality = 0.03; self.repro_cooldown = 2; self.max_age = 200


class Agent:
    __slots__ = ("e", "fear", "deme", "age", "cd", "aid", "safety")
    def __init__(self, e, fear, deme, aid):
        self.e = e; self.fear = fear; self.deme = deme; self.age = 0; self.cd = 0; self.aid = aid; self.safety = 0.0


class World:
    def __init__(self, cfg):
        self.c = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.next_id = 0
        self.t = 0
        self.pools = [0.0] * cfg.G
        self.spools = [0.0] * cfg.G   # safety-currency pools (separate resource)
        self.fam = [0] * cfg.G           # famine ticks remaining per deme
        self.agents = []
        for g in range(cfg.G):
            n_hi = round(cfg.K * cfg.init_frac_hi) if cfg.init_frac_hi >= 0 else 0
            for k in range(cfg.K):
                if cfg.init_frac_hi >= 0:
                    fear0 = cfg.fear_hi if k < n_hi else cfg.fear_lo   # invasion: two fixed types
                else:
                    fear0 = cfg.init_fear
                self.agents.append(Agent(self.c.repro_threshold, fear0, g, self._id()))

    def _id(self):
        i = self.next_id; self.next_id += 1; return i

    def _draw(self, salt, aid):
        return np.random.default_rng((self.c.seed * 2654435761 + self.t * 40503 + aid * 97 + salt) & 0x7FFFFFFF)

    def total_energy(self):
        return sum(self.pools) + sum(self.spools) + sum(a.e + a.safety for a in self.agents)

    def tick(self):
        c = self.c; self.t += 1
        supply_added = 0.0; dissipated = 0.0

        # famine onset/decay per deme (correlated within a deme)
        for g in range(c.G):
            if self.fam[g] > 0:
                self.fam[g] -= 1
            elif self._draw(1, g).random() < c.famine_prob:
                self.fam[g] = c.famine_dur

        # REPLENISH (famine = no supply to that deme)
        for g in range(c.G):
            fam = self.fam[g] > 0
            add = 0.0 if fam else c.S
            self.pools[g] += add; supply_added += add
            if c.safety_currency:
                sadd = 0.0 if fam else c.safety_supply
                self.spools[g] += sadd; supply_added += sadd

        by_deme = [[] for _ in range(c.G)]
        for a in self.agents:
            by_deme[a.deme].append(a)

        # INCOME: density-dependent fair share x private idiosyncratic shock; order-independent
        for g in range(c.G):
            live = sorted(by_deme[g], key=lambda z: z.aid)
            if not live:
                continue
            share = self.pools[g] / len(live)
            claims = []
            for a in live:
                shock = 1.0
                if c.private:
                    shock = float(np.exp(self._draw(2, a.aid).normal(0.0, c.income_cv) - 0.5 * c.income_cv ** 2))
                claims.append(min(c.income_cap, share) * shock)
            tot = sum(claims)
            scale = 1.0 if tot <= 1e-12 else min(1.0, max(0.0, self.pools[g]) / tot)
            for a, cl in zip(live, claims):
                inc = cl * scale
                a.e += inc; self.pools[g] -= inc

        # SAFETY-INCOME: fill the buffer toward fear*safety_max from the SEPARATE safety pool.
        # This does NOT touch energy -> reproduction is untouched (genuine decoupling).
        if c.safety_currency:
            for g in range(c.G):
                live = sorted(by_deme[g], key=lambda z: z.aid)
                if not live:
                    continue
                demand = [max(0.0, a.fear * c.safety_max - a.safety) for a in live]
                demand = [min(d, c.safety_build) for d in demand]
                tot = sum(demand)
                sc = 1.0 if tot <= 1e-12 else min(1.0, max(0.0, self.spools[g]) / tot)
                for a, dd in zip(live, demand):
                    got = dd * sc; a.safety += got; self.spools[g] -= got

        # METABOLISM (+ decoupled SAFETY store dynamics if enabled)
        for a in self.agents:
            in_fam = self.fam[a.deme] > 0
            if c.safety_currency and in_fam:
                # famine: pay metabolism from the SAFETY buffer first, then energy
                d = c.metab
                from_s = min(a.safety, d); a.safety -= from_s; d -= from_s
                de = min(d, a.e); a.e -= de
                dissipated += from_s + de
            else:
                d = min(c.metab, a.e); a.e -= d; dissipated += d
                if c.safety_currency:
                    up = min(c.safety_upkeep * a.safety, a.e); a.e -= up; dissipated += up
            a.age += 1
            if a.cd > 0:
                a.cd -= 1

        # REPRODUCTION (local; fear -> hoard reserve before breeding; single currency)
        deme_count = [0] * c.G
        for a in self.agents:
            deme_count[a.deme] += 1
        births = []
        for a in self.agents:
            gate = c.repro_threshold if c.safety_currency else (c.repro_threshold + a.fear * c.fear_reserve)
            if a.cd <= 0 and a.e >= gate + c.child_cost and deme_count[a.deme] < c.K:
                a.e -= c.child_cost
                fear = float(np.clip(a.fear + self._draw(3, a.aid).normal(0, c.mut_fear), 0.0, c.fear_max))
                births.append(Agent(c.child_cost, fear, a.deme, self._id()))
                a.cd = c.repro_cooldown
                deme_count[a.deme] += 1
        self.agents.extend(births)

        # DEATH: starvation (e<=0), old age, or rare random mortality -> energy recycled to deme pool
        survivors = []
        for a in self.agents:
            rand_death = self._draw(4, a.aid).random() < c.base_mortality
            if a.e <= 0.0 or a.age > c.max_age or rand_death:
                self.pools[a.deme] += max(0.0, a.e)
                self.spools[a.deme] += max(0.0, a.safety)
            else:
                survivors.append(a)
        self.agents = survivors

        # MIGRATION (limited gene flow between demes)
        if c.groups and c.migrate_prob > 0:
            for a in self.agents:
                if self._draw(5, a.aid).random() < c.migrate_prob:
                    a.deme = int(self._draw(6, a.aid).integers(0, c.G))

        self._supply_last = supply_added; self._diss_last = dissipated
        return supply_added, dissipated


def evolve(cfg, record_every=200):
    w = World(cfg)
    traj = []
    for t in range(cfg.ticks):
        pre = w.total_energy()
        s, d = w.tick()
        err = abs(w.total_energy() - (pre + s - d))
        assert err < 1e-6, f"CONSERVATION VIOLATION t={t} err={err:.2e}"
        assert all(a.e >= -1e-9 for a in w.agents) and all(p >= -1e-9 for p in w.pools), f"neg energy t={t}"
        if not w.agents:
            traj.append((t, 0, float("nan"))); break
        if t % record_every == 0 or t == cfg.ticks - 1:
            traj.append((t, len(w.agents), float(np.mean([a.fear for a in w.agents]))))
    return w, traj


def run_label(label, **kw):
    fears, ns = [], []
    for seed in range(4):
        cfg = Cfg(seed=seed, **kw)
        w, traj = evolve(cfg)
        n = len(w.agents)
        fears.append(traj[-1][2] if n else float("nan")); ns.append(n)
    fears = np.array(fears)
    print(f"{label:34} final mean_fear = {np.nanmean(fears):.3f} ± {np.nanstd(fears):.3f}   "
          f"(pop {np.mean(ns):.0f}, n_seeds={len(fears)})", flush=True)
    return np.nanmean(fears)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ticks", type=int, default=4000)
    a = ap.parse_args()
    T = dict(ticks=a.ticks)
    print("Does FEAR (reserve hoarding, SINGLE currency) evolve from 0?  (fear_max=1.5)", flush=True)
    print("Decisive test = FULL(reserve=20, fear useful) vs NEUTRAL(reserve=0, fear inert):", flush=True)
    print("the gap is real selection; NEUTRAL is the pure mutation-boundary-bias baseline.\n", flush=True)
    full = run_label("FULL  (reserve=20, fear USEFUL)", fear_reserve=20.0, **T)
    neut = run_label("NEUTRAL (reserve=0, fear INERT)", fear_reserve=0.0, **T)
    print(f"  --> selection signal = FULL - NEUTRAL = {full - neut:+.3f}\n", flush=True)
    print("-- ablations (each vs its own bias; remove one escape) --", flush=True)
    run_label("no private risk", private=False, **T)
    run_label("no slow-life (fast turnover)", slow_life=False, **T)
    run_label("no groups (well-mixed; may go extinct)", groups=False, **T)
