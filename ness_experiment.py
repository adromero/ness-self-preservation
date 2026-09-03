"""
NESS Experiment: decay vs. repair in a neural network
======================================================
Tests whether a network under continuous weight decay + online repair
behaves like a non-equilibrium steady state (NESS), looking for four signatures:

  1. Steady state under throughput  (performance plateaus below clean ceiling)
  2. Collapse on starvation         (repair off -> decay to chance level)
  3. Phase transition               (sharp boundary in d/r space, not smooth)
  4. Irrecoverability / hysteresis  (starve too long -> repair can't recover)

Runs on CPU or GPU. Designed for an overnight run on consumer hardware.
Outputs: results/*.csv and results/*.png

Usage:
    pip install torch torchvision matplotlib pandas
    python ness_experiment.py            # full overnight run
    python ness_experiment.py --quick    # ~15 min sanity check
"""

import argparse
import copy
import csv
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CHANCE = 0.10  # 10-class chance level
DEAD_THRESHOLD = 0.15  # below this we call the network "dead"


# ----------------------------------------------------------------------------
# Model + data
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)


def get_data(batch_size=256):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST("data", train=True, download=True, transform=tfm)
    test = datasets.MNIST("data", train=False, download=True, transform=tfm)
    # num_workers=0: for a tiny MLP the DataLoader worker IPC costs more than it
    # saves, and it avoids worker-process churn across the thousands of run_life
    # iterators created over a multi-hour run. Data pipeline only — the batches,
    # batch size, and shuffling are unchanged.
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True,
                              pin_memory=(DEVICE.type == "cuda"))
    test_loader = DataLoader(test, batch_size=1024, shuffle=False, num_workers=0,
                             pin_memory=(DEVICE.type == "cuda"))
    return train_loader, test_loader


def evaluate(model, test_loader, max_batches=None):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


# ----------------------------------------------------------------------------
# Decay + repair primitives
# ----------------------------------------------------------------------------
def decay_step(model, d, generator):
    """One tick of substrate decay.

    d is the noise scale, applied multiplicatively relative to each layer's
    weight std so decay is meaningful across layers. Also randomly zeroes a
    small fraction of weights proportional to d (dropout-like permanent damage
    until repaired).
    """
    with torch.no_grad():
        for p in model.parameters():
            std = p.std().clamp(min=1e-8)
            noise = torch.randn(p.shape, generator=generator, device=p.device)
            p.add_(noise * std * d)
            # zero out a fraction ~ d/10 of weights (bit-death)
            mask = torch.rand(p.shape, generator=generator, device=p.device) < (d / 10)
            p.masked_fill_(mask, 0.0)


def repair_steps(model, optimizer, train_iter, train_loader, r):
    """r gradient steps of online repair (metabolism)."""
    model.train()
    for _ in range(r):
        try:
            x, y = next(train_iter[0])
        except StopIteration:
            train_iter[0] = iter(train_loader)
            x, y = next(train_iter[0])
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()


# ----------------------------------------------------------------------------
# Experiment phases
# ----------------------------------------------------------------------------
def pretrain(train_loader, test_loader, epochs=3):
    print(f"[pretrain] training healthy network ({epochs} epochs) on {DEVICE}")
    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        acc = evaluate(model, test_loader)
        print(f"  epoch {ep+1}: test acc {acc:.4f}")
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "healthy.pt"))
    return model


def run_life(model_state, d, r, ticks, train_loader, test_loader,
             eval_every, lr, seed, log_writer=None, tag=""):
    """Run `ticks` timesteps of decay+repair. Returns list of (tick, acc)."""
    model = MLP().to(DEVICE)
    model.load_state_dict(model_state)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    gen = torch.Generator(device=DEVICE.type if DEVICE.type != "cuda" else "cuda")
    gen.manual_seed(seed)
    train_iter = [iter(train_loader)]
    history = []
    for t in range(ticks):
        decay_step(model, d, gen)
        if r > 0:
            repair_steps(model, opt, train_iter, train_loader, r)
        if t % eval_every == 0 or t == ticks - 1:
            acc = evaluate(model, test_loader, max_batches=3)
            history.append((t, acc))
            if log_writer:
                log_writer.writerow([tag, d, r, t, f"{acc:.4f}"])
    return history, model.state_dict()


def pick_probe(sweep_results, default_d, default_r, min_ss=0.5):
    """Implements the "probe_d adjusted below based on sweep if needed" intent:
    keep the requested operating point if it is comfortably alive, else choose a
    clearly-alive, meaningfully-stressed NESS point so starvation/recovery start
    from a genuine steady state rather than a dead one."""
    if sweep_results.get((default_d, default_r), 0.0) > max(DEAD_THRESHOLD, min_ss):
        return default_d, default_r
    viable = [((d, r), ss) for (d, r), ss in sweep_results.items()
              if r > 0 and ss > min_ss]
    if viable:  # highest decay that still holds min_ss (tie-break higher repair)
        (d, r), _ = max(viable, key=lambda kv: (kv[0][0], kv[0][1]))
        return d, r
    alive = [((d, r), ss) for (d, r), ss in sweep_results.items() if r > 0]
    if alive:  # fall back to the single healthiest surviving point
        (d, r), _ = max(alive, key=lambda kv: kv[1])
        return d, r
    return default_d, default_r


def phase_sweep(healthy_state, train_loader, test_loader, args, log_writer,
                seed_offset=0):
    """Signature 1 + 3: sweep decay d and repair r, find the phase boundary."""
    print("\n[phase sweep] mapping the life/death boundary")
    results = {}
    for d in args.decay_rates:
        for r in args.repair_budgets:
            t0 = time.time()
            hist, _ = run_life(healthy_state, d, r, args.ticks, train_loader,
                               test_loader, args.eval_every, args.lr,
                               seed=(hash((d, r)) + seed_offset) % (2**31),
                               log_writer=log_writer,
                               tag="sweep")
            # steady-state accuracy: mean of last 25% of evals
            tail = [a for _, a in hist[max(1, int(len(hist) * 0.75)):]]
            ss = sum(tail) / len(tail)
            alive = ss > DEAD_THRESHOLD
            results[(d, r)] = ss
            print(f"  d={d:<8g} r={r:<3} steady-state acc={ss:.3f} "
                  f"{'ALIVE' if alive else 'dead '} ({time.time()-t0:.0f}s)")
    return results


def starvation_and_recovery(healthy_state, train_loader, test_loader, args,
                            log_writer, seed_offset=0):
    """Signature 2 + 4: pick a viable (d, r), starve for varying durations,
    then restore repair and see if the network recovers. Looks for a point
    of no return (hysteresis)."""
    d, r = args.probe_d, args.probe_r
    print(f"\n[starvation] probing at d={d}, r={r}")

    # First reach steady state
    print("  reaching steady state...")
    _, ss_state = run_life(healthy_state, d, r, args.ticks, train_loader,
                           test_loader, args.eval_every, args.lr,
                           seed=1234 + seed_offset,
                           log_writer=log_writer, tag="prelude")

    recovery_results = []
    for starve_ticks in args.starve_durations:
        # Starve: decay on, repair off
        print(f"  starving for {starve_ticks} ticks...", end=" ", flush=True)
        _, starved_state = run_life(ss_state, d, 0, starve_ticks, train_loader,
                                    test_loader, max(1, starve_ticks // 10),
                                    args.lr, seed=starve_ticks + seed_offset,
                                    log_writer=log_writer, tag="starve")
        model = MLP().to(DEVICE)
        model.load_state_dict(starved_state)
        acc_after_starve = evaluate(model, test_loader)

        # Attempt recovery: repair back on, generous duration
        rec_hist, _ = run_life(starved_state, d, r, args.recovery_ticks,
                               train_loader, test_loader, args.eval_every,
                               args.lr, seed=starve_ticks + 999 + seed_offset,
                               log_writer=log_writer, tag="recover")
        acc_after_recovery = rec_hist[-1][1]
        recovered = acc_after_recovery > 0.8 * max(0.5, acc_after_starve + 0.3)
        print(f"post-starve acc={acc_after_starve:.3f}  "
              f"post-recovery acc={acc_after_recovery:.3f}")
        recovery_results.append(
            (starve_ticks, acc_after_starve, acc_after_recovery))
        log_writer.writerow(["recovery_summary", d, r, starve_ticks,
                             f"{acc_after_starve:.4f}|{acc_after_recovery:.4f}"])
    return recovery_results


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def make_plots(sweep_results, recovery_results, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Phase diagram
    ds = sorted({d for d, _ in sweep_results})
    rs = sorted({r for _, r in sweep_results})
    grid = np.zeros((len(ds), len(rs)))
    for i, d in enumerate(ds):
        for j, r in enumerate(rs):
            grid[i, j] = sweep_results[(d, r)]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                   vmin=CHANCE, vmax=1.0)
    ax.set_xticks(range(len(rs)), [str(r) for r in rs])
    ax.set_yticks(range(len(ds)), [f"{d:g}" for d in ds])
    ax.set_xlabel("repair budget r (grad steps / tick)")
    ax.set_ylabel("decay rate d")
    ax.set_title("Phase diagram: steady-state accuracy\n"
                 "(sharp boundary = life/death transition)")
    fig.colorbar(im, label="steady-state accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "phase_diagram.png"), dpi=150)

    # Recovery / hysteresis
    if recovery_results:
        st, post_starve, post_rec = zip(*recovery_results)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(st, post_starve, "o-", label="acc after starvation")
        ax.plot(st, post_rec, "s-", label="acc after recovery attempt")
        ax.axhline(CHANCE, ls="--", c="gray", label="chance")
        ax.set_xlabel("starvation duration (ticks)")
        ax.set_ylabel("test accuracy")
        ax.set_title("Irrecoverability test\n"
                     "(recovery curve dropping to chance = point of no return)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "recovery.png"), dpi=150)
    print(f"\nPlots saved to {RESULTS_DIR}/")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="~15 min sanity run")
    args_cli = p.parse_args()

    class Args:
        pass
    args = Args()
    if args_cli.quick:
        args.ticks = 150
        args.eval_every = 15
        args.decay_rates = [0.005, 0.02, 0.08]
        args.repair_budgets = [0, 1, 4]
        args.starve_durations = [20, 60, 150]
        args.recovery_ticks = 150
        pre_epochs = 1
    else:
        args.ticks = 1500
        args.eval_every = 50
        args.decay_rates = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
        args.repair_budgets = [0, 1, 2, 4, 8, 16]
        args.starve_durations = [25, 50, 100, 200, 400, 800, 1600]
        args.recovery_ticks = 1500
        pre_epochs = 3
    args.lr = 1e-3
    args.probe_d = 0.02   # adjusted below based on sweep if needed
    args.probe_r = 4

    print(f"Device: {DEVICE}")
    train_loader, test_loader = get_data()
    model = pretrain(train_loader, test_loader, epochs=pre_epochs)
    healthy_state = copy.deepcopy(model.state_dict())
    healthy_acc = evaluate(model, test_loader)
    print(f"[pretrain] healthy ceiling: {healthy_acc:.4f}")

    log_path = os.path.join(RESULTS_DIR, "log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "d", "r", "tick", "acc"])
        sweep_results = phase_sweep(healthy_state, train_loader, test_loader,
                                    args, w)
        # Implement "probe_d adjusted below based on sweep if needed":
        args.probe_d, args.probe_r = pick_probe(sweep_results,
                                                args.probe_d, args.probe_r)
        print(f"[probe] operating point for starvation/recovery: "
              f"d={args.probe_d}, r={args.probe_r}")
        recovery_results = starvation_and_recovery(
            healthy_state, train_loader, test_loader, args, w)

    make_plots(sweep_results, recovery_results, args)

    # Verdict summary
    print("\n" + "=" * 60)
    print("SIGNATURE CHECKLIST")
    print("=" * 60)
    viable = [(k, v) for k, v in sweep_results.items() if v > DEAD_THRESHOLD
              and k[1] > 0]
    dead_no_repair = [v for (d, r), v in sweep_results.items() if r == 0]
    print(f"1. Steady state under throughput: "
          f"{'YES' if viable else 'NO'} "
          f"({len(viable)} viable (d,r) combos, best ss acc "
          f"{max(v for _, v in viable):.3f} vs ceiling {healthy_acc:.3f})"
          if viable else "1. Steady state under throughput: NO")
    print(f"2. Collapse on starvation: "
          f"{'YES' if dead_no_repair and max(dead_no_repair) < DEAD_THRESHOLD else 'PARTIAL/NO'} "
          f"(r=0 runs end at acc {max(dead_no_repair):.3f})"
          if dead_no_repair else "2. (no r=0 runs)")
    if recovery_results:
        recovered = [rec for _, _, rec in recovery_results]
        died = any(rec < DEAD_THRESHOLD + 0.1 for rec in recovered)
        always_fine = all(rec > 0.7 for rec in recovered)
        if died:
            print("4. Irrecoverability: YES - found a point of no return "
                  "(see recovery.png)")
        elif always_fine:
            print("4. Irrecoverability: NO - network always recovered. "
                  "Jeopardy claim NOT supported at these settings.")
        else:
            print("4. Irrecoverability: AMBIGUOUS - partial recovery. "
                  "Inspect recovery.png; consider longer starvation.")
    print("3. Phase transition sharpness: inspect phase_diagram.png "
          "(sharp color boundary vs smooth gradient)")
    print(f"\nRaw data: {log_path}")


if __name__ == "__main__":
    main()
