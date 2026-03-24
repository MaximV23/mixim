"""
Parameter sweep:
  n_layers:          [1, 5, 10]
  n_mixes_per_layer: [1, 5, 10]
  corruption:        [0, 10, 20, 30, 50] % of total mixes
  n_runs:            3 per config  (for error bars)

Adaptive threshold: ceil(n_mixes / flush_percent)
  → guarantees flush_percent × threshold ≥ n_mixes, preventing pool
    starvation at deeper layers.

Figures produced:
  A  entropy vs % compromised  — one subplot per layer count, lines per mixes/layer
  B  entropy vs mixes/layer    — lines per layer count, panels for 0% and 30% corruption
  C  entropy vs layers         — lines per mixes/layer, panels for 0% and 30% corruption
  D  heatmap layers×mixes      — at 0% and 50% corruption
"""

import math
import time
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Simulation import Simulation
from util import Weights

# ── Fixed parameters ──────────────────────────────────────────────────────────
N_CLIENTS        = 100
LAMBDA_C         = 1.0           # 1 msg/s per client → ~100 msgs/s total
MIX_TYPE         = 'pool'
TOPOLOGY         = 'stratified'
FULLY_CONNECTED  = True
ROUTING          = 'source'
E2E_LATENCY      = 1.0
FLUSH_PERCENT    = 0.5           # 50% forwarded: keeps pools fed through all layers
FLUSH_TIMEOUT    = 2.0
SIM_DURATION     = 200           # long enough for deep topologies to stabilise

CLIENT_DUMMIES       = False
RATE_CLIENT_DUMMIES  = 1.0
LINK_DUMMIES         = False
MULTIPLE_HOP_DUMMIES = False
RATE_MIX_DUMMIES     = 1.0

N_RUNS = 3   # repetitions per config — used for error bars

# ── Sweep parameters ──────────────────────────────────────────────────────────
LAYERS_SWEEP  = [1, 5, 10]
MIXES_SWEEP   = [1, 5, 10]
CORRUPT_FRACS = [0.0, 0.1, 0.2, 0.3, 0.5]   # fraction of total mixes to corrupt

MAX_ENTROPY = math.log2(N_CLIENTS)   # ≈ 6.644 bits — full anonymity ceiling
LOG_DIR     = 'Logs/'


# ── Simulation helpers ────────────────────────────────────────────────────────

def threshold_for(n_mixes):
    """Adaptive pool threshold.

    Stability condition across layers:
        flush_percent × threshold ≥ n_mixes_per_layer
    → each layer receives at least 1 msg per flush from the layer above.
    Using ceiling ensures the condition holds strictly.
    """
    return max(10, math.ceil(n_mixes / FLUSH_PERCENT))


def run_single(params):
    """Run one simulation. Must run in its own OS process because
    Network.network_dict is a class-level variable (not reset between runs)."""
    n_layers, n_mixes, corrupt_count, run_idx = params
    total_mixes = n_layers * n_mixes

    if corrupt_count > total_mixes:
        return {
            'n_layers': n_layers, 'n_mixes': n_mixes,
            'corrupt_count': corrupt_count, 'run_idx': run_idx,
            'entropy_mean': None, 'entropy_within_std': None,
            'skipped': True, 'reason': 'corrupt > total mixes',
        }

    mu = max((E2E_LATENCY - (n_layers + 1) * 0.05) / n_layers, 0.01)
    threshold = threshold_for(n_mixes)
    weights   = Weights(n_layers, n_mixes)

    sim = Simulation(
        mix_type=MIX_TYPE, simDuration=SIM_DURATION,
        rate_client=1.0 / LAMBDA_C, mu=mu, logging=False,
        topology=TOPOLOGY, fully_connected=FULLY_CONNECTED,
        n_clients=N_CLIENTS, flush_percent=FLUSH_PERCENT, printing=False,
        flush_timeout=FLUSH_TIMEOUT, threshold=threshold, routing=ROUTING,
        n_layers=n_layers, n_mixes_per_layer=n_mixes,
        corrupt=corrupt_count, unifrom_corruption=True,
        probability_dist_mixes=weights, nbr_cascacdes=3,
        client_dummies=CLIENT_DUMMIES, rate_client_dummies=RATE_CLIENT_DUMMIES,
        link_based_dummies=LINK_DUMMIES, multiple_hops_dummies=MULTIPLE_HOP_DUMMIES,
        rate_mix_dummies=RATE_MIX_DUMMIES, Network_template=None,
    )

    try:
        entropy_list, entropy_mean, _, _ = sim.run()
    except Exception as exc:
        return {
            'n_layers': n_layers, 'n_mixes': n_mixes,
            'corrupt_count': corrupt_count, 'run_idx': run_idx,
            'entropy_mean': None, 'entropy_within_std': None,
            'skipped': True, 'reason': str(exc),
        }

    return {
        'n_layers':           n_layers,
        'n_mixes':            n_mixes,
        'corrupt_count':      corrupt_count,
        'corrupt_frac':       corrupt_count / total_mixes,
        'run_idx':            run_idx,
        'entropy_mean':       entropy_mean,
        'entropy_within_std': float(np.std(entropy_list)) if entropy_list else 0.0,
        'skipped':            False,
        'reason':             '',
    }


def build_configs():
    configs = []
    for n_layers, n_mixes, frac in itertools.product(LAYERS_SWEEP, MIXES_SWEEP, CORRUPT_FRACS):
        total          = n_layers * n_mixes
        corrupt_count  = min(int(round(frac * total)), total)
        for run_idx in range(N_RUNS):
            configs.append((n_layers, n_mixes, corrupt_count, run_idx))
    return configs


def run_all():
    configs = build_configs()
    n_unique = len(set((c[0], c[1], c[2]) for c in configs))
    print(f"Running {len(configs)} simulations  "
          f"({N_RUNS} runs × {n_unique} unique configs)")
    print(f"  FLUSH_PERCENT={FLUSH_PERCENT}, threshold=adaptive, "
          f"simDuration={SIM_DURATION}s, {N_CLIENTS} clients @ {LAMBDA_C} msg/s")

    t0 = time.time()
    with Pool(processes=4, maxtasksperchild=1) as pool:
        results = pool.map(run_single, configs, chunksize=1)
    print(f"Finished in {time.time() - t0:.1f}s\n")

    df = pd.DataFrame(results)
    df.to_csv(f'{LOG_DIR}plot_test_raw.csv', index=False)
    return df


def aggregate(df):
    valid = df[~df['skipped']].copy()
    valid['corrupt_frac'] = valid['corrupt_count'] / (valid['n_layers'] * valid['n_mixes'])

    agg = (valid
           .groupby(['n_layers', 'n_mixes', 'corrupt_count'])
           .agg(
               entropy_avg   = ('entropy_mean', 'mean'),
               # SEM across runs: std / sqrt(n) — shows run-to-run variability
               entropy_sem   = ('entropy_mean',
                                lambda x: x.std(ddof=1) / np.sqrt(len(x))
                                          if len(x) > 1 else 0.0),
               n_valid_runs  = ('entropy_mean', 'count'),
               corrupt_frac  = ('corrupt_frac', 'first'),
           )
           .reset_index())
    return agg


# ── Lookup helper ─────────────────────────────────────────────────────────────

def lookup(agg, n_layers, n_mixes, frac):
    """Return (entropy_avg, entropy_sem) for the config closest to frac."""
    total        = n_layers * n_mixes
    corrupt_count = min(int(round(frac * total)), total)
    row = agg[(agg['n_layers']      == n_layers)  &
              (agg['n_mixes']       == n_mixes)    &
              (agg['corrupt_count'] == corrupt_count)]
    if row.empty:
        return None, None
    return float(row['entropy_avg'].iloc[0]), float(row['entropy_sem'].iloc[0])


# ── Figure A — Entropy vs % compromised ──────────────────────────────────────

def fig_a(agg):
    """One subplot per layer count.  Lines = mixes/layer.
    Directly answers: can the GPA + corrupt-node adversary find the sender?"""
    fig, axes = plt.subplots(1, len(LAYERS_SWEEP), figsize=(14, 4.5), sharey=True)
    fig.suptitle(
        'Figure A — Entropy vs Fraction of Compromised Nodes\n'
        f'Pool mix · {N_CLIENTS} clients · {LAMBDA_C} msg/s · stratified topology',
        fontsize=10)

    colors  = {1: '#1f77b4', 5: '#ff7f0e', 10: '#2ca02c'}
    markers = {1: 'o',       5: 's',       10: '^'}

    for ax, n_layers in zip(axes, LAYERS_SWEEP):
        for n_mixes in MIXES_SWEEP:
            sub = agg[(agg['n_layers'] == n_layers) &
                      (agg['n_mixes']  == n_mixes)].sort_values('corrupt_frac')
            if sub.empty:
                continue
            x    = sub['corrupt_frac'] * 100
            y    = sub['entropy_avg']
            yerr = sub['entropy_sem']
            ax.plot(x, y,
                    marker=markers[n_mixes], color=colors[n_mixes],
                    linewidth=1.8, label=f'{n_mixes} mix/layer')
            ax.fill_between(x, y - yerr, y + yerr,
                            alpha=0.15, color=colors[n_mixes])

        ax.axhline(MAX_ENTROPY, color='dimgrey', linestyle=':', linewidth=1,
                   label='Max H (full anon)')
        ax.set_title(f'{n_layers} layer{"s" if n_layers > 1 else ""}', fontsize=10)
        ax.set_xlabel('Compromised mixes (%)')
        ax.set_ylim(bottom=-0.3)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc='upper right')

    axes[0].set_ylabel('Entropy  H  (bits)')
    fig.tight_layout()
    return fig


# ── Figure B — Entropy vs mixes/layer ────────────────────────────────────────

def fig_b(agg):
    """One subplot per corruption level (0% / 30%).  Lines = layer count.
    Answers: does making the network wider improve anonymity?"""
    show_fracs   = [0.0, 0.3]
    frac_labels  = ['0% corrupted', '30% corrupted']
    layer_colors = {1: '#1f77b4', 5: '#ff7f0e', 10: '#2ca02c'}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    fig.suptitle(
        'Figure B — Entropy vs Mixes per Layer\n'
        f'Pool mix · {N_CLIENTS} clients · {LAMBDA_C} msg/s',
        fontsize=10)

    for ax, frac, label in zip(axes, show_fracs, frac_labels):
        for n_layers in LAYERS_SWEEP:
            xs, ys, es = [], [], []
            for n_mixes in MIXES_SWEEP:
                y, e = lookup(agg, n_layers, n_mixes, frac)
                if y is not None:
                    xs.append(n_mixes)
                    ys.append(y)
                    es.append(e)
            if xs:
                ax.errorbar(xs, ys, yerr=es,
                            marker='o', color=layer_colors[n_layers],
                            linewidth=1.8, capsize=3,
                            label=f'{n_layers} layer{"s" if n_layers > 1 else ""}')

        ax.axhline(MAX_ENTROPY, color='dimgrey', linestyle=':', linewidth=1,
                   label='Max H')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Mixes per layer')
        ax.set_xticks(MIXES_SWEEP)
        ax.set_ylim(bottom=-0.3)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Entropy  H  (bits)')
    fig.tight_layout()
    return fig


# ── Figure C — Entropy vs layers ─────────────────────────────────────────────

def fig_c(agg):
    """One subplot per corruption level (0% / 30%).  Lines = mixes/layer.
    Answers: does making the network deeper improve anonymity?"""
    show_fracs  = [0.0, 0.3]
    frac_labels = ['0% corrupted', '30% corrupted']
    mix_colors  = {1: '#d62728', 5: '#9467bd', 10: '#8c564b'}
    mix_markers = {1: 'o',       5: 's',       10: '^'}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    fig.suptitle(
        'Figure C — Entropy vs Number of Layers\n'
        f'Pool mix · {N_CLIENTS} clients · {LAMBDA_C} msg/s',
        fontsize=10)

    for ax, frac, label in zip(axes, show_fracs, frac_labels):
        for n_mixes in MIXES_SWEEP:
            xs, ys, es = [], [], []
            for n_layers in LAYERS_SWEEP:
                y, e = lookup(agg, n_layers, n_mixes, frac)
                if y is not None:
                    xs.append(n_layers)
                    ys.append(y)
                    es.append(e)
            if xs:
                ax.errorbar(xs, ys, yerr=es,
                            marker=mix_markers[n_mixes], color=mix_colors[n_mixes],
                            linewidth=1.8, capsize=3,
                            label=f'{n_mixes} mix/layer')

        ax.axhline(MAX_ENTROPY, color='dimgrey', linestyle=':', linewidth=1,
                   label='Max H')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Number of layers')
        ax.set_xticks(LAYERS_SWEEP)
        ax.set_ylim(bottom=-0.3)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Entropy  H  (bits)')
    fig.tight_layout()
    return fig


# ── Figure D — Heatmap ────────────────────────────────────────────────────────

def fig_d(agg):
    """Heatmap: rows = layers, cols = mixes/layer, colour = entropy.
    Side-by-side at 0% and 50% corruption shows how the topology's
    anonymity surface collapses under a strong adversary."""
    fracs  = [0.0, 0.5]
    titles = ['0% corrupted', '50% corrupted']

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.suptitle(
        'Figure D — Entropy Heatmap  (layers × mixes/layer)\n'
        f'Pool mix · {N_CLIENTS} clients · {LAMBDA_C} msg/s',
        fontsize=10)

    vmax = MAX_ENTROPY + 1.5

    for ax, frac, title in zip(axes, fracs, titles):
        matrix = np.full((len(LAYERS_SWEEP), len(MIXES_SWEEP)), np.nan)
        for i, n_layers in enumerate(LAYERS_SWEEP):
            for j, n_mixes in enumerate(MIXES_SWEEP):
                y, _ = lookup(agg, n_layers, n_mixes, frac)
                if y is not None:
                    matrix[i, j] = y

        im = ax.imshow(matrix, vmin=0, vmax=vmax,
                       cmap='RdYlGn', aspect='auto')

        ax.set_xticks(range(len(MIXES_SWEEP)))
        ax.set_xticklabels(MIXES_SWEEP)
        ax.set_yticks(range(len(LAYERS_SWEEP)))
        ax.set_yticklabels(LAYERS_SWEEP)
        ax.set_xlabel('Mixes per layer')
        ax.set_ylabel('Layers')
        ax.set_title(title, fontsize=10)

        # Annotate cells
        for i in range(len(LAYERS_SWEEP)):
            for j in range(len(MIXES_SWEEP)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.1f}',
                            ha='center', va='center', fontsize=9,
                            color='black' if v > vmax * 0.3 else 'white')

        fig.colorbar(im, ax=ax, label='Entropy (bits)', shrink=0.85)

    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df_raw = run_all()

    n_skipped = df_raw['skipped'].sum()
    n_ok      = (~df_raw['skipped']).sum()
    print(f"Results: {n_ok} succeeded, {n_skipped} skipped/errored")

    agg = aggregate(df_raw)
    agg.to_csv(f'{LOG_DIR}plot_test_agg.csv', index=False)

    print("\nAggregated results:")
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(agg[['n_layers', 'n_mixes', 'corrupt_frac',
               'entropy_avg', 'entropy_sem', 'n_valid_runs']].to_string(index=False))

    print("\nGenerating figures …")
    figures = {
        'figA_entropy_vs_corruption.png': fig_a(agg),
        'figB_entropy_vs_mixes.png':      fig_b(agg),
        'figC_entropy_vs_layers.png':     fig_c(agg),
        'figD_heatmap.png':               fig_d(agg),
    }

    for fname, fig in figures.items():
        path = f'{LOG_DIR}{fname}'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved {path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
