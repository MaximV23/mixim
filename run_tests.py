"""
Mix-net simulation test suite.

Parameter sweep:
  - mix_type       : pool (fixed)
  - n_clients      : 100  (fixed)
  - lambda_c       : 1 msg/s (fixed)
  - n_layers       : 1, 5, 10
  - n_mixes_per_layer : 1, 10
  - corrupt_mixes  : 0, 1, 3

Threat model: GPA + compromised mix nodes.
Metric: Shannon entropy of the adversary's sender distribution.
  - max_entropy = log2(n_clients) = log2(100) ≈ 6.644 bits  → full anonymity
  - entropy = 0                                              → sender identified
"""

import time
import math
import itertools
from multiprocessing import Pool
import pandas as pd

from Simulation import Simulation
from util import Weights

# ── Fixed parameters ──────────────────────────────────────────────────────────
N_CLIENTS       = 100
LAMBDA_C        = 1.0          # 1 message per time unit
MIX_TYPE        = 'pool'
TOPOLOGY        = 'stratified'
FULLY_CONNECTED = True
ROUTING         = 'source'
E2E_LATENCY     = 1.0          # target end-to-end latency budget
THRESHOLD       = 100          # pool fires when it holds this many messages
FLUSH_PERCENT   = 0.1          # fraction forwarded on each flush
FLUSH_TIMEOUT   = 2.0
SIM_DURATION    = 50

# Dummies disabled for a clean baseline measurement
CLIENT_DUMMIES       = False
RATE_CLIENT_DUMMIES  = 1.0
LINK_DUMMIES         = False
MULTIPLE_HOP_DUMMIES = False
RATE_MIX_DUMMIES     = 1.0

# ── Sweep parameters ──────────────────────────────────────────────────────────
LAYERS_SWEEP          = [1, 5, 10]
MIXES_PER_LAYER_SWEEP = [1, 10]
CORRUPT_SWEEP         = [0, 1, 3]

MAX_ENTROPY = math.log2(N_CLIENTS)   # ≈ 6.644 bits


def run_single(params):
    """Run one simulation configuration. Must execute in its own process
    because Network uses class-level dicts that are not reset between runs."""
    n_layers, n_mixes_per_layer, corrupt_mixes = params

    total_mixes = n_layers * n_mixes_per_layer
    # Skip invalid configs where corrupt count exceeds available mixes
    if corrupt_mixes > total_mixes:
        return {
            'n_layers':          n_layers,
            'n_mixes_per_layer': n_mixes_per_layer,
            'corrupt_mixes':     corrupt_mixes,
            'entropy_mean':      None,
            'entropy_median':    None,
            'entropy_q25':       None,
            'anonymity_ratio':   None,
            'adversary_wins':    None,
            'skipped':           True,
            'reason':            f'corrupt ({corrupt_mixes}) > total mixes ({total_mixes})',
        }

    # mu: per-hop delay derived from E2E budget minus link delays
    mu = (E2E_LATENCY - (n_layers + 1) * 0.05) / n_layers
    if mu <= 0:
        mu = 0.01   # floor to keep simulation runnable

    weights = Weights(n_layers, n_mixes_per_layer)

    sim = Simulation(
        mix_type=MIX_TYPE,
        simDuration=SIM_DURATION,
        rate_client=1.0 / LAMBDA_C,
        mu=mu,
        logging=False,
        topology=TOPOLOGY,
        fully_connected=FULLY_CONNECTED,
        n_clients=N_CLIENTS,
        flush_percent=FLUSH_PERCENT,
        printing=False,
        flush_timeout=FLUSH_TIMEOUT,
        threshold=THRESHOLD,
        routing=ROUTING,
        n_layers=n_layers,
        n_mixes_per_layer=n_mixes_per_layer,
        corrupt=corrupt_mixes,
        unifrom_corruption=True,
        probability_dist_mixes=weights,
        nbr_cascacdes=3,
        client_dummies=CLIENT_DUMMIES,
        rate_client_dummies=RATE_CLIENT_DUMMIES,
        link_based_dummies=LINK_DUMMIES,
        multiple_hops_dummies=MULTIPLE_HOP_DUMMIES,
        rate_mix_dummies=RATE_MIX_DUMMIES,
        Network_template=None,
    )

    t0 = time.time()
    try:
        entropy_list, entropy_mean, entropy_median, entropy_q25 = sim.run()
    except Exception as exc:
        return {
            'n_layers':          n_layers,
            'n_mixes_per_layer': n_mixes_per_layer,
            'corrupt_mixes':     corrupt_mixes,
            'entropy_mean':      None,
            'entropy_median':    None,
            'entropy_q25':       None,
            'anonymity_ratio':   None,
            'adversary_wins':    None,
            'elapsed_s':         round(time.time() - t0, 2),
            'skipped':           True,
            'reason':            f'ERROR: {exc}',
        }
    elapsed = time.time() - t0

    anonymity_ratio = entropy_mean / MAX_ENTROPY if entropy_mean is not None else None
    # "adversary wins" heuristic: mean entropy < 1 bit  →  near-certain identification
    adversary_wins = entropy_mean < 1.0 if entropy_mean is not None else None

    return {
        'n_layers':          n_layers,
        'n_mixes_per_layer': n_mixes_per_layer,
        'corrupt_mixes':     corrupt_mixes,
        'entropy_mean':      round(entropy_mean, 4),
        'entropy_median':    round(entropy_median, 4),
        'entropy_q25':       round(entropy_q25, 4),
        'anonymity_ratio':   round(anonymity_ratio, 4),
        'adversary_wins':    adversary_wins,
        'elapsed_s':         round(elapsed, 2),
        'skipped':           False,
        'reason':            '',
    }


def main():
    configs = list(itertools.product(
        LAYERS_SWEEP,
        MIXES_PER_LAYER_SWEEP,
        CORRUPT_SWEEP,
    ))

    print(f"Running {len(configs)} configurations …")
    print(f"  mix_type={MIX_TYPE}, clients={N_CLIENTS}, λ={LAMBDA_C} msg/s")
    print(f"  layers={LAYERS_SWEEP}, mixes/layer={MIXES_PER_LAYER_SWEEP}, corrupt={CORRUPT_SWEEP}")
    print(f"  max_entropy = log2({N_CLIENTS}) = {MAX_ENTROPY:.4f} bits\n")

    # maxtasksperchild=1 forces a fresh process per task, which is required
    # because Network.network_dict is a class-level variable.
    with Pool(processes=4, maxtasksperchild=1) as pool:
        results = pool.map(run_single, configs, chunksize=1)

    df = pd.DataFrame(results)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n═══════════════════════════════════════════════════════════════════")
    print("                        RESULTS SUMMARY")
    print("═══════════════════════════════════════════════════════════════════")
    print(f"{'layers':>8} {'mix/layer':>10} {'corrupt':>8} │ "
          f"{'H_mean':>8} {'H_median':>9} {'H_q25':>7} │ "
          f"{'H/H_max':>8} {'adv_wins':>9}")
    print("─" * 75)

    for _, row in df.iterrows():
        if row['skipped']:
            print(f"{int(row['n_layers']):>8} {int(row['n_mixes_per_layer']):>10} "
                  f"{int(row['corrupt_mixes']):>8} │  SKIPPED ({row['reason']})")
        else:
            flag = " ◄ WINS" if row['adversary_wins'] else ""
            print(f"{int(row['n_layers']):>8} {int(row['n_mixes_per_layer']):>10} "
                  f"{int(row['corrupt_mixes']):>8} │ "
                  f"{row['entropy_mean']:>8.4f} {row['entropy_median']:>9.4f} "
                  f"{row['entropy_q25']:>7.4f} │ "
                  f"{row['anonymity_ratio']:>8.4f} {str(row['adversary_wins']):>9}{flag}")

    print("═" * 75)
    print(f"\nmax_entropy (full anonymity) = {MAX_ENTROPY:.4f} bits")
    print("adversary_wins = True when mean entropy < 1.0 bit\n")

    # ── Save to CSV ───────────────────────────────────────────────────────────
    out_path = 'Logs/test_results.csv'
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
