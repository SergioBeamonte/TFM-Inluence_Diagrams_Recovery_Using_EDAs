"""Mini-estudio: efecto del tamaño de población (size_gen) en los 4 optimizadores.

Comparativa controlada con resto de hiperparámetros fijos para aislar el efecto
de la población. Mismo criterio de parada (top50) para todos.

Salidas:
  - example/bypass2/explore_size_gen.csv                       (raw, 1 fila por rep)
  - example/bypass2/grid_search_results_<OPT>_sizegen.csv      (agregado dashboard)

Lanzar:
    python explore_size_gen.py
"""
import os
import time

import numpy as np
import pandas as pd

from id_recovery import IDRecovery


CONFIG_BASE = {
    'xdsl_path':  r'example\bypass2\network-bypass2.xdsl',
    'rules_csv':  r'example\bypass2\reglas_generadas.csv',
    'min_max_ut': True,
    'u_range':    (0, 10),
    'alpha':      0.5,
    'elite_factor': 0.0,
    'n_decision_rules':     4,        # 20% de 20 reglas
    'fitness_type':         'regret',
    'stop_mode':            'top50',
    'mode':                 'both',
    'symmetric_sampling':   True,
    'chance_temperature':   1.0,
    'utility_temperature':  1.0,
}

# Ordenados de más rápido a más lento (para tener resultados parciales antes)
OPTIMIZERS = ['umda', 'keda', 'emna', 'egna']
SIZES      = [30, 50, 100, 200]
N_REPS     = 10
MAX_ITER   = 60
TARGET_FITNESS = 1e-5
BASE_SEED  = 42

RAW_CSV = r'example\bypass2\explore_size_gen.csv'

CONSTANTS_FOR_AGG = {
    'mode':                 'both',
    'sampling_mode':        'symmetric',
    'fitness_type':         'regret',
    'stop_mode':            'top50',
    'n_decision_rules':     4,
    'n_decision_rules_pct': 20,
    'total_rules':          20,
    'chance_temperature':   1.0,
    'utility_temperature':  1.0,
}


def run_one(optimizer, size_gen, rep, seed):
    exp = IDRecovery(**CONFIG_BASE, optimizer_type=optimizer, random_seed=seed)
    try:
        exp.run(g=size_gen, i=MAX_ITER, target_fitness=TARGET_FITNESS)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    !! {optimizer} sg={size_gen} rep={rep} aborted: {type(e).__name__}: {e}")
        return None
    if not exp.history:
        return None
    last = exp.history[-1]
    return {
        'optimizer':       optimizer,
        'size_gen':        size_gen,
        'rep':             rep,
        'seed':            seed,
        'stop_generation': last['gen'],
        'best_fitness':    float(np.min(last['fitness'])),
        'best_accuracy':   float(np.max(last['accuracies'])),
        'mean_accuracy':   float(np.mean(last['accuracies'])),
        'mse_chance':      float(np.min(last['errors_chance'])),
        'mse_utility':     float(np.min(last['errors_utility'])),
    }


def load_existing():
    if os.path.exists(RAW_CSV):
        df = pd.read_csv(RAW_CSV)
        done = set(zip(df['optimizer'], df['size_gen'], df['rep']))
        return df.to_dict('records'), done
    return [], set()


def aggregate_and_save(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    for opt in OPTIMIZERS:
        sub = df[df['optimizer'] == opt]
        if sub.empty:
            continue
        agg_rows = []
        for sg, g in sub.groupby('size_gen'):
            row = dict(CONSTANTS_FOR_AGG)
            row['size_gen'] = int(sg)
            row['stop_gen_mean'] = g['stop_generation'].mean()
            row['stop_gen_std']  = g['stop_generation'].std()
            row['stop_gen_min']  = g['stop_generation'].min()
            row['stop_gen_max']  = g['stop_generation'].max()
            row['fitness_mean']  = g['best_fitness'].mean()
            row['fitness_std']   = g['best_fitness'].std()
            row['accuracy_mean'] = g['best_accuracy'].mean()
            row['accuracy_std']  = g['best_accuracy'].std()
            row['accuracy_min']  = g['best_accuracy'].min()
            row['accuracy_max']  = g['best_accuracy'].max()
            row['mse_chance_mean']    = g['mse_chance'].mean()
            row['mse_chance_std']     = g['mse_chance'].std()
            row['mse_utility_mean']   = g['mse_utility'].mean()
            row['mse_utility_std']    = g['mse_utility'].std()
            row['entropy_norm_mean']  = float('nan')
            row['entropy_norm_std']   = float('nan')
            row['util_dev_mean']      = float('nan')
            row['util_dev_std']       = float('nan')
            row['n_reps_completed']   = len(g)   # informativo
            agg_rows.append(row)
        out_path = f'example/bypass2/grid_search_results_{opt.upper()}_sizegen.csv'
        pd.DataFrame(agg_rows).to_csv(out_path, index=False)


def main():
    rows, done = load_existing()
    print(f"Cargados {len(rows)} runs previos. Combinaciones ya hechas: {len(done)}")

    total = len(OPTIMIZERS) * len(SIZES) * N_REPS
    todo = total - len(done)
    print(f"Plan total: {total} runs ({todo} pendientes).")

    t0 = time.time()
    completed_now = 0

    for opt in OPTIMIZERS:
        for sg in SIZES:
            for rep in range(N_REPS):
                key = (opt, sg, rep)
                if key in done:
                    continue
                seed = BASE_SEED + rep
                r = run_one(opt, sg, rep, seed)
                completed_now += 1
                if r is not None:
                    rows.append(r)
                    acc = r['best_accuracy']
                    gen = r['stop_generation']
                else:
                    acc, gen = float('nan'), 0
                elapsed = time.time() - t0
                eta = elapsed / completed_now * (todo - completed_now) if completed_now else 0
                print(f"[{completed_now}/{todo}] {opt} sg={sg} rep={rep}: "
                      f"acc={acc:.0f}% gen={gen}  "
                      f"({elapsed/60:.1f}min / ETA {eta/60:.1f}min)")
                # Persist after every run (raw + agregados)
                pd.DataFrame(rows).to_csv(RAW_CSV, index=False)
                if completed_now % 5 == 0:
                    aggregate_and_save(rows)

    # Final
    aggregate_and_save(rows)
    print(f"\n=== DONE: {completed_now} runs nuevos en {(time.time() - t0) / 60:.1f} min ===")

    df = pd.DataFrame(rows)
    print("\nAccuracy mean por (optimizer, size_gen):")
    print(df.pivot_table(values='best_accuracy', index='optimizer',
                          columns='size_gen', aggfunc='mean').round(2))
    print("\nStop generation mean por (optimizer, size_gen):")
    print(df.pivot_table(values='stop_generation', index='optimizer',
                          columns='size_gen', aggfunc='mean').round(2))


if __name__ == '__main__':
    main()
