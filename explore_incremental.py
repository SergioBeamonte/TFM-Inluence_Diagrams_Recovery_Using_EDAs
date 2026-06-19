"""Estudio: curriculum incremental de reglas.

Arrancamos con 1 regla, entrenamos hasta que el stop_mode (top80 por fitness)
se cumple, añadimos otra regla aleatoria y seguimos. Iteramos hasta cubrir
todas las reglas (o hasta agotar generaciones). La curva (accuracy vs gen)
debe tener forma de S por tramos, con "puntos gordos" justo en las
generaciones donde se introduce una nueva regla.

Salida: 1 fila por (red, eda, rep, gen) con accuracy, n_train_rules,
rule_added_after_gen y gen_cpu_time. La curva promedio entre reps y los
markers de adición se calculan a posteriori desde este CSV.

Plan: 2 redes × 4 EDAs × 10 reps = 80 corridas.
  - bypass2: mode='both', regret, symmetric, T=1, top80, sg per opt
  - nhlv1  : mode='utility_only', resto igual
"""
import os
import time
import numpy as np
import pandas as pd

from id_recovery import IDRecovery


NETS = {
    'bypass2': {
        'xdsl_path': r'example\bypass2\network-bypass2.xdsl',
        'rules_csv': r'example\bypass2\reglas_generadas.csv',
        'mode': 'both',
    },
    # nhlv1 descartado: tardaba demasiado. Solo corremos bypass2.
    # 'nhlv1': {
    #     'xdsl_path': r'example\nhlv1\network-nhlv1.xdsl',
    #     'rules_csv': r'example\nhlv1\reglas_generadas.csv',
    #     'mode': 'utility_only',
    # },
}

OPTIMIZERS = ['umda', 'emna', 'egna', 'keda']
FITNESS_TYPES = ['binary', 'margin', 'softmax', 'regret', 'entropy']
SIZE_GEN_PER_OPT = {'umda': 50, 'emna': 50, 'egna': 50, 'keda': 400}

COMMON = {
    'u_range':            (0, 10),
    'min_max_ut':         True,
    'alpha':              0.5,
    'elite_factor':       0.0,
    'stop_mode':          'top80',
    'symmetric_sampling': False,
    'chance_temperature': 1.0,
    'utility_temperature': 1.0,
}

# Damos margen para que entren todas las reglas (techo n_decision_rules=-1 → todas).
MAX_ITER       = 500
TARGET_FITNESS = 1e-5
N_REPS         = 10
BASE_SEED      = 42

RAW_CSV = r'example\explore_incremental.csv'


def run_one(net_name, optimizer, fitness_type, rep, seed):
    cfg = NETS[net_name]
    sg = SIZE_GEN_PER_OPT[optimizer]
    params = dict(COMMON)
    params.update({
        'xdsl_path': cfg['xdsl_path'],
        'rules_csv': cfg['rules_csv'],
        'mode':      cfg['mode'],
        'fitness_type':             fitness_type,
        'n_decision_rules':         -1,       # pool = todas las reglas
        'optimizer_type':           optimizer,
        'random_seed':              seed,
        'incremental_rules':        True,
        'incremental_start_with':   1,
    })
    exp = IDRecovery(**params)
    
    original_fitness = exp.fitness
    consecutive_success = [0]
    def custom_fitness(v):
        val = original_fitness(v)
        if exp.evals_this_gen == 0 and len(exp.history) > 0:
            accs = exp.history[-1]['accuracies']
            perfect_ratio = np.sum(accs >= 99.999) / len(accs)
            if perfect_ratio >= 0.999:
                consecutive_success[0] += 1
                if consecutive_success[0] >= 3:
                    raise StopIteration("El 100% de los individuos aciertan todas las reglas (3 gens seguidas)")
            else:
                consecutive_success[0] = 0
        return val
    exp.fitness = custom_fitness

    try:
        exp.run(g=sg, i=MAX_ITER, target_fitness=TARGET_FITNESS, patience=float('inf'))
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    !! {net_name}/{optimizer} rep={rep} ABORT: "
              f"{type(e).__name__}: {e}")
        return []
    if not exp.history:
        return []

    rows = []
    total_rules = len(exp.all_rules)
    for h in exp.history:
        accs = h['accuracies']
        rows.append({
            'net':            net_name,
            'optimizer':      optimizer,
            'rep':            rep,
            'fitness_type':   fitness_type,
            'seed':           seed,
            'size_gen':       sg,
            'mode':           cfg['mode'],
            'gen':            int(h['gen']),
            'n_train_rules':  int(h['n_train_rules']),
            'rule_added_after_gen': bool(h.get('rule_added_after_gen', False)),
            'max_accuracy':   float(np.max(accs)),
            'n_rules_correct': int(round(np.max(accs) / 100.0 * total_rules)),
            'mean_accuracy':  float(np.mean(accs)),
            'pct_success_indv': float(np.sum(accs >= 99.999) / len(accs) * 100.0),
            'gen_cpu_time':   float(h.get('gen_cpu_time', float('nan'))),
        })
    return rows


def load_existing():
    if os.path.exists(RAW_CSV):
        df = pd.read_csv(RAW_CSV)
        # Compat con CSVs viejos sin fitness_type: tratarlos como regret.
        if 'fitness_type' not in df.columns:
            df['fitness_type'] = 'regret'
        done = set(zip(df['net'], df['optimizer'], df['fitness_type'], df['rep']))
        return df.to_dict('records'), done
    return [], set()


def main():
    rows, done = load_existing()
    combos = [(n, o, f, r) for n in NETS
                            for o in OPTIMIZERS
                            for f in FITNESS_TYPES
                            for r in range(N_REPS)]
    total = len(combos)
    todo  = total - len(done)
    print(f"Plan: {total} corridas ({todo} pendientes, {len(done)} hechas)")
    print(f"Filas en CSV previo: {len(rows)}")

    t0 = time.time()
    completed_now = 0
    for net_name, opt, fit, rep in combos:
        if (net_name, opt, fit, rep) in done:
            continue
        seed = BASE_SEED + rep
        new_rows = run_one(net_name, opt, fit, rep, seed)
        completed_now += 1
        if new_rows:
            rows.extend(new_rows)
            last = new_rows[-1]
            final_acc = last['mean_accuracy']
            final_k   = last['n_train_rules']
            n_gens    = last['gen']
            n_events  = sum(1 for r in new_rows if r['rule_added_after_gen'])
        else:
            final_acc, final_k, n_gens, n_events = float('nan'), 0, 0, 0
        elapsed = time.time() - t0
        eta = elapsed / completed_now * (todo - completed_now) if completed_now else 0
        print(f"[{completed_now}/{todo}] {net_name}/{opt}/{fit} rep={rep}: "
              f"gens={n_gens} reglas={final_k} eventos={n_events} mean_acc={final_acc:.0f}%  "
              f"({elapsed/60:.1f}min / ETA {eta/60:.1f}min)")
        pd.DataFrame(rows).to_csv(RAW_CSV, index=False)

    print(f"\n=== DONE: {completed_now} corridas nuevas en {(time.time()-t0)/60:.1f} min ===")
    df = pd.DataFrame(rows)
    if df.empty:
        return
    summary = (df.sort_values('gen').groupby(['net','optimizer','fitness_type','rep']).tail(1)
                 [['net','optimizer','fitness_type','rep','gen','n_train_rules','max_accuracy','mean_accuracy','pct_success_indv']])
    print("\n=== Resumen final por corrida ===")
    print(summary.groupby(['net','optimizer','fitness_type'])
                 [['gen','n_train_rules','max_accuracy','mean_accuracy','pct_success_indv']]
                 .mean().round(1).to_string())


if __name__ == '__main__':
    main()
