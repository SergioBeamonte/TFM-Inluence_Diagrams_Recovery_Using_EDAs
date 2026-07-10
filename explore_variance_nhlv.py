"""Estudio de VARIANZA sobre NHLv (non-Hodgkin lymphoma), modo utility_only.

Análogo a explore_variance.py (bypass) pero sobre la red grande y difícil, con el
alcance recortado para caber en una noche (Plan B):

  - modo utility_only  -> las CPTs están FIJAS a su valor real; solo se buscan
    utilidades. Por eso NO hay MSE a las probabilidades (no es medible): aquí solo
    descomponemos la varianza del ACCURACY.
  - 10% de reglas (7 de 67), como el histórico de NHLv.
  - 3 conjuntos de reglas  x  3 semillas de init  x  5 fitness  x  {umda, emna} = 90 runs.
  - parada top90, min_iter=1, con CAP de generaciones (MAX_ITER=40) para acotar la
    cola de runs largos, que es lo que dispara el coste en NHLv (~18 s/gen).

Lanzar:
    python explore_variance_nhlv.py            # corre (reanudable)
    python explore_variance_nhlv.py --summary  # descomposición de varianza del accuracy
"""
import os
import sys
import time

import numpy as np
import pandas as pd

from id_recovery import IDRecovery

# ─── DISEÑO ─────────────────────────────────────────────────────────────────────
STOP_MODE = 'top90'
MIN_ITER  = 1
MAX_ITER  = 40         # CAP de generaciones (acota el coste en NHLv)
SIZE_GEN  = 50
TARGET_FITNESS = 1e-5

FITNESS_TYPES = ['binary', 'margin', 'softmax', 'regret', 'entropy']
OPTIMIZERS    = ['umda', 'emna']
RULE_SEEDS    = [0, 1, 2]              # 3 conjuntos de reglas
INIT_SEEDS    = [0, 1, 2]              # 3 inicializaciones uniformes
PCT           = 10                     # % de reglas (NHLv histórico)

NET = {
    'xdsl_path': r'example\nhlv1\network-nhlv1.xdsl',
    'rules_csv': r'example\nhlv1\reglas_generadas.csv',
    'mode':      'utility_only',
    'name':      'nhlv1_util',
}

COMMON = dict(
    min_max_ut=True, u_range=(0, 10), alpha=0.5, elite_factor=0.0,
    symmetric_sampling=False, chance_temperature=1.0, utility_temperature=1.0,
)

RAW_CSV = r'example\explore_variance_nhlv.csv'


def total_rules_of(net):
    probe = IDRecovery(xdsl_path=net['xdsl_path'], rules_csv=net['rules_csv'],
                       mode=net['mode'], n_decision_rules=-1,
                       fitness_type='binary', optimizer_type='umda', **COMMON)
    return len(probe.all_rules)


def n_rules_for_pct(total, pct):
    return max(1, int(round(pct / 100.0 * total)))


def run_one(net, opt, fit, rule_seed, init_seed, n_rules, total):
    exp = IDRecovery(
        xdsl_path=net['xdsl_path'], rules_csv=net['rules_csv'], mode=net['mode'],
        n_decision_rules=n_rules, stop_mode=STOP_MODE, optimizer_type=opt,
        fitness_type=fit, random_seed=init_seed, rule_seed=rule_seed, **COMMON,
    )
    try:
        exp.run(g=SIZE_GEN, i=MAX_ITER, target_fitness=TARGET_FITNESS, min_iter=MIN_ITER)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    !! {opt} {fit} rule={rule_seed} init={init_seed} aborted: {type(e).__name__}")
        return None
    if not exp.history:
        return None

    last = exp.history[-1]
    pop_acc = np.asarray(last['accuracies'], dtype=float)
    pop_fit = np.asarray(last['fitness'], dtype=float)
    gen_cpus = [float(h.get('gen_cpu_time', np.nan)) for h in exp.history]
    sat_acc = pop_acc[pop_fit <= 1e-9]

    return {
        'network': net['name'], 'optimizer': opt, 'fitness_type': fit,
        'rule_seed': rule_seed, 'init_seed': init_seed, 'size_gen': SIZE_GEN,
        'pct': PCT, 'n_decision_rules': n_rules, 'total_rules': total,
        'acc_best': float(np.max(pop_acc)), 'acc_mean': float(np.mean(pop_acc)),
        'acc_worst': float(np.min(pop_acc)),
        'sat_acc_mean': float(np.mean(sat_acc)) if sat_acc.size else float('nan'),
        'n_satisfiers': int(sat_acc.size),
        'stop_generation': int(last['gen']),
        'cpu_total': float(np.nansum(gen_cpus)) if gen_cpus else float('nan'),
    }


def load_existing():
    try:
        rows = pd.read_csv(RAW_CSV).to_dict('records') if os.path.exists(RAW_CSV) else []
    except Exception as e:
        print(f"  !! CSV ilegible ({e}); se empieza de cero.")
        return [], set()
    done = set((r['optimizer'], r['fitness_type'], r['rule_seed'], r['init_seed']) for r in rows)
    return rows, done


def _atomic_write(path, df):
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main():
    rows, done = load_existing()
    total = total_rules_of(NET)
    n_rules = n_rules_for_pct(total, PCT)
    print(f"Reglas totales NHLv: {total} · usando {PCT}% = {n_rules} reglas")

    plan = [(opt, fit, rs, iseed)
            for opt in OPTIMIZERS
            for fit in FITNESS_TYPES
            for rs in RULE_SEEDS
            for iseed in INIT_SEEDS]
    todo = [p for p in plan if p not in done]
    print(f"Plan total: {len(plan)} runs · hechos: {len(done)} · pendientes: {len(todo)}")

    t0 = time.time()
    for k, (opt, fit, rs, iseed) in enumerate(todo, 1):
        row = run_one(NET, opt, fit, rs, iseed, n_rules, total)
        if row is not None:
            rows.append(row)
            msg = (f"acc[w/m/b]={row['acc_worst']:.0f}/{row['acc_mean']:.0f}/{row['acc_best']:.0f} "
                   f"gen={row['stop_generation']} cpu={row['cpu_total']:.0f}s")
        else:
            msg = 'ABORTED'
        elapsed = time.time() - t0
        eta = elapsed / k * (len(todo) - k)
        print(f"[{k}/{len(todo)}] {opt} {fit} rule={rs} init={iseed}: {msg}  "
              f"({elapsed/60:.1f}min / ETA {eta/60:.1f}min)")
        _atomic_write(RAW_CSV, pd.DataFrame(rows))

    print(f"\n=== DONE en {(time.time()-t0)/60:.1f} min ===")


def summary():
    """Descomposición de la varianza del ACCURACY por (optimizer, fitness).
    (En utility_only no hay MSE a probabilidades: las CPTs están fijas.)"""
    df = pd.read_csv(RAW_CSV)
    out = []
    for (opt, fit), g in df.groupby(['optimizer', 'fitness_type']):
        piv = g.pivot_table('acc_mean', 'rule_seed', 'init_seed')  # reglas x init
        out.append({
            'optimizer': opt, 'fitness_type': fit, 'n': len(g),
            'acc_mean': piv.values.mean(),
            'acc_std_rules': piv.std(axis=0, ddof=1).mean(),   # entre conjuntos de reglas
            'acc_std_init':  piv.std(axis=1, ddof=1).mean(),   # entre semillas de init
            'acc_std_total': piv.values.flatten().std(ddof=1),
        })
    res = pd.DataFrame(out).sort_values(['optimizer', 'fitness_type'])
    pd.set_option('display.width', 200, 'display.max_columns', 30)
    print("\n=== Descomposición de la varianza del accuracy (NHLv utility_only, 10% reglas) ===\n")
    print(res.round(2).to_string(index=False))
    res.to_csv(r'example\explore_variance_nhlv_summary.csv', index=False)
    print("\nGuardado: example\\explore_variance_nhlv_summary.csv")


if __name__ == '__main__':
    if '--summary' in sys.argv:
        summary()
    else:
        main()
