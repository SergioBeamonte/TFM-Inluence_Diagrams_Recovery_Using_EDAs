"""Estudio de CAPACIDAD: cuanta supervision necesita el EDA para que la POBLACION
resuelva el problema, comparando arranque en frio vs en caliente.

Idea
----
Para un nº creciente de reglas de entrenamiento k = 1..10 (nidadas: las k reglas
son las k-1 anteriores + 1 nueva), entrenamos el modelo y observamos que fraccion
de la POBLACION recupera TODAS las reglas reales (`pct_pop_full`, sobre el total,
no solo las de entrenamiento). Marcamos con estrella la generacion en que la
poblacion CRUZA el 50%. Si en algun momento se supera el 95%, se frena.

Dos metodos (misma secuencia de reglas, distinta inicializacion del EDA):
  - 'cold'  : cada k arranca con inicializacion ALEATORIA (uniforme). Hasta 100
              generaciones por k. Mide cuanta supervision hace falta partiendo
              de cero cada vez.
  - 'warm'  : cada k arranca REAJUSTANDO el EDA a la POBLACION FINAL del k-1
              (init_data de EDAspy). Hasta 30 generaciones por k. Mide si reusar
              lo aprendido con k-1 reglas acelera resolver con k.

Salida: example/explore_init_coldwarm.csv (1 fila por metodo x optimizador x
fitness x rep x k x generacion). Solo bypass2 (mode='both'), 4 EDAs x 5 fitness
x 10 reps. No sobreescribe el explore_capacity.csv antiguo.
"""
import os
import io
import time
import contextlib
import numpy as np
import pandas as pd

from id_recovery import IDRecovery


NET = {
    'name':      'bypass2',
    'xdsl_path': r'example\bypass2\network-bypass2.xdsl',
    'rules_csv': r'example\bypass2\reglas_generadas.csv',
    'mode':      'both',
}

OPTIMIZERS = ['umda', 'emna', 'egna', 'keda']
FITNESS_TYPES = ['binary', 'margin', 'softmax', 'regret', 'entropy']
SIZE_GEN_PER_OPT = {'umda': 50, 'emna': 50, 'egna': 50, 'keda': 400}

COMMON_KW = dict(
    u_range=(0, 10),
    min_max_ut=True,
    alpha=0.5,
    elite_factor=0.0,
    stop_mode='top50',          # informativo; aqui no usamos curriculum interno
    symmetric_sampling=False,
    chance_temperature=1.0,
    utility_temperature=1.0,
)

MAX_RULES   = 10        # k = 1..10
N_REPS      = 10
ITERS_COLD  = 100       # generaciones maximas por k (metodo en frio)
ITERS_WARM  = 30        # generaciones maximas por k (metodo en caliente)
CROSS_50    = 50.0      # estrella: la poblacion cruza este % de individuos perfectos
STOP_95     = 95.0      # parada: mas del 95% de la poblacion recupera todas las reglas
GLOBAL_FULL = 99.999    # un individuo "perfecto" recupera TODAS las reglas reales
BASE_SEED   = 42
TARGET_FITNESS = 1e-5

RAW_CSV = r'example\explore_init_coldwarm.csv'


def _pct_full(accs):
    """% de individuos que recuperan TODAS las reglas reales (accuracy global ~100)."""
    accs = np.asarray(accs)
    return float(np.sum(accs >= GLOBAL_FULL) / len(accs) * 100.0)


def _new_exp(optimizer, fitness_type, seed):
    """IDRecovery silencioso (n_decision_rules=-1; las reglas de train se fijan luego)."""
    with contextlib.redirect_stdout(io.StringIO()):
        exp = IDRecovery(
            xdsl_path=NET['xdsl_path'], rules_csv=NET['rules_csv'], mode=NET['mode'],
            fitness_type=fitness_type, optimizer_type=optimizer,
            n_decision_rules=-1, random_seed=seed, **COMMON_KW)
    return exp


def _set_train_rules(exp, rule_indices):
    """Fija las reglas de entrenamiento (subconjunto de all_rules) por indice."""
    exp.train_rules = [exp.all_rules[i] for i in rule_indices]
    id_set = {id(r) for r in exp.train_rules}
    exp.train_mask = [id(r) in id_set for r in exp.all_rules]


def _early_stop_at_95(exp):
    """Envuelve exp.fitness para frenar la corrida si la poblacion supera el 95%."""
    orig = exp.fitness

    def wrapped(v, _orig=orig, _exp=exp):
        val = _orig(v)
        if _exp.evals_this_gen == 0 and len(_exp.history) > 0:
            if _pct_full(_exp.history[-1]['accuracies']) >= STOP_95:
                raise StopIteration("La poblacion supera el 95% de individuos perfectos")
        return val
    exp.fitness = wrapped


def run_method(method, optimizer, fitness_type, rep):
    """Recorre k=1..10 para un metodo y devuelve filas por generacion."""
    seed = BASE_SEED + rep
    sg = SIZE_GEN_PER_OPT[optimizer]
    max_iter = ITERS_COLD if method == 'cold' else ITERS_WARM
    # Permutacion nidada de reglas, identica para 'cold' y 'warm' (mismo seed).
    perm = None
    prev_pop = None      # poblacion final del k anterior (solo se usa en 'warm')
    cum_gen = 0
    rows = []

    for k in range(1, MAX_RULES + 1):
        exp = _new_exp(optimizer, fitness_type, seed)
        total_rules = len(exp.all_rules)
        if perm is None:
            perm = np.random.default_rng(seed).permutation(total_rules)
        _set_train_rules(exp, perm[:k])
        _early_stop_at_95(exp)

        init_pop = None if method == 'cold' else prev_pop
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exp.run(g=sg, i=max_iter, target_fitness=TARGET_FITNESS,
                        patience=float('inf'), init_population=init_pop)
        except (ValueError, np.linalg.LinAlgError):
            pass  # p.ej. KEDA con covarianza singular: conservamos la historia parcial

        if not exp.history:
            break

        crossed = False
        stage_reached_95 = False
        for h in exp.history:
            accs = h['accuracies']
            pf = _pct_full(accs)
            mx = float(np.max(accs))
            star = (not crossed) and pf >= CROSS_50
            if star:
                crossed = True
            if pf >= STOP_95:
                stage_reached_95 = True
            rows.append({
                'method':         method,
                'net':            NET['name'],
                'optimizer':      optimizer,
                'fitness_type':   fitness_type,
                'rep':            rep,
                'seed':           seed,
                'size_gen':       sg,
                'mode':           NET['mode'],
                'total_rules':    total_rules,
                'n_train_rules':  k,
                'stage_gen':      int(h['gen']),
                'cum_gen':        cum_gen + int(h['gen']),
                'max_accuracy':   mx,
                'mean_accuracy':  float(np.mean(accs)),
                'n_rules_correct': int(round(mx / 100.0 * total_rules)),
                'pct_pop_full':   pf,
                'star_cross50':   bool(star),
                'gen_cpu_time':   float(h.get('gen_cpu_time', float('nan'))),
            })
        cum_gen += len(exp.history)
        prev_pop = exp.final_population        # semilla para el k siguiente (warm)

        if stage_reached_95:
            break        # la poblacion ya resuelve el problema: no hace falta mas supervision

    return rows


def load_existing():
    """Reanuda si el CSV ya tiene el formato nuevo (con columna 'method')."""
    if os.path.exists(RAW_CSV):
        df = pd.read_csv(RAW_CSV)
        if 'method' in df.columns:
            done = set(zip(df['method'], df['optimizer'], df['fitness_type'], df['rep']))
            return df.to_dict('records'), done
    return [], set()


def main():
    rows, done = load_existing()
    combos = [(m, o, f, r)
              for m in ('cold', 'warm')
              for o in OPTIMIZERS
              for f in FITNESS_TYPES
              for r in range(N_REPS)]
    total = len(combos)
    todo = total - len(done)
    print(f"Plan capacidad (cold/warm): {total} corridas ({todo} pendientes, {len(done)} hechas)")

    t0, completed_now = time.time(), 0
    for method, opt, fit, rep in combos:
        if (method, opt, fit, rep) in done:
            continue
        new_rows = run_method(method, opt, fit, rep)
        completed_now += 1
        # Resumen de la corrida.
        if new_rows:
            df_run = pd.DataFrame(new_rows)
            cross = df_run[df_run['star_cross50']]
            if not cross.empty:
                first = cross.iloc[0]
                cross_msg = f"cruza 50% en k={int(first['n_train_rules'])} (gen acum {int(first['cum_gen'])})"
            else:
                cross_msg = "no cruza 50%"
            reached95 = (df_run['pct_pop_full'] >= STOP_95).any()
            k_max = int(df_run['n_train_rules'].max())
            rows.extend(new_rows)
        else:
            cross_msg, reached95, k_max = "sin datos", False, 0
        elapsed = time.time() - t0
        eta = elapsed / completed_now * (todo - completed_now) if completed_now else 0
        print(f"[{completed_now}/{todo}] {method}/{opt}/{fit} rep={rep}: "
              f"{cross_msg} | k_max={k_max} | 95%={'si' if reached95 else 'no'}  "
              f"({elapsed/60:.1f}min / ETA {eta/60:.1f}min)")
        pd.DataFrame(rows).to_csv(RAW_CSV, index=False)

    print(f"\n=== DONE: {completed_now} corridas nuevas en {(time.time()-t0)/60:.1f} min ===")
    df = pd.DataFrame(rows)
    if df.empty:
        return
    # Resumen: en que k cruza la poblacion el 50%, cold vs warm.
    cross = df[df['star_cross50']].sort_values('cum_gen').groupby(
        ['method', 'optimizer', 'fitness_type', 'rep']).head(1)
    print("\n=== k medio en que la POBLACION cruza el 50% (cold vs warm) ===")
    print(cross.groupby(['method', 'optimizer'])['n_train_rules'].mean().round(1).to_string())


if __name__ == '__main__':
    main()
