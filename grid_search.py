"""
Grid Search sistemático para IDRecovery.

Ejecuta un barrido de hiperparámetros (n_decision_rules, fitness_type, stop_mode)
con N repeticiones por combinación, y guarda:
  - grid_search_results.csv:  estadísticas agregadas (mean/std/min/max) por combinación
  - grid_search_curves.csv:   curvas de evolución promediadas por combinación

Uso:
    py grid_search.py
    py grid_search.py --optimizer_type egna
    py grid_search.py --xdsl_path ruta/red.xdsl --rules_csv ruta/reglas.csv --min_max_ut True --optimizer_type umda
"""

import argparse
import os
import sys
import csv
import time
import itertools
import multiprocessing as mp
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN  — Modifica aquí los parámetros del grid
# ═══════════════════════════════════════════════════════════════════════════════

# --- Red y reglas ---
BASE_FOLDER = r"example\nhlv1"
# BASE_FOLDER = r"example\bypass2"

# XDSL_PATH = os.path.join(BASE_FOLDER, r"network-bypass2.xdsl")
XDSL_PATH = os.path.join(BASE_FOLDER, r"network-nhlv1.xdsl")
RULES_CSV = os.path.join(BASE_FOLDER, r"reglas_generadas.csv")

# --- Parámetros fijos del optimizador ---
BASE_CONFIG = {
    'xdsl_path': XDSL_PATH,
    'rules_csv': RULES_CSV,
    'min_max_ut': True,
    'u_range': (0, 10),
    'alpha': 0.5,
    'elite_factor': 0.0,
    'optimizer_type': 'umda',
}

# --- Parámetros del optimizador ---
SIZE_GEN = 50       # tamaño de población por generación
MAX_ITER = 60       # máximo de generaciones
TARGET_FITNESS = 1e-5
# KEDA necesita samples > variables para que gaussian_kde no sea singular. Con
# bypass2 mode='both' tenemos 95 vars; truncation_length (SIZE_GEN*alpha) debe
# ser > 95. Con sg=200 (trunc=100) seguía abortando ~46% de runs por degeneración
# de la población tardía. Subimos a 400 (trunc=200) para dejar margen sobrado.
SIZE_GEN_PER_OPTIMIZER = {'umda': 50, 'egna': 50, 'emna': 50, 'keda': 400}

# --- Grid de búsqueda ---
# GRID INICIAL del modelo pequeño: barrido amplio para evaluar el método y aislar
# después con grids específicos (temps, optimizadores). Lanzar una vez por
# optimizer_type para acumular CSVs separados por optimizador.
RULES_PERCENTAGES = [5, 10, 20, 40]
FITNESS_TYPES = ["binary", "margin", "softmax", "regret", "entropy"]
STOP_MODES = ["top50"]
CHANCE_TEMPERATURES = [1.0]
UTILITY_TEMPERATURES = [1.0]
MODES = ['utility_only']
SAMPLING_MODES = ['non_symmetric']

# --- Repeticiones ---
N_REPETITIONS = 5
BASE_SEED = 42

# --- Archivos de salida ---
RESULTS_CSV = os.path.join(BASE_FOLDER, f"grid_search_results_{BASE_CONFIG['optimizer_type']}.csv")
CURVES_CSV = os.path.join(BASE_FOLDER, f"grid_search_curves_{BASE_CONFIG['optimizer_type']}.csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════════

def count_total_rules(rules_csv_path):
    """Cuenta el número total de reglas en el CSV."""
    with open(rules_csv_path, 'r') as f:
        return sum(1 for _ in csv.DictReader(f))


def get_completed_combinations(results_csv_path):
    """Lee el CSV de resultados y devuelve un set de combinaciones ya completadas.

    Filas antiguas sin columnas nuevas se consideran T=1.0 y mode='both' para que
    runs históricos no se reejecuten al introducir nuevas dimensiones del grid.
    """
    completed = set()
    if not os.path.exists(results_csv_path):
        return completed
    with open(results_csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ct = row.get('chance_temperature') or '1.0'
            ut = row.get('utility_temperature') or '1.0'
            md = row.get('mode') or 'both'
            sm = row.get('sampling_mode') or 'non_symmetric'
            key = (md, sm, ct, ut, row['fitness_type'], row['stop_mode'], row['n_decision_rules'])
            completed.add(key)
    return completed


def compute_n_rules(total_rules, percentage):
    """Calcula el número de reglas a partir de un porcentaje."""
    return max(1, round(total_rules * percentage / 100))


def run_single_experiment(config, g, i, target_fitness):
    """Ejecuta un único IDRecovery.run() y devuelve el objeto y resultados.

    Las métricas finales se calculan sobre la ÚLTIMA generación registrada:
    - best_*: mejor individuo en esa generación (np.min/np.max segun la metrica)
    - mean_*: media sobre toda la poblacion de esa generacion

    Si la EDA lanza una excepción (p.ej. KEDA con menos muestras que dimensiones
    → matriz de covarianza singular en gaussian_kde) la combinación se marca con
    NaN para que el grid siga adelante en vez de tumbarse entero.
    """
    from id_recovery import IDRecovery

    exp = IDRecovery(**config)
    try:
        best_vector = exp.run(g=g, i=i, target_fitness=target_fitness)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    !! Run abortado por error del optimizador: {type(e).__name__}: {e}")
        # Fabricamos un exp vacío con history=[] para que el caller lo trate como NaN.
        exp.history = []
        return exp, {
            'stop_generation':    0,
            'best_fitness':       float('nan'),
            'best_accuracy':      float('nan'),
            'best_mse_chance':    float('nan'),
            'best_mse_utility':   float('nan'),
            'best_entropy_norm':  float('nan'),
            'best_util_dev':      float('nan'),
            'cpu_per_gen':        float('nan'),
            'cpu_total':          float('nan'),
        }

    # Extraer métricas de la última generación del historial
    if exp.history:
        last_gen = exp.history[-1]
        stop_generation = last_gen['gen']
        best_fitness   = float(np.mean(last_gen['fitness']))
        best_accuracy  = float(np.max(last_gen['accuracies']))
        best_mse_chance   = float(np.min(last_gen['errors_chance']))
        best_mse_utility  = float(np.min(last_gen['errors_utility']))
        best_entropy_norm = float(np.min(last_gen['entropy_norm']))
        best_util_dev     = float(np.max(last_gen['util_dev']))
        # Tiempo de CPU por generación (segundos). gen_cpu_time mide el tiempo
        # de CPU consumido en muestreo + fitness de toda la población. Útil
        # para comparar el coste computacional real de cada EDA sin que un
        # mayor size_gen oculte el tiempo extra por iteración, y sin que la
        # contención con otros procesos contamine la medida.
        gen_cpus = [float(h.get('gen_cpu_time', float('nan'))) for h in exp.history]
        cpu_per_gen = float(np.nanmean(gen_cpus)) if gen_cpus else float('nan')
        cpu_total = float(np.nansum(gen_cpus)) if gen_cpus else float('nan')
    else:
        stop_generation = 0
        best_fitness      = float(exp.best_historical_fitness) if exp.best_historical_fitness != float('inf') else float('nan')
        best_accuracy     = float('nan')
        best_mse_chance   = float('nan')
        best_mse_utility  = float('nan')
        best_entropy_norm = float('nan')
        best_util_dev     = float('nan')
        cpu_per_gen       = float('nan')
        cpu_total         = float('nan')

    results = {
        'stop_generation':    stop_generation,
        'best_fitness':       best_fitness,
        'best_accuracy':      best_accuracy,
        'best_mse_chance':    best_mse_chance,
        'best_mse_utility':   best_mse_utility,
        'best_entropy_norm':  best_entropy_norm,
        'best_util_dev':      best_util_dev,
        'cpu_per_gen':        cpu_per_gen,
        'cpu_total':          cpu_total,
    }

    return exp, results


def average_histories(experiments):
    """
    Promedia los historiales de N experimentos (igual que BatchExperimenter._average_histories).
    Devuelve una lista de dicts con las curvas promediadas — un punto por generación.

    Para cada generación se promedia primero sobre la poblacion (dentro de cada experimento)
    y despues sobre los experimentos. Asi cada curva representa la evolucion media del batch.
    """
    # Ignoramos experimentos con history vacío (corridas que fallaron por error
    # del optimizador). Si TODAS fallaron, devolvemos curvas vacías.
    experiments = [e for e in experiments if e.history]
    if not experiments:
        return []
    max_gens = max(len(exp.history) for exp in experiments)
    avg_history = []

    for gen_idx in range(max_gens):
        gen_fitness_list = []
        gen_err_chance_list = []
        gen_err_utility_list = []
        gen_accs_list = []
        gen_ent_list = []
        gen_dev_list = []
        gen_cpu_list = []

        for exp in experiments:
            hist_idx = min(gen_idx, len(exp.history) - 1)
            gen_data = exp.history[hist_idx]

            # Tomamos la media de la población para esa generación
            gen_fitness_list.append(float(np.mean(gen_data['fitness'])))
            gen_err_chance_list.append(float(np.mean(gen_data['errors_chance'])))
            gen_err_utility_list.append(float(np.mean(gen_data['errors_utility'])))
            gen_accs_list.append(float(np.mean(gen_data['accuracies'])))
            gen_ent_list.append(float(np.mean(gen_data['entropy_norm'])))
            gen_dev_list.append(float(np.mean(gen_data['util_dev'])))
            gen_cpu_list.append(float(gen_data.get('gen_cpu_time', float('nan'))))

        avg_history.append({
            'generation': gen_idx + 1,
            'mean_fitness': float(np.mean(gen_fitness_list)),
            'mean_accuracy': float(np.mean(gen_accs_list)),
            'mean_error_chance': float(np.mean(gen_err_chance_list)),
            'mean_error_utility': float(np.mean(gen_err_utility_list)),
            'mean_entropy_norm': float(np.mean(gen_ent_list)),
            'mean_util_dev': float(np.mean(gen_dev_list)),
            'mean_gen_cpu': float(np.nanmean(gen_cpu_list)),
        })

    return avg_history


# ═══════════════════════════════════════════════════════════════════════════════
#  ESCRITURA DE CSV
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_HEADER = [
    'mode', 'sampling_mode', 'chance_temperature', 'utility_temperature',
    'fitness_type', 'stop_mode', 'n_decision_rules', 'n_decision_rules_pct',
    'total_rules',
    'stop_gen_mean', 'stop_gen_std', 'stop_gen_min', 'stop_gen_max',
    'fitness_mean', 'fitness_std',
    'accuracy_mean', 'accuracy_std', 'accuracy_min', 'accuracy_max',
    'mse_chance_mean', 'mse_chance_std',
    'mse_utility_mean', 'mse_utility_std',
    'entropy_norm_mean', 'entropy_norm_std',
    'util_dev_mean', 'util_dev_std',
    # Tiempo de CPU (segundos): coste por generación y coste total de la corrida.
    'cpu_per_gen_mean', 'cpu_per_gen_std',
    'cpu_total_mean', 'cpu_total_std',
]

CURVES_HEADER = [
    'mode', 'sampling_mode', 'chance_temperature', 'utility_temperature',
    'fitness_type', 'stop_mode', 'n_decision_rules', 'n_decision_rules_pct',
    'generation', 'mean_fitness', 'mean_accuracy',
    'mean_error_chance', 'mean_error_utility', 'mean_entropy_norm', 'mean_util_dev',
    'mean_gen_cpu']


def ensure_csv_header(filepath, header):
    """Crea el archivo CSV con cabecera si no existe."""
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_results_row(filepath, row_dict):
    """Añade una fila al CSV de resultados."""
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER)
        writer.writerow(row_dict)


def append_curves_rows(filepath, rows):
    """Añade múltiples filas al CSV de curvas."""
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CURVES_HEADER)
        writer.writerows(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

ALL_OPTIMIZERS = ['umda', 'egna', 'emna', 'keda']


class _PrefixWriter:
    """Antepone un prefijo (p.ej. '[UMDA] ') a cada línea escrita en el stream.

    Se usa para que la salida de los procesos paralelos (uno por optimizador)
    sea legible aunque se entrelace en el mismo stdout del notebook.
    """
    def __init__(self, prefix, stream):
        self._prefix = prefix
        self._stream = stream
        self._line_start = True

    def write(self, text):
        if not text:
            return 0
        parts = text.split('\n')
        out = []
        for i, segment in enumerate(parts):
            has_newline = i < len(parts) - 1
            if segment and self._line_start:
                out.append(self._prefix)
                self._line_start = False
            out.append(segment)
            if has_newline:
                out.append('\n')
                self._line_start = True
        self._stream.write(''.join(out))
        return len(text)

    def flush(self):
        self._stream.flush()


def _worker(effective_config, base_folder):
    """Punto de entrada de cada proceso hijo: prefija stdout y corre el grid."""
    tag = f"[{effective_config['optimizer_type'].upper()}] "
    sys.stdout = _PrefixWriter(tag, sys.stdout)
    run_grid_for_optimizer(effective_config, base_folder)


def main():
    parser = argparse.ArgumentParser(description='Grid Search sistemático para IDRecovery.')
    parser.add_argument('--xdsl_path', default=None, help='Ruta al fichero .xdsl')
    parser.add_argument('--rules_csv', default=None, help='Ruta al CSV de reglas')
    parser.add_argument('--min_max_ut', type=lambda x: x.lower() not in ('false', '0', 'no'),
                        default=None, help='Normalizar utilidades (True/False)')
    parser.add_argument('--optimizer_type', default=None,
                        help="Optimizador(es): un nombre (umda), una lista "
                             "(umda,egna) o 'all' para los 4 en paralelo.")
    parser.add_argument('--base_folder', default=None,
                        help='Carpeta donde escribir los CSVs (default: BASE_FOLDER del script)')
    args = parser.parse_args()

    # Construir config efectiva: valores del script como base, args sobreescriben si se pasan
    base_effective_config = dict(BASE_CONFIG)
    if args.xdsl_path is not None:
        base_effective_config['xdsl_path'] = args.xdsl_path
    if args.rules_csv is not None:
        base_effective_config['rules_csv'] = args.rules_csv
    if args.min_max_ut is not None:
        base_effective_config['min_max_ut'] = args.min_max_ut

    base_folder = args.base_folder if args.base_folder else BASE_FOLDER

    # Resolver la lista de optimizadores a ejecutar.
    opt_arg = (args.optimizer_type or base_effective_config['optimizer_type']).lower()
    if opt_arg == 'all':
        optimizers = list(ALL_OPTIMIZERS)
    else:
        optimizers = [o.strip() for o in opt_arg.split(',') if o.strip()]

    # Un solo optimizador: ejecución directa (sin overhead de procesos).
    if len(optimizers) == 1:
        cfg = dict(base_effective_config)
        cfg['optimizer_type'] = optimizers[0]
        run_grid_for_optimizer(cfg, base_folder)
        return

    # Varios optimizadores: un proceso por optimizador, en paralelo. Cada uno
    # escribe en CSVs distintos (sufijo por optimizador), así que no hay choque.
    print("=" * 70)
    print(f"  GRID SEARCH — {len(optimizers)} optimizadores EN PARALELO: "
          f"{', '.join(o.upper() for o in optimizers)}")
    print("=" * 70, flush=True)

    procs = []
    for opt in optimizers:
        cfg = dict(base_effective_config)
        cfg['optimizer_type'] = opt
        p = mp.Process(target=_worker, args=(cfg, base_folder), name=opt.upper())
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    failed = [p.name for p in procs if p.exitcode != 0]
    print("\n" + "=" * 70)
    if failed:
        print(f"  TERMINADO con fallos en: {', '.join(failed)}")
    else:
        print("  TODOS LOS OPTIMIZADORES TERMINARON OK")
    print("=" * 70)


def run_grid_for_optimizer(effective_config, base_folder):
    effective_rules_csv = effective_config['rules_csv']
    suffix = f"_{effective_config['optimizer_type'].upper()}"
    results_csv = os.path.join(base_folder, f"grid_search_results{suffix}.csv")
    curves_csv = os.path.join(base_folder, f"grid_search_curves{suffix}.csv")

    # Tamaño de población efectivo (override por optimizador si está mapeado).
    effective_size_gen = SIZE_GEN_PER_OPTIMIZER.get(effective_config['optimizer_type'], SIZE_GEN)

    print("=" * 70)
    print("  GRID SEARCH — IDRecovery")
    print("=" * 70)
    print(f"Optimizador: {effective_config['optimizer_type'].upper()}")
    print(f"xdsl_path:   {effective_config['xdsl_path']}")
    print(f"rules_csv:   {effective_rules_csv}")
    print(f"min_max_ut:  {effective_config['min_max_ut']}")
    print(f"base_folder: {base_folder}")
    print(f"results:     {results_csv}")
    print(f"curves:      {curves_csv}")
    print(f"size_gen:    {effective_size_gen}")

    # Contar reglas totales
    total_rules = count_total_rules(effective_rules_csv)
    print(f"Total de reglas en el dataset: {total_rules}")

    # Calcular valores de n_decision_rules
    rules_values = [(pct, compute_n_rules(total_rules, pct)) for pct in RULES_PERCENTAGES]
    print(f"n_decision_rules (pct -> n): {rules_values}")

    # Generar todas las combinaciones
    grid = list(itertools.product(
        MODES,
        SAMPLING_MODES,
        CHANCE_TEMPERATURES,
        UTILITY_TEMPERATURES,
        FITNESS_TYPES,
        STOP_MODES,
        rules_values,
    ))
    total_combinations = len(grid)
    total_runs = total_combinations * N_REPETITIONS
    print(f"Combinaciones únicas: {total_combinations}")
    print(f"Total de ejecuciones: {total_runs}")
    print(f"Repeticiones por combinación: {N_REPETITIONS}")
    print()

    # Preparar CSVs
    ensure_csv_header(results_csv, RESULTS_HEADER)
    ensure_csv_header(curves_csv, CURVES_HEADER)

    # Cargar combinaciones ya completadas
    completed = get_completed_combinations(results_csv)
    print(f"Combinaciones ya completadas: {len(completed)}")
    print("=" * 70)
    print()
    
    global_start = time.time()
    combo_done = len(completed)
    
    for combo_idx, (mode, sampling_mode, chance_t, utility_t, fitness_type, stop_mode, (rules_pct, n_rules)) in enumerate(grid, 1):
        combo_key = (mode, sampling_mode, f"{chance_t}", f"{utility_t}", fitness_type, stop_mode, str(n_rules))

        if combo_key in completed:
            print(f"[{combo_idx}/{total_combinations}] SALTANDO (ya completada): "
                  f"mode={mode} | sampling={sampling_mode} | Tc={chance_t} | Tu={utility_t} | "
                  f"fitness={fitness_type} | stop={stop_mode} | rules={n_rules} ({rules_pct}%)")
            continue

        print(f"\n{'-' * 70}")
        print(f"[{combo_idx}/{total_combinations}] "
              f"mode={mode} | sampling={sampling_mode} | Tc={chance_t} | Tu={utility_t} | "
              f"fitness={fitness_type} | stop={stop_mode} | rules={n_rules} ({rules_pct}%)")
        print(f"{'-' * 70}")

        # --- Ejecutar N repeticiones ---
        experiments = []
        all_results = []

        combo_start = time.time()

        for rep in range(N_REPETITIONS):
            seed = BASE_SEED + rep
            print(f"\n  > Repetición {rep + 1}/{N_REPETITIONS} (seed={seed})")

            config = {
                **effective_config,
                'n_decision_rules': n_rules,
                'fitness_type': fitness_type,
                'stop_mode': stop_mode,
                'chance_temperature': chance_t,
                'utility_temperature': utility_t,
                'mode': mode,
                'symmetric_sampling': (sampling_mode == 'symmetric'),
                'random_seed': seed,
            }

            exp, results = run_single_experiment(config, effective_size_gen, MAX_ITER, TARGET_FITNESS)
            experiments.append(exp)
            all_results.append(results)

            print(f"    -> Parada en gen {results['stop_generation']} | "
                  f"Mejor fitness: {results['best_fitness']:.6f} | "
                  f"Mejor accuracy: {results['best_accuracy']:.1f}%")
        
        combo_elapsed = time.time() - combo_start
        
        # --- Calcular estadísticas agregadas sobre las N repeticiones ---
        # Cada lista tiene N_REPETITIONS valores, uno por experimento (última generación).
        stop_gens     = [r['stop_generation']   for r in all_results]
        best_fits     = [r['best_fitness']       for r in all_results]
        best_accs     = [r['best_accuracy']      for r in all_results]
        mse_chances   = [r['best_mse_chance']    for r in all_results]
        mse_utilities = [r['best_mse_utility']   for r in all_results]
        entropy_norms = [r['best_entropy_norm']  for r in all_results]
        util_devs     = [r['best_util_dev']      for r in all_results]
        cpu_per_gens  = [r['cpu_per_gen']        for r in all_results]
        cpu_totals    = [r['cpu_total']          for r in all_results]

        row = {
            'mode':                mode,
            'sampling_mode':       sampling_mode,
            'chance_temperature':  chance_t,
            'utility_temperature': utility_t,
            'fitness_type':        fitness_type,
            'stop_mode':           stop_mode,
            'n_decision_rules':    n_rules,
            'n_decision_rules_pct': rules_pct,
            'total_rules':         total_rules,
            'stop_gen_mean':       f"{np.mean(stop_gens):.2f}",
            'stop_gen_std':        f"{np.std(stop_gens):.2f}",
            'stop_gen_min':        min(stop_gens),
            'stop_gen_max':        max(stop_gens),
            'fitness_mean':        f"{np.mean(best_fits):.6f}",
            'fitness_std':         f"{np.std(best_fits):.6f}",
            'accuracy_mean':       f"{np.mean(best_accs):.2f}",
            'accuracy_std':        f"{np.std(best_accs):.2f}",
            'accuracy_min':        f"{min(best_accs):.2f}",
            'accuracy_max':        f"{max(best_accs):.2f}",
            'mse_chance_mean':     f"{np.mean(mse_chances):.6f}",
            'mse_chance_std':      f"{np.std(mse_chances):.6f}",
            'mse_utility_mean':    f"{np.mean(mse_utilities):.6f}",
            'mse_utility_std':     f"{np.std(mse_utilities):.6f}",
            'entropy_norm_mean':   f"{np.mean(entropy_norms):.6f}",
            'entropy_norm_std':    f"{np.std(entropy_norms):.6f}",
            'util_dev_mean':       f"{np.mean(util_devs):.6f}",
            'util_dev_std':        f"{np.std(util_devs):.6f}",
            'cpu_per_gen_mean':    f"{np.nanmean(cpu_per_gens):.4f}",
            'cpu_per_gen_std':     f"{np.nanstd(cpu_per_gens):.4f}",
            'cpu_total_mean':      f"{np.nanmean(cpu_totals):.4f}",
            'cpu_total_std':       f"{np.nanstd(cpu_totals):.4f}",
        }

        append_results_row(results_csv, row)

        # --- Calcular y guardar curvas promediadas ---
        avg_curves = average_histories(experiments)
        curve_rows = []
        for point in avg_curves:
            curve_rows.append({
                'mode': mode,
                'sampling_mode': sampling_mode,
                'chance_temperature': chance_t,
                'utility_temperature': utility_t,
                'fitness_type': fitness_type,
                'stop_mode': stop_mode,
                'n_decision_rules': n_rules,
                'n_decision_rules_pct': rules_pct,
                'generation': point['generation'],
                'mean_fitness': f"{point['mean_fitness']:.6f}",
                'mean_accuracy': f"{point['mean_accuracy']:.2f}",
                'mean_error_chance':  f"{point['mean_error_chance']:.6f}",
                'mean_error_utility': f"{point['mean_error_utility']:.6f}",
                'mean_entropy_norm':  f"{point['mean_entropy_norm']:.6f}",
                'mean_util_dev':      f"{point['mean_util_dev']:.6f}",
                'mean_gen_cpu':       f"{point['mean_gen_cpu']:.4f}",
            })

        append_curves_rows(curves_csv, curve_rows)
        
        combo_done += 1
        total_elapsed = time.time() - global_start
        avg_per_combo = total_elapsed / combo_done
        remaining = (total_combinations - combo_idx) * avg_per_combo
        
        print(f"\n  OK Combinación completada en {combo_elapsed:.1f}s")
        print(f"    Progreso: {combo_done}/{total_combinations} | "
              f"Tiempo total: {total_elapsed/60:.1f}min | "
              f"Estimado restante: {remaining/60:.1f}min")
    
    print(f"\n{'=' * 70}")
    print(f"  GRID SEARCH FINALIZADO")
    print(f"  Resultados: {results_csv}")
    print(f"  Curvas:     {curves_csv}")
    print(f"  Tiempo total: {(time.time() - global_start)/60:.1f} minutos")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
