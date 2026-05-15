"""
Grid Search sistemático para IDRecovery.

Ejecuta un barrido de hiperparámetros (n_decision_rules, fitness_type, stop_mode)
con N repeticiones por combinación, y guarda:
  - grid_search_results.csv:  estadísticas agregadas (mean/std/min/max) por combinación
  - grid_search_curves.csv:   curvas de evolución promediadas por combinación

Uso:
    py grid_search.py
    py grid_search.py --optimizer_type egda
    py grid_search.py --xdsl_path ruta/red.xdsl --rules_csv ruta/reglas.csv --min_max_ut True --optimizer_type umda
"""

import argparse
import os
import sys
import csv
import time
import itertools
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN  — Modifica aquí los parámetros del grid
# ═══════════════════════════════════════════════════════════════════════════════

# --- Red y reglas ---
BASE_FOLDER = r"example\bypass2"

XDSL_PATH = os.path.join(BASE_FOLDER, r"network-bypass2.xdsl")
RULES_CSV = os.path.join(BASE_FOLDER, r"reglas_generadas.csv")

# --- Parámetros fijos del optimizador ---
BASE_CONFIG = {
    'xdsl_path': XDSL_PATH,
    'rules_csv': RULES_CSV,
    'min_max_ut': True,
    'save_plots': False,
    'u_range': (0, 10),
    'alpha': 0.5,
    'elite_factor': 0.0,
    'optimizer_type': 'umda',
}

# --- Parámetros del optimizador ---
SIZE_GEN = 100       # tamaño de población por generación
MAX_ITER = 100       # máximo de generaciones
TARGET_FITNESS = 1e-5

# --- Grid de búsqueda ---
# n_decision_rules se calcula como % del total de reglas
RULES_PERCENTAGES = [5, 10, 20, 40, 60]  # porcentajes
FITNESS_TYPES = ["binary", "margin", "softmax", "regret_reg", "regret", "entropy"]
STOP_MODES = ["top10", "top30", "top70", "top90"]

# --- Repeticiones ---
N_REPETITIONS = 5
BASE_SEED = 42

# --- Archivos de salida ---
RESULTS_CSV = os.path.join(BASE_FOLDER, r"grid_search_results.csv")
CURVES_CSV = os.path.join(BASE_FOLDER, r"grid_search_curves.csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════════

def count_total_rules(rules_csv_path):
    """Cuenta el número total de reglas en el CSV."""
    with open(rules_csv_path, 'r') as f:
        return sum(1 for _ in csv.DictReader(f))


def get_completed_combinations(results_csv_path):
    """Lee el CSV de resultados y devuelve un set de combinaciones ya completadas."""
    completed = set()
    if not os.path.exists(results_csv_path):
        return completed
    with open(results_csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['fitness_type'], row['stop_mode'], row['n_decision_rules'])
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
    """
    from id_recovery import IDRecovery

    exp = IDRecovery(**config)
    best_vector = exp.run(g=g, i=i, target_fitness=target_fitness)

    # Extraer métricas de la última generación del historial
    if exp.history:
        last_gen = exp.history[-1]
        stop_generation = last_gen['gen']
        best_fitness = float(np.min(last_gen['fitness']))
        mean_fitness = float(np.mean(last_gen['fitness']))
        best_accuracy = float(np.max(last_gen['accuracies']))
        mean_accuracy = float(np.mean(last_gen['accuracies']))
        # MSE separados por tipo de nodo (chance = CPTs, utility = utilidades reescaladas)
        best_mse_chance = float(np.min(last_gen['errors_chance']))
        mean_mse_chance = float(np.mean(last_gen['errors_chance']))
        best_mse_utility = float(np.min(last_gen['errors_utility']))
        mean_mse_utility = float(np.mean(last_gen['errors_utility']))
    else:
        # Si no hay historial (paró en la primera evaluación)
        stop_generation = 0
        best_fitness = float(exp.best_historical_fitness) if exp.best_historical_fitness != float('inf') else float('nan')
        mean_fitness = float('nan')
        best_accuracy = float('nan')
        mean_accuracy = float('nan')
        best_mse_chance = mean_mse_chance = float('nan')
        best_mse_utility = mean_mse_utility = float('nan')

    results = {
        'stop_generation': stop_generation,
        'best_fitness': best_fitness,
        'mean_fitness': mean_fitness,
        'best_accuracy': best_accuracy,
        'mean_accuracy': mean_accuracy,
        'best_mse_chance': best_mse_chance,
        'mean_mse_chance': mean_mse_chance,
        'best_mse_utility': best_mse_utility,
        'mean_mse_utility': mean_mse_utility,
    }

    return exp, results


def average_histories(experiments):
    """
    Promedia los historiales de N experimentos (igual que BatchExperimenter._average_histories).
    Devuelve una lista de dicts con las curvas promediadas — un punto por generación.

    Para cada generación se promedia primero sobre la poblacion (dentro de cada experimento)
    y despues sobre los experimentos. Asi cada curva representa la evolucion media del batch.
    """
    max_gens = max(len(exp.history) for exp in experiments)
    avg_history = []

    for gen_idx in range(max_gens):
        gen_fitness_list = []
        gen_err_chance_list = []
        gen_err_utility_list = []
        gen_accs_list = []

        for exp in experiments:
            hist_idx = min(gen_idx, len(exp.history) - 1)
            gen_data = exp.history[hist_idx]

            # Tomamos la media de la población para esa generación
            gen_fitness_list.append(float(np.mean(gen_data['fitness'])))
            gen_err_chance_list.append(float(np.mean(gen_data['errors_chance'])))
            gen_err_utility_list.append(float(np.mean(gen_data['errors_utility'])))
            gen_accs_list.append(float(np.mean(gen_data['accuracies'])))

        avg_history.append({
            'generation': gen_idx + 1,
            'mean_fitness': float(np.mean(gen_fitness_list)),
            'mean_accuracy': float(np.mean(gen_accs_list)),
            'mean_error_chance': float(np.mean(gen_err_chance_list)),
            'mean_error_utility': float(np.mean(gen_err_utility_list)),
        })

    return avg_history


# ═══════════════════════════════════════════════════════════════════════════════
#  ESCRITURA DE CSV
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_HEADER = [
    'fitness_type', 'stop_mode', 'n_decision_rules', 'n_decision_rules_pct',
    'total_rules',
    'stop_gen_mean', 'stop_gen_std', 'stop_gen_min', 'stop_gen_max',
    'best_fitness_mean', 'best_fitness_std',
    'mean_accuracy_mean', 'mean_accuracy_std',
    'best_accuracy_mean', 'best_accuracy_std', 'best_accuracy_min', 'best_accuracy_max',
    # MSE de CPTs (nodos chance) — mejor por experimento, agregado sobre el batch
    'best_mse_chance_mean', 'best_mse_chance_std', 'best_mse_chance_min', 'best_mse_chance_max',
    'mean_mse_chance_mean', 'mean_mse_chance_std', 'mean_mse_chance_min', 'mean_mse_chance_max',
    # MSE de utilidades (reescaladas a [u_min, u_max]) — mejor por experimento, agregado sobre el batch
    'best_mse_utility_mean', 'best_mse_utility_std', 'best_mse_utility_min', 'best_mse_utility_max',
    'mean_mse_utility_mean', 'mean_mse_utility_std', 'mean_mse_utility_min', 'mean_mse_utility_max',
]

CURVES_HEADER = [
    'fitness_type', 'stop_mode', 'n_decision_rules', 'n_decision_rules_pct',
    'generation', 'mean_fitness', 'mean_accuracy',
    'mean_error_chance', 'mean_error_utility',
]


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

def main():
    parser = argparse.ArgumentParser(description='Grid Search sistemático para IDRecovery.')
    parser.add_argument('--xdsl_path', default=None, help='Ruta al fichero .xdsl')
    parser.add_argument('--rules_csv', default=None, help='Ruta al CSV de reglas')
    parser.add_argument('--min_max_ut', type=lambda x: x.lower() not in ('false', '0', 'no'),
                        default=None, help='Normalizar utilidades (True/False)')
    parser.add_argument('--optimizer_type', default=None, choices=['umda', 'egda'],
                        help='Tipo de optimizador')
    args = parser.parse_args()

    # Construir config efectiva: valores del script como base, args sobreescriben si se pasan
    effective_config = dict(BASE_CONFIG)
    if args.xdsl_path is not None:
        effective_config['xdsl_path'] = args.xdsl_path
    if args.rules_csv is not None:
        effective_config['rules_csv'] = args.rules_csv
    if args.min_max_ut is not None:
        effective_config['min_max_ut'] = args.min_max_ut
    if args.optimizer_type is not None:
        effective_config['optimizer_type'] = args.optimizer_type

    effective_rules_csv = effective_config['rules_csv']
    suffix = f"_{effective_config['optimizer_type'].upper()}"
    results_csv = os.path.join(BASE_FOLDER, f"grid_search_results{suffix}.csv")
    curves_csv = os.path.join(BASE_FOLDER, f"grid_search_curves{suffix}.csv")

    print("=" * 70)
    print("  GRID SEARCH — IDRecovery")
    print("=" * 70)
    print(f"Optimizador: {effective_config['optimizer_type'].upper()}")
    print(f"xdsl_path:   {effective_config['xdsl_path']}")
    print(f"rules_csv:   {effective_rules_csv}")
    print(f"min_max_ut:  {effective_config['min_max_ut']}")

    # Contar reglas totales
    total_rules = count_total_rules(effective_rules_csv)
    print(f"Total de reglas en el dataset: {total_rules}")

    # Calcular valores de n_decision_rules
    rules_values = [(pct, compute_n_rules(total_rules, pct)) for pct in RULES_PERCENTAGES]
    print(f"n_decision_rules (pct -> n): {rules_values}")

    # Generar todas las combinaciones
    grid = list(itertools.product(
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
    
    for combo_idx, (fitness_type, stop_mode, (rules_pct, n_rules)) in enumerate(grid, 1):
        combo_key = (fitness_type, stop_mode, str(n_rules))
        
        if combo_key in completed:
            print(f"[{combo_idx}/{total_combinations}] SALTANDO (ya completada): "
                  f"fitness={fitness_type} | stop={stop_mode} | rules={n_rules} ({rules_pct}%)")
            continue
        
        print(f"\n{'-' * 70}")
        print(f"[{combo_idx}/{total_combinations}] "
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
                'random_seed': seed,
            }

            exp, results = run_single_experiment(config, SIZE_GEN, MAX_ITER, TARGET_FITNESS)
            experiments.append(exp)
            all_results.append(results)

            print(f"    -> Parada en gen {results['stop_generation']} | "
                  f"Mejor fitness: {results['best_fitness']:.6f} | "
                  f"Mejor accuracy: {results['best_accuracy']:.1f}%")
        
        combo_elapsed = time.time() - combo_start
        
        # --- Calcular estadísticas agregadas ---
        # Cada lista tiene N_REPETITIONS valores: uno por experimento.
        # Cada valor proviene de la ultima generacion (etapa final) del experimento.
        stop_gens = [r['stop_generation'] for r in all_results]
        best_fits = [r['best_fitness'] for r in all_results]
        mean_accs = [r['mean_accuracy'] for r in all_results]
        best_accs = [r['best_accuracy'] for r in all_results]
        best_mse_chance = [r['best_mse_chance'] for r in all_results]
        mean_mse_chance = [r['mean_mse_chance'] for r in all_results]
        best_mse_utility = [r['best_mse_utility'] for r in all_results]
        mean_mse_utility = [r['mean_mse_utility'] for r in all_results]

        row = {
            'fitness_type': fitness_type,
            'stop_mode': stop_mode,
            'n_decision_rules': n_rules,
            'n_decision_rules_pct': rules_pct,
            'total_rules': total_rules,
            
            'stop_gen_mean': f"{np.mean(stop_gens):.2f}",
            'stop_gen_std': f"{np.std(stop_gens):.2f}",
            'stop_gen_min': min(stop_gens),
            'stop_gen_max': max(stop_gens),
            'best_fitness_mean': f"{np.mean(best_fits):.6f}",
            'best_fitness_std': f"{np.std(best_fits):.6f}",
            'mean_accuracy_mean': f"{np.mean(mean_accs):.2f}",
            'mean_accuracy_std': f"{np.std(mean_accs):.2f}",
            'best_accuracy_mean': f"{np.mean(best_accs):.2f}",
            'best_accuracy_std': f"{np.std(best_accs):.2f}",
            'best_accuracy_min': f"{min(best_accs):.2f}",
            'best_accuracy_max': f"{max(best_accs):.2f}",
            'best_mse_chance_mean': f"{np.mean(best_mse_chance):.6f}",
            'best_mse_chance_std':  f"{np.std(best_mse_chance):.6f}",
            'best_mse_chance_min':  f"{min(best_mse_chance):.6f}",
            'best_mse_chance_max':  f"{max(best_mse_chance):.6f}",
            'mean_mse_chance_mean': f"{np.mean(mean_mse_chance):.6f}",
            'mean_mse_chance_std':  f"{np.std(mean_mse_chance):.6f}",
            'mean_mse_chance_min':  f"{min(mean_mse_chance):.6f}",
            'mean_mse_chance_max':  f"{max(mean_mse_chance):.6f}",
            'best_mse_utility_mean': f"{np.mean(best_mse_utility):.6f}",
            'best_mse_utility_std':  f"{np.std(best_mse_utility):.6f}",
            'best_mse_utility_min':  f"{min(best_mse_utility):.6f}",
            'best_mse_utility_max':  f"{max(best_mse_utility):.6f}",
            'mean_mse_utility_mean': f"{np.mean(mean_mse_utility):.6f}",
            'mean_mse_utility_std':  f"{np.std(mean_mse_utility):.6f}",
            'mean_mse_utility_min':  f"{min(mean_mse_utility):.6f}",
            'mean_mse_utility_max':  f"{max(mean_mse_utility):.6f}",
        }

        append_results_row(results_csv, row)

        # --- Calcular y guardar curvas promediadas ---
        avg_curves = average_histories(experiments)
        curve_rows = []
        for point in avg_curves:
            curve_rows.append({
                'fitness_type': fitness_type,
                'stop_mode': stop_mode,
                'n_decision_rules': n_rules,
                'n_decision_rules_pct': rules_pct,
                'generation': point['generation'],
                'mean_fitness': f"{point['mean_fitness']:.6f}",
                'mean_accuracy': f"{point['mean_accuracy']:.2f}",
                'mean_error_chance':  f"{point['mean_error_chance']:.6f}",
                'mean_error_utility': f"{point['mean_error_utility']:.6f}",
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
