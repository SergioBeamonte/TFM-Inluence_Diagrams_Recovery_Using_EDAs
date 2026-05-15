"""
Grid Search sistemático para IDRecovery.

Ejecuta un barrido de hiperparámetros (n_decision_rules, fitness_type, stop_mode)
con N repeticiones por combinación, y guarda:
  - grid_search_results.csv:  estadísticas agregadas (mean/std/min/max) por combinación
  - grid_search_curves.csv:   curvas de evolución promediadas por combinación

Uso:
    py grid_search.py
"""

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
FITNESS_TYPES = ["binary", "margin", "softmax", "regret_reg", "regret" "entropy"]
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
    """Ejecuta un único IDRecovery.run() y devuelve el objeto y resultados de la última gen."""
    from id_recovery import IDRecovery
    
    exp = IDRecovery(**config)
    exp.run(g=g, i=i, target_fitness=target_fitness)
    
    if exp.history:
        last_gen = exp.history[-1]
        results = {
            'stop_generation': last_gen['gen'],
            'final_fitness': float(np.min(last_gen['fitness'])),
            'final_accuracy': float(np.max(last_gen['accuracies'])),
            'final_error': float(np.min(last_gen['errors'])) # Asumiendo que 'errors' es tu MSE
        }
    else:
        results = {
            'stop_generation': 0,
            'final_fitness': float('nan'),
            'final_accuracy': float('nan'),
            'final_error': float('nan')
        }
    
    return exp, results


def average_histories(experiments):
    max_gens = max(len(exp.history) for exp in experiments)
    avg_history = []
    
    for gen_idx in range(max_gens):
        fits, accs, errs = [], [], []
        
        for exp in experiments:
            hist_idx = min(gen_idx, len(exp.history) - 1)
            gen_data = exp.history[hist_idx]
            
            # Cogemos el rendimiento promedio de la población en esa generación
            fits.append(float(np.mean(gen_data['fitness'])))
            accs.append(float(np.mean(gen_data['accuracies'])))
            errs.append(float(np.mean(gen_data['errors'])))
            
        avg_history.append({
            'generation': gen_idx + 1,
            'fitness': float(np.mean(fits)),
            'accuracy': float(np.mean(accs)),
            'mse': float(np.mean(errs)),
        })
    
    return avg_history


# ═══════════════════════════════════════════════════════════════════════════════
#  ESCRITURA DE CSV
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_HEADER = [
    'fitness_type', 'stop_mode', 'n_decision_rules', 'n_decision_rules_pct', 'total_rules',
    'stop_gen_mejor', 'stop_gen_peor', 'stop_gen_media', 'stop_gen_std',
    'fitness_mejor', 'fitness_peor', 'fitness_media', 'fitness_std',
    'accuracy_mejor', 'accuracy_peor', 'accuracy_media', 'accuracy_std',
    'error_mejor', 'error_peor', 'error_media', 'error_std'
]

CURVES_HEADER = [
    'fitness_type', 'stop_mode', 'n_decision_rules', 'n_decision_rules_pct',
    'generation', 'fitness', 'accuracy', 'mse'
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
    print("=" * 70)
    print("  GRID SEARCH — IDRecovery")
    print("=" * 70)
    
    # Contar reglas totales
    total_rules = count_total_rules(RULES_CSV)
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
    ensure_csv_header(RESULTS_CSV, RESULTS_HEADER)
    ensure_csv_header(CURVES_CSV, CURVES_HEADER)
    
    # Cargar combinaciones ya completadas
    completed = get_completed_combinations(RESULTS_CSV)
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
                **BASE_CONFIG,
                'n_decision_rules': n_rules,
                'fitness_type': fitness_type,
                'stop_mode': stop_mode,
                'random_seed': seed,
            }
            
            exp, results = run_single_experiment(config, SIZE_GEN, MAX_ITER, TARGET_FITNESS)
            experiments.append(exp)
            all_results.append(results)
            
            print(f"    -> Parada en gen {results['stop_generation']} | "
                  f"Mejor fitness: {results['final_fitness']:.6f} | "
                  f"Mejor accuracy: {results['final_accuracy']:.1f}%")
        
        combo_elapsed = time.time() - combo_start
        
        # --- Calcular estadísticas agregadas ---
        stop_gens = [r['stop_generation'] for r in all_results]
        fits = [r['final_fitness'] for r in all_results]
        accs = [r['final_accuracy'] for r in all_results]
        errs = [r['final_error'] for r in all_results]
        
        row = {
            'fitness_type': fitness_type,
            'stop_mode': stop_mode,
            'n_decision_rules': n_rules,
            'n_decision_rules_pct': rules_pct,
            'total_rules': total_rules,
            
            'stop_gen_mejor': min(stop_gens),
            'stop_gen_peor': max(stop_gens),
            'stop_gen_media': f"{np.mean(stop_gens):.2f}",
            'stop_gen_std': f"{np.std(stop_gens):.2f}",
            
            'fitness_mejor': f"{min(fits):.6f}",
            'fitness_peor': f"{max(fits):.6f}",
            'fitness_media': f"{np.mean(fits):.6f}",
            'fitness_std': f"{np.std(fits):.6f}",
            
            'accuracy_mejor': f"{max(accs):.2f}",
            'accuracy_peor': f"{min(accs):.2f}",
            'accuracy_media': f"{np.mean(accs):.2f}",
            'accuracy_std': f"{np.std(accs):.2f}",
            
            'error_mejor': f"{min(errs):.6f}",
            'error_peor': f"{max(errs):.6f}",
            'error_media': f"{np.mean(errs):.6f}",
            'error_std': f"{np.std(errs):.6f}",
        }
        
        append_results_row(RESULTS_CSV, row)
        
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
                'fitness': f"{point['fitness']:.6f}",
                'accuracy': f"{point['accuracy']:.2f}",
                'mse': f"{point['mse']:.6f}",
            })
        
        append_curves_rows(CURVES_CSV, curve_rows)
        
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
    print(f"  Resultados: {RESULTS_CSV}")
    print(f"  Curvas:     {CURVES_CSV}")
    print(f"  Tiempo total: {(time.time() - global_start)/60:.1f} minutos")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
