"""
Módulo con las clases principales para la recuperación de Diagramas de Influencia
mediante Algoritmos de Estimación de Distribuciones (EDA).

Clases:
    - IDRecovery: Optimizador EDA que recupera parámetros de un ID a partir de reglas.
    - AveragedExperiment: Contenedor de resultados promediados.
    - BatchExperimenter: Ejecuta múltiples experimentos y promedia resultados.
"""

import pysmile_license
import pysmile
import numpy as np
import random
import csv
from scipy.special import expit
from EDAspy.optimization import UMDAc, EGNA, EMNA


class IDRecovery:
    def __init__(self, xdsl_path, rules_csv, min_max_ut=False, u_range=(0, 10),
                 chance_init_bounds=(-5, 5), utility_init_bounds=(-10, 10),
                 alpha=0.5, elite_factor=0.0, n_decision_rules=-1,
                 fitness_type='regret', stop_mode='best', optimizer_type='umda',
                 chance_temperature=1.0, utility_temperature=1.0,
                 mode='both', random_seed=42):

        self.net = pysmile.Network()
        self.net.read_file(xdsl_path)
        self.u_min, self.u_max = u_range
        self.min_max_ut = min_max_ut
        self.alpha = alpha
        self.elite_factor = elite_factor
        self.n_decision_rules = n_decision_rules
        self.fitness_type = fitness_type
        self.stop_mode = stop_mode
        self.optimizer_type = optimizer_type.lower()
        # Temperaturas de las funciones de decodificación. softmax(raw/T) y sigmoid(raw/T):
        #   T < 1  → curva AFILADA (sharp/peaked). Pequeños cambios en raw saltan a 0/1.
        #            Útil para CPTs cuasi-determinísticas o utilidades bimodales.
        #   T = 1  → comportamiento clásico.
        #   T > 1  → curva ESTIRADA (flat). Casi lineal cerca de 0, satura tarde.
        #            Útil cuando las CPTs son intermedias o quieres suavidad en utilidades.
        self.chance_temperature = float(chance_temperature)
        self.utility_temperature = float(utility_temperature)
        # Modo de optimización (descompone el problema):
        #   'both'         → optimiza CPTs y utilidades a la vez (caso general).
        #   'utility_only' → fija las CPTs a sus valores originales y solo busca utilidades.
        #                    Caso realista cuando las CPTs vienen de datos y solo se desconocen
        #                    las preferencias del decisor.
        #   'chance_only'  → fija las utilidades a las originales y solo busca CPTs.
        #                    Útil para diagnóstico: preferencias dadas, dinámica por aprender.
        if mode not in ('both', 'utility_only', 'chance_only'):
            raise ValueError(f"mode debe ser 'both' | 'utility_only' | 'chance_only', no '{mode}'")
        self.mode = mode

        self.chance_init_bounds = chance_init_bounds
        self.utility_init_bounds = utility_init_bounds

        self.chance_nodes = self._get_nodes(["CPT", "TRUTHTABLE"])
        self.utility_nodes = self._get_nodes(["UTILITY"])

        self.original_defs = {
            name: np.array(self.net.get_node_definition(self.net.get_node(name)))
            for name in self.chance_nodes + self.utility_nodes
        }

        # Reescala las utilidades originales al mismo rango [u_min, u_max] que usa el decodificador,
        # para que real_error compare escalas equivalentes (las decisiones óptimas son invariantes
        # a transformaciones afines positivas, así que esto sólo normaliza la escala numérica).
        # El min/max se toma GLOBAL sobre toda la red para preservar el orden relativo entre nodos.
        self.original_defs_scaled = {name: arr for name, arr in self.original_defs.items()
                                     if name in self.chance_nodes}
        if self.utility_nodes:
            all_u = np.concatenate([self.original_defs[n].flatten() for n in self.utility_nodes])
            self.orig_u_min, self.orig_u_max = float(all_u.min()), float(all_u.max())
            span = self.orig_u_max - self.orig_u_min
            for name in self.utility_nodes:
                if span > 0:
                    rescaled = (self.original_defs[name] - self.orig_u_min) / span \
                               * (self.u_max - self.u_min) + self.u_min
                else:
                    rescaled = np.full_like(self.original_defs[name],
                                            (self.u_max + self.u_min) / 2, dtype=float)
                self.original_defs_scaled[name] = rescaled

        self.specs = self._build_specs()
        self.total_vars = sum(s['free_size'] for s in self.specs)
        if self.total_vars == 0:
            raise ValueError(
                f"mode='{self.mode}' deja 0 variables libres: la red no tiene nodos del tipo "
                "que querías optimizar (o todas las utilidades están ancladas por min_max_ut)."
            )

        self.all_rules = self._compile_rules(rules_csv)

        # Sembramos tanto random (para el sampleo de reglas) como numpy.random (para la
        # inicialización y muestreo interno de EDAspy, que usa np.random globalmente).
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        n_all = len(self.all_rules)
        if 0 < self.n_decision_rules < n_all:
            self.train_rules = random.sample(self.all_rules, self.n_decision_rules)
            train_id_set = {id(r) for r in self.train_rules}
            self.train_mask = [id(r) in train_id_set for r in self.all_rules]
        else:
            self.train_rules = self.all_rules
            self.train_mask = [True] * n_all

        print(f"Optimizador: {self.optimizer_type.upper()}")
        print(f"Modo de Fitness: {self.fitness_type.upper()}")
        print(f"Reglas totales para evaluar precisión: {len(self.all_rules)}")
        print(f"Reglas usadas para entrenar (Fitness): {len(self.train_rules)}")

        self.eval_count = 0
        self.current_gen = 1
        self.evals_this_gen = 0

        self.gen_fitness = []
        self.gen_errors_chance = []
        self.gen_errors_utility = []
        self.gen_accuracies = []
        # Diagnósticos de sesgo en la parametrización logit/sigmoide:
        #   gen_entropy_norm: suma de H(p)/log(k) sobre todas las filas de todas las CPTs.
        #                     Bajo => CPTs casi determinísticas. Alto => CPTs uniformes (sesgo).
        #   gen_util_dev:     suma de |u - mean(u)| sobre todos los valores de cada nodo utility,
        #                     acumulado entre nodos. Bajo => utilidades planas (sesgo). Alto => spread.
        self.gen_entropy_norm = []
        self.gen_util_dev = []

        self.history = []

        # Inicializados de verdad por run(); valores neutros que NO disparan parada anticipada
        # ni cierre de generación si fitness se llamara antes de run().
        self.size_gen = float('inf')
        self.target_fitness = -float('inf')
        self.best_historical_fitness = float('inf')
        self.best_historical_ind = None
        self.patience = 0
        self.min_delta = 0.0
        self.stagnation_counter = 0
        self.best_stagnation_fitness = float('inf')

    def _get_nodes(self, types):
        valid = [getattr(pysmile.NodeType, t) for t in types if hasattr(pysmile.NodeType, t)]
        return [self.net.get_node_id(h) for h in self.net.get_all_nodes() if self.net.get_node_type(h) in valid]

    def _build_specs(self):
        specs = []
        for name in self.chance_nodes + self.utility_nodes:
            h = self.net.get_node(name)
            size = len(self.net.get_node_definition(h))
            kind = 'chance' if name in self.chance_nodes else 'utility'
            parents = self.net.get_parents(h)
            shape = tuple([self.net.get_outcome_count(p) for p in parents]) + ((self.net.get_outcome_count(h),) if kind == 'chance' else ())

            mask = np.ones(size, dtype=bool)
            fixed = []  # lista de (flat_idx, valor_fijo), precomputado para no llamar a
                        # np.ravel_multi_index en cada decode.

            # Modo descompuesto: si toca fijar este tipo de nodo, mete TODAS las
            # entradas del nodo a su valor original (escalado para utilities) y
            # anula la máscara — free_size pasa a 0 y no aporta variables a la EDA.
            fully_fixed_by_mode = (
                (self.mode == 'utility_only' and kind == 'chance') or
                (self.mode == 'chance_only' and kind == 'utility')
            )

            if fully_fixed_by_mode:
                original_vals = self.original_defs_scaled[name].flatten()
                for flat_idx in range(size):
                    fixed.append((flat_idx, float(original_vals[flat_idx])))
                mask[:] = False
            elif kind == 'utility' and self.min_max_ut:
                original_u = self.original_defs[name]
                best_flat = int(np.argmax(original_u))
                worst_flat = int(np.argmin(original_u))

                fixed.append((best_flat, self.u_max))
                mask[best_flat] = False

                if worst_flat != best_flat:
                    fixed.append((worst_flat, self.u_min))
                    mask[worst_flat] = False

            specs.append({'name': name, 'kind': kind, 'size': size, 'free_size': mask.sum(), 'shape': shape, 'mask': mask, 'fixed': fixed})
        return specs

    def _compile_rules(self, path):
        compiled = []
        with open(path, 'r') as f:
            for row in csv.DictReader(f):
                ev = {k: int(v) - 1 for k, v in row.items() if int(v or 0) > 0}
                target = next(k for k, v in row.items() if int(v or 0) < 0)
                act_idx = abs(int(row[target])) - 1
                h = self.net.get_node(target)
                mult, c_idx = 1, 0
                for p in reversed(self.net.get_parents(h)):
                    c_idx += ev[self.net.get_node_id(p)] * mult
                    mult *= self.net.get_outcome_count(p)
                compiled.append({'node': target, 'c_idx': c_idx, 'a_idx': act_idx, 'n_act': self.net.get_outcome_count(h)})
        return compiled

    def _decode_vector(self, vector):
        pos = 0
        real_error_chance = 0.0
        real_error_utility = 0.0
        # Diagnósticos de sesgo: ver __init__.
        total_entropy_norm = 0.0
        total_util_dev = 0.0
        decoded_vals = {}

        for s in self.specs:
            raw = vector[pos:pos+s['free_size']]
            pos += s['free_size']

            if s['kind'] == 'chance':
                if s['free_size'] == 0:
                    # Nodo CPT fijo a su definición original (modo utility_only).
                    val = np.zeros(s['size'])
                    for flat, fv in s['fixed']:
                        val[flat] = fv
                    probs = val.reshape(s['shape'])
                else:
                    # softmax(raw / T). T<1 afila, T>1 estira hacia uniforme.
                    raw_r = raw.reshape(s['shape']) / self.chance_temperature
                    res = np.exp(raw_r - raw_r.max(axis=-1, keepdims=True))
                    probs = res / res.sum(axis=-1, keepdims=True)
                    val = probs.flatten()

                # Entropía normalizada acumulada (suma sobre filas del simplex).
                k = s['shape'][-1]
                log_k = np.log(k) if k > 1 else 1.0
                with np.errstate(divide='ignore', invalid='ignore'):
                    p_log_p = np.where(probs > 0, probs * np.log(probs), 0.0)
                H_per_row = -p_log_p.sum(axis=-1)
                total_entropy_norm += float((H_per_row / log_k).sum())
            else:
                # sigmoid(raw / T) * (u_max - u_min) + u_min. T<1 afila hacia los extremos,
                # T>1 estira el rango cuasi-lineal alrededor del midpoint.
                val = np.zeros(s['size'])
                val[s['mask']] = expit(raw / self.utility_temperature) * (self.u_max - self.u_min) + self.u_min
                for flat, fv in s['fixed']:
                    val[flat] = fv

                # Suma de desviaciones absolutas de las utilidades respecto a su media.
                if val.size > 0:
                    total_util_dev += float(np.abs(val - val.mean()).sum())

            decoded_vals[s['name']] = val
            node_mse = float(np.mean((val - self.original_defs_scaled[s['name']])**2))
            if s['kind'] == 'chance':
                real_error_chance += node_mse
            else:
                real_error_utility += node_mse

        return decoded_vals, real_error_chance, real_error_utility, total_entropy_norm, total_util_dev

    def fitness(self, vector):
        decoded_vals, real_error_chance, real_error_utility, entropy_norm, util_dev = self._decode_vector(vector)

        for name, val in decoded_vals.items():
            self.net.set_node_definition(name, val.tolist())

        try:
            self.net.update_beliefs()
        except pysmile.SMILEException:
            return 1e6


        penalty_score = 0
        rules_fulfilled = 0
        ftype = self.fitness_type
        train_mask = self.train_mask

        for idx, r in enumerate(self.all_rules):
            try:
                utils = self.net.get_node_value(r['node'])[r['c_idx']*r['n_act'] : (r['c_idx']+1)*r['n_act']]
                max_u = max(utils)
                rule_u = utils[r['a_idx']]

                if (max_u - rule_u) <= 1e-5:
                    rules_fulfilled += 1

                if not train_mask[idx]:
                    continue

                if ftype == 'binary':
                    if (max_u - rule_u) > 0:
                        penalty_score += 1

                elif ftype == 'regret' or ftype == 'regret_reg':
                    penalty_score += (max_u - rule_u)

                elif ftype == 'margin':
                    margen = 1.0
                    utilidades_alternativas = [u for i, u in enumerate(utils) if i != r['a_idx']]
                    mejor_alternativa = max(utilidades_alternativas) if utilidades_alternativas else 0
                    penalty_score += max(0, (mejor_alternativa + margen) - rule_u)

                elif ftype == 'softmax':
                    exp_utils = np.exp(np.array(utils) - max_u)
                    prob_rule = exp_utils[r['a_idx']] / np.sum(exp_utils)
                    penalty_score += -np.log(prob_rule + 1e-9)

                elif ftype == 'entropy':
                    exp_utils = np.exp(np.array(utils) - max_u)
                    probs = exp_utils / np.sum(exp_utils)
                    prob_rule = probs[r['a_idx']]
                    nll_loss = -np.log(prob_rule + 1e-9)
                    entropy = -np.sum(probs * np.log(probs + 1e-9))
                    alpha = 0.1
                    penalty_score += nll_loss + (alpha * entropy)

            except IndexError:
                return 1e6

        if ftype == 'regret_reg':
            penalty_score += 0.01 * np.sum(vector**2)

        if penalty_score < self.best_historical_fitness:
            self.best_historical_fitness = penalty_score
            self.best_historical_ind = vector

        accuracy = (rules_fulfilled / len(self.all_rules)) * 100

        self.gen_fitness.append(penalty_score)
        self.gen_errors_chance.append(real_error_chance)
        self.gen_errors_utility.append(real_error_utility)
        self.gen_entropy_norm.append(entropy_norm)
        self.gen_util_dev.append(util_dev)
        self.gen_accuracies.append(accuracy)
        self.eval_count += 1
        self.evals_this_gen += 1

        # --- FINAL DE GENERACIÓN ---

        if self.evals_this_gen >= self.size_gen:
            
            errors_chance_arr = np.array(self.gen_errors_chance)
            errors_utility_arr = np.array(self.gen_errors_utility)
            self.history.append({
                'gen': self.current_gen,
                'errors_chance': errors_chance_arr,
                'errors_utility': errors_utility_arr,
                'errors': errors_chance_arr + errors_utility_arr,
                'fitness': np.array(self.gen_fitness),
                'accuracies': np.array(self.gen_accuracies),
                'entropy_norm': np.array(self.gen_entropy_norm),
                'util_dev': np.array(self.gen_util_dev),
            })
            
            sorted_fitness = np.sort(self.gen_fitness)
            mejor_fitness_generacion = sorted_fitness[0]
            
            # --- CHEQUEO DE ESTANCAMIENTO (Solo para distribuciones) ---
            if self.fitness_type in ['softmax', 'entropy']:
                if mejor_fitness_generacion < (self.best_stagnation_fitness - self.min_delta):
                    self.best_stagnation_fitness = mejor_fitness_generacion
                    self.stagnation_counter = 0 # Resetea la paciencia
                else:
                    self.stagnation_counter += 1 # Gasta paciencia
                    
                if self.stagnation_counter >= self.patience:
                    raise StopIteration(f"Estancamiento: {self.patience} generaciones seguidas sin mejorar al menos {self.min_delta}.")
            
            # --- LÓGICA DE PARADA ORIGINAL ---
            if self.stop_mode == 'top10':
                target_idx = max(1, int(self.size_gen * 0.10)) - 1
                msg_parada = f"El Top 10% (>={target_idx+1} individuos) alcanzó"
            elif self.stop_mode == 'top30':
                target_idx = max(1, int(self.size_gen * 0.30)) - 1
                msg_parada = f"El Top 30% (>={target_idx+1} individuos) alcanzó"
            elif self.stop_mode == 'top50':
                target_idx = max(1, int(self.size_gen * 0.50)) - 1
                msg_parada = f"El Top 50% (>={target_idx+1} individuos) alcanzó"
            elif self.stop_mode == 'top70':
                target_idx = max(1, int(self.size_gen * 0.70)) - 1
                msg_parada = f"El Top 70% (>={target_idx+1} individuos) alcanzó"
            elif self.stop_mode == 'top90':
                target_idx = max(1, int(self.size_gen * 0.90)) - 1
                msg_parada = f"El Top 90% (>={target_idx+1} individuos) alcanzó"
            elif self.stop_mode == 'top95':
                target_idx = max(1, int(self.size_gen * 0.95)) - 1
                msg_parada = f"El Top 95% (>={target_idx+1} individuos) alcanzó"
            elif self.stop_mode == 'top99':
                target_idx = max(1, int(self.size_gen * 0.99)) - 1
                msg_parada = f"El Top 99% (>={target_idx+1} individuos) alcanzó"
            else:
                target_idx = 0
                msg_parada = "El mejor individuo absoluto alcanzó"
                
            value_to_check = sorted_fitness[target_idx]
            
            self.current_gen += 1
            self.evals_this_gen = 0
            self.gen_fitness = []
            self.gen_errors_chance = []
            self.gen_errors_utility = []
            self.gen_entropy_norm = []
            self.gen_util_dev = []
            self.gen_accuracies = []

            if value_to_check <= self.target_fitness:
                raise StopIteration(f"{msg_parada} un Score <= {self.target_fitness:.4f}")
            
        return penalty_score

    def run(self, g=100, i=100, target_fitness=1e-5, patience=10, min_delta=1e-4):
        self.size_gen = g
        
        # --- INICIALIZAR VARIABLES DE ESTANCAMIENTO ---
        self.patience = patience
        self.min_delta = min_delta
        self.stagnation_counter = 0
        self.best_stagnation_fitness = float('inf')
        
        # --- AJUSTE DINÁMICO DE TARGET FITNESS ---
        if self.fitness_type == 'binary':
            self.target_fitness = 0.0
            
        elif self.fitness_type in ['softmax', 'entropy']:
            # Buscamos un ~99% de confianza media. -log(0.99) = 0.01005
            confianza_deseada = 0.99
            nll_esperado = -np.log(confianza_deseada)
            margen_entropia = (self.alpha * 0.05) if self.fitness_type == 'entropy' else 0.0
            
            self.target_fitness = len(self.train_rules) * (nll_esperado + margen_entropia)
            print(f"Meta de Fitness ajustada para {self.fitness_type.upper()}: <= {self.target_fitness:.4f}")
            
        else:
            self.target_fitness = target_fitness
        
        self.best_historical_ind = None
        self.best_historical_fitness = float('inf')
        
        lower_bounds = []
        upper_bounds = []
        for s in self.specs:
            lb = self.chance_init_bounds[0] if s['kind'] == 'chance' else self.utility_init_bounds[0]
            ub = self.chance_init_bounds[1] if s['kind'] == 'chance' else self.utility_init_bounds[1]
            lower_bounds.extend([lb] * s['free_size'])
            upper_bounds.extend([ub] * s['free_size'])
            
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

        optimizer_kwargs = {
            'size_gen': g,
            'max_iter': i,
            'dead_iter': min(20, i),
            'n_variables': self.total_vars,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds,
            'alpha': self.alpha,
            'elite_factor': self.elite_factor,
            'disp': False
        }

        if self.optimizer_type == 'umda':
            optimizer = UMDAc(**optimizer_kwargs)
        elif self.optimizer_type == 'egna':
            optimizer = EGNA(**optimizer_kwargs)
        elif self.optimizer_type == 'emna':
            optimizer = EMNA(**optimizer_kwargs)
        else:
            raise ValueError(f"Optimizador '{self.optimizer_type}' no reconocido.")
        
        print("Iniciando optimización...")
        
        try:
            res = optimizer.minimize(self.fitness)
            mejor_vector = res.best_ind
            print("Optimización terminada por fin natural (max_iter o dead_iter interno del optimizador).")
        except StopIteration as e:
            print(f"\n¡PARADA ANTICIPADA! {e}")
            mejor_vector = self.best_historical_ind
            self.stagnation_counter = 0

        return mejor_vector


class AveragedExperiment:
    """Objeto contenedor que imita a IDRecovery para inyectarlo en tus gráficas."""
    def __init__(self, history, **kwargs):
        self.history = history
        for key, value in kwargs.items():
            setattr(self, key, value)


class BatchExperimenter:
    def __init__(self, n_experiments=5, shuffle_rules_per_run=True, base_seed=42, **id_recovery_kwargs):
        self.n_experiments = n_experiments
        self.shuffle_rules_per_run = shuffle_rules_per_run
        self.base_seed = base_seed
        self.id_recovery_kwargs = id_recovery_kwargs
        
        self.experiments = []
        self.best_overall_vector = None
        self.best_overall_fitness = float('inf')

    def run_batch(self, g=100, i=100, target_fitness=1e-5):
        print(f"--- Iniciando Lote de {self.n_experiments} Experimentos ---")
        
        for exp_idx in range(self.n_experiments):
            print(f"\n>> Ejecutando Experimento {exp_idx + 1}/{self.n_experiments}")
            
            # Determinamos la semilla para esta ejecución
            current_seed = self.base_seed + exp_idx if self.shuffle_rules_per_run else self.base_seed
            
            # Copiamos los argumentos e inyectamos el random_seed
            kwargs_for_this_run = self.id_recovery_kwargs.copy()
            kwargs_for_this_run['random_seed'] = current_seed
                
            # Instanciamos y corremos el experimento
            exp = IDRecovery(**kwargs_for_this_run)
            best_vector = exp.run(g=g, i=i, target_fitness=target_fitness)
            
            self.experiments.append(exp)
            
            # Rastreamos el mejor individuo absoluto de todo el lote
            if exp.best_historical_fitness < self.best_overall_fitness:
                self.best_overall_fitness = exp.best_historical_fitness
                self.best_overall_vector = best_vector

        print("\n--- Lote finalizado. Promediando resultados ---")
        averaged_obj = self._average_histories()
        
        return averaged_obj, self.best_overall_vector

    def _average_histories(self):
        # Encontramos la generación máxima alcanzada (por si algunos terminaron antes)
        max_gens = max(len(exp.history) for exp in self.experiments)
        avg_history = []

        for gen_idx in range(max_gens):
            gen_fitness_list = []
            gen_errors_chance_list = []
            gen_errors_utility_list = []
            gen_accs_list = []

            for exp in self.experiments:
                # Si el experimento paró por 'stop_mode', usamos su última generación registrada (padding)
                hist_idx = min(gen_idx, len(exp.history) - 1)
                gen_data = exp.history[hist_idx]

                gen_fitness_list.append(gen_data['fitness'])
                gen_errors_chance_list.append(gen_data['errors_chance'])
                gen_errors_utility_list.append(gen_data['errors_utility'])
                gen_accs_list.append(gen_data['accuracies'])

            # Promediamos arreglos de numpy a lo largo del eje 0 (los individuos)
            avg_history.append({
                'gen': gen_idx + 1,
                'fitness': np.mean(gen_fitness_list, axis=0),
                'errors_chance': np.mean(gen_errors_chance_list, axis=0),
                'errors_utility': np.mean(gen_errors_utility_list, axis=0),
                'accuracies': np.mean(gen_accs_list, axis=0),
                'fitness_std': np.std(gen_fitness_list, axis=0),
                'accuracies_std': np.std(gen_accs_list, axis=0)
            })

        return AveragedExperiment(avg_history, **self.id_recovery_kwargs)
