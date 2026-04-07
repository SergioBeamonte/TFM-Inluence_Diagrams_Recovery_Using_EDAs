import os
import csv
import numpy as np
import pysmile
import pysmile_license

from EDAspy.optimization.univariate import UMDAc
from extractor import NetworkExtractor
from engines import ShachterEngine
from models import NodeKind

class IDReconstructor:
    def __init__(self, xdsl_path, rules_csv, best_util_config, worst_util_config):
        self.xdsl_path = xdsl_path
        self.rules_csv = rules_csv
        self.best_util_config = best_util_config
        self.worst_util_config = worst_util_config
        
        # 1. Load initial model
        self.net = pysmile.Network()
        self.net.read_file(xdsl_path)
        self.nodes = NetworkExtractor.extract(self.net)
        
        # 2. Parse rules
        self.rules = self._parse_rules(rules_csv)
        
        # 3. Identify parameters to optimize
        self.param_specs = self._get_param_specs()
        self.total_vars = sum(spec['size'] for spec in self.param_specs)
        
        print(f"[+] Loaded {len(self.rules)} rules.")
        print(f"[+] Total variables to optimize: {self.total_vars}")

    def _parse_rules(self, csv_path):
        rules = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                evidence = {}
                target_decision = None
                target_action = None
                
                for node_name, val_str in row.items():
                    val = int(val_str)
                    if val == 0:
                        continue # Not part of the rule
                    
                    nd = self.nodes[node_name]
                    if val > 0:
                        # State index val-1
                        evidence[node_name] = nd.states[val - 1]
                    elif val < 0:
                        # Target decision: Index |val|-1
                        target_decision = node_name
                        target_action = nd.states[abs(val) - 1]
                
                if target_decision:
                    rules.append((evidence, target_decision, target_action))
        return rules

    def _get_param_specs(self):
        specs = []
        # We optimize all CHANCE node tables and UTILITY tables
        for name, nd in self.nodes.items():
            if nd.kind == NodeKind.CHANCE:
                specs.append({
                    'name': name,
                    'kind': 'chance',
                    'size': nd.table.size,
                    'shape': nd.table.shape
                })
            elif nd.kind == NodeKind.UTILITY:
                # We subtract 2 from size because Best/Worst are fixed
                # Wait, it's easier to optimize the full table and then overwrite.
                # Or optimize size entries and just ignore two?
                # Optimization is smoother if we let it optimize all and then overwrite 
                # OR only translate certain indices.
                # Let's optimize total size and overwrite.
                specs.append({
                    'name': name,
                    'kind': 'utility',
                    'size': nd.table.size,
                    'shape': nd.table.shape
                })
        return specs

    def _vector_to_nodes(self, vector):
        pos = 0
        for spec in self.param_specs:
            name = spec['name']
            nd = self.nodes[name]
            size = spec['size']
            raw_vals = np.clip(vector[pos : pos + size].copy(), 0, 1)
            pos += size
            
            if spec['kind'] == 'chance':
                # Normalization row by row
                # Table shape is (*parents, own_states)
                n_states = nd.table.shape[-1]
                reshaped = raw_vals.reshape(spec['shape'])
                
                # Apply softmax or simple division
                # To ensure no division by zero, add epsilon
                row_sums = reshaped.sum(axis=-1, keepdims=True)
                # Use uniform distribution where sum is 0
                zero_rows = (row_sums == 0).squeeze(-1)
                reshaped[zero_rows] = 1.0 / n_states
                row_sums[row_sums == 0] = 1.0 # Update row_sums to 1.0 for these rows
                nd.table = reshaped / row_sums
                
            elif spec['kind'] == 'utility':
                # Scaling [0, 1] -> [0, 10]
                reshaped = (raw_vals.reshape(spec['shape'])) * 10.0
                
                # Apply Best/Worst constraints
                # Identify indices for best/worst configs
                best_idx = self._get_table_idx(nd, self.best_util_config)
                worst_idx = self._get_table_idx(nd, self.worst_util_config)
                
                reshaped[best_idx] = 10.0
                reshaped[worst_idx] = 0.0
                
                nd.table = reshaped

    def _get_table_idx(self, nd, config):
        # Parents order in nd.parents
        # Find combination indices
        idx = []
        for p_name in nd.parents:
            state = config[p_name]
            idx.append(self.nodes[p_name].states.index(state))
        return tuple(idx)

    def fitness(self, solution):
        # solution is a vector of [0, 1]
        self._vector_to_nodes(solution)
        
        # Evaluate rules
        correct = 0
        engine = ShachterEngine()
        
        for evidence, target_node, expected_action in self.rules:
            try:
                # Note: evaluate modifies the nodes internal tables if needed?
                # ShachterEngine.evaluate uses a copy of nodes, but it uses the Table values.
                # We updated the table values in self._vector_to_nodes.
                res = engine.evaluate(self.nodes, evidence, self.net)
                
                if res.optimal_decisions.get(target_node) == expected_action:
                    correct += 1
            except Exception as e:
                # In case of evaluation failure (e.g. cycle created by reversal?)
                pass
        
        accuracy = correct / len(self.rules)
        return 1.0 - accuracy # Minimize cost

    def run(self, size_gen=50, max_iter=30):
        # Configure UMDAc
        # Note: lower_bound and upper_bound can be vectors
        umda = UMDAc(
            size_gen=size_gen,
            max_iter=max_iter,
            dead_iter=10,
            n_variables=self.total_vars,
            lower_bound=0.0,
            upper_bound=1.0
        )
        
        print(f"[*] Starting UMDAc optimization (gen={size_gen}, iter={max_iter})...")
        result = umda.minimize(self.fitness, verbose=True)
        
        print("\n--- OPTIMIZATION FINISHED ---")
        print(f"Best Accuracy: {1.0 - result.best_cost:.4f}")
        
        # Apply best solution to current nodes
        self._vector_to_nodes(result.best_ind)
        return result

    def save_model(self, output_path):
        # Update pysmile network with new parameters
        for name, nd in self.nodes.items():
            if nd.kind == NodeKind.DECISION:
                continue
            
            h = self.net.get_node(name)
            # Flatten table for pysmile
            flat_table = nd.table.flatten().tolist()
            self.net.set_node_definition(h, flat_table)
            
        self.net.write_file(output_path)
        print(f"[+] Saved reconstructed model to: {output_path}")

if __name__ == "__main__":
    XDSL_PATH = "copia_modelo.xdsl"
    RULES_CSV = "reglas_ejemplo.csv"
    
    # User-requested constraints
    BEST_CONFIG = {"LIFEQ": "LIVE2AHQ", "ECONOMICALC": "LOW"}
    WORST_CONFIG = {"LIFEQ": "DEAD", "ECONOMICALC": "HIGH"}
    
    reconstructor = IDReconstructor(XDSL_PATH, RULES_CSV, BEST_CONFIG, WORST_CONFIG)
    reconstructor.run(size_gen=40, max_iter=20) # Small values for quick test
    reconstructor.save_model("reconstructed_model.xdsl")
