import csv
import os
import itertools
import random
import numpy as np
import pysmile
import pysmile_license  # Asegúrate de mantener tu licencia

class RuleGenerator:
    """
    Generador de reglas basado en la evaluación nativa de pySMILE.
    Evalúa el Diagrama de Influencia completo, extrae la política óptima
    y permite exportar una muestra aleatoria del total de reglas posibles.
    """
    def __init__(self, xdsl_path: str):
        if not os.path.exists(xdsl_path):
            raise FileNotFoundError(f"No se encontró el archivo: {xdsl_path}")

        self.net = pysmile.Network()
        self.net.read_file(xdsl_path)
        
        print(f"[i] Evaluando la red bayesiana de {xdsl_path}...")
        self.net.update_beliefs()
        print("[+] Evaluación completada.")

    def _get_chance_nodes(self) -> list:
        names = []
        possible_types = ["CPT", "TRUTHTABLE", "TRUTH_TABLE", "NOISY_MAX", "NOISY_ADD", "EQUATION"]
        valid_types = [getattr(pysmile.NodeType, t) for t in possible_types if hasattr(pysmile.NodeType, t)]
        
        for handle in self.net.get_all_nodes():
            if self.net.get_node_type(handle) in valid_types:
                names.append(self.net.get_node_id(handle))
        return names

    def _get_decision_nodes(self) -> list:
        names = []
        possible_types = ["LIST", "DECISION"]
        valid_types = [getattr(pysmile.NodeType, t) for t in possible_types if hasattr(pysmile.NodeType, t)]
        
        for handle in self.net.get_all_nodes():
            if self.net.get_node_type(handle) in valid_types:
                names.append(self.net.get_node_id(handle))
        return names

    def generate_csv(self, n_rules: int, output_path: str):
        """
        Genera todas las reglas posibles y selecciona 'n_rules' al azar.
        Si 'n_rules' es mayor que el total de combinaciones, exporta todas.
        """
        chance_nodes = sorted(self._get_chance_nodes())
        decision_nodes = sorted(self._get_decision_nodes())
        node_order = chance_nodes + decision_nodes

        all_rules = [] # Aquí guardaremos TODAS las reglas posibles antes de filtrar

        for dn_name in decision_nodes:
            dn_handle = self.net.get_node(dn_name)
            parent_handles = self.net.get_parents(dn_handle)
            
            eu_table = self.net.get_node_value(dn_handle)
            dn_state_count = self.net.get_outcome_count(dn_handle)

            if not parent_handles:
                best_action_idx = np.argmax(eu_table)
                row = self._create_row(node_order, dn_name, best_action_idx, [], [])
                all_rules.append(row)
                continue

            parent_state_counts = [self.net.get_outcome_count(p) for p in parent_handles]
            parent_ranges = [range(c) for c in parent_state_counts]
            parent_names = [self.net.get_node_id(p) for p in parent_handles]

            combinations = list(itertools.product(*parent_ranges))
            
            for i, combo in enumerate(combinations):
                start_idx = i * dn_state_count
                end_idx = start_idx + dn_state_count
                action_utilities = eu_table[start_idx:end_idx]
                
                best_action_idx = np.argmax(action_utilities)
                row = self._create_row(node_order, dn_name, best_action_idx, parent_names, combo)
                all_rules.append(row)

        # ---- LÓGICA DE SELECCIÓN ALEATORIA Y LÍMITES ----
        total_possible = len(all_rules)
        
        if n_rules >= total_possible:
            print(f"[i] Se pidieron {n_rules} reglas, pero el máximo posible es {total_possible}.")
            print(f"[i] Ajustando automáticamente para exportar las {total_possible} reglas.")
            final_rules = all_rules
        else:
            print(f"[i] Seleccionando {n_rules} reglas aleatoriamente de un total de {total_possible} posibles.")
            final_rules = random.sample(all_rules, n_rules)

        # ---- ESCRITURA DEL CSV ----
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(node_order)
            writer.writerows(final_rules)

        print(f"[+] Exportadas {len(final_rules)} reglas en: {output_path}")

    def _create_row(self, node_order: list, target_dn: str, action_idx: int, parent_names: list, parent_combo: tuple) -> list:
        row = []
        parent_dict = dict(zip(parent_names, parent_combo))
        
        for name in node_order:
            if name == target_dn:
                row.append(-(action_idx + 1))
            elif name in parent_dict:
                row.append(parent_dict[name] + 1)
            else:
                row.append(0)
        return row

    def export_mappings(self, output_path: str = "mappings.txt"):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("LEYENDA DE CODIFICACIÓN PARA REGLAS CSV\n")
            f.write("========================================\n\n")
            f.write("REGLA GENERAL:\n")
            f.write("1. VALORES POSITIVOS (>0): Nodo de Evidencia Observada (Input).\n")
            f.write("   Estado = Índice del valor - 1\n")
            f.write("2. VALORES NEGATIVOS (<0): Nodo de Decisión Óptima (Output).\n")
            f.write("   Acción = Índice del valor absoluto - 1\n")
            f.write("3. VALOR CERO (0): El nodo no es relevante para la regla actual.\n\n")
            
            chance_nodes = self._get_chance_nodes()
            decision_nodes = self._get_decision_nodes()
            all_nodes = sorted(chance_nodes + decision_nodes)

            for name in all_nodes:
                handle = self.net.get_node(name)
                node_type = "DECISION" if name in decision_nodes else "CHANCE"
                
                f.write(f"Nodo: {name} ({node_type})\n")
                for i in range(self.net.get_outcome_count(handle)):
                    state_name = self.net.get_outcome_id(handle, i)
                    f.write(f"  {i+1} : {state_name}\n")
                f.write("-" * 30 + "\n")

    def export_utility_tables(self, output_path: str = "utility_tables.txt"):
        """
        Genera un archivo de texto con las tablas de utilidad de cada nodo de decisión.
        Al finalizar, extrae e imprime el MÁXIMO y MÍNIMO absolutos de TODA la red,
        mostrando qué decisión y qué estados de los nodos condicionantes lo provocan.
        """

        decision_nodes = sorted(self._get_decision_nodes())
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Variables para rastrear el récord de toda la red
        net_max_val = -float('inf')
        net_min_val = float('inf')
        net_max_info = {}
        net_min_info = {}

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("TABLAS DE UTILIDAD ESPERADA POR NODO DE DECISIÓN\n")
            f.write("=" * 60 + "\n\n")

            for dn_name in decision_nodes:
                dn_handle = self.net.get_node(dn_name)
                parent_handles = self.net.get_parents(dn_handle)

                dn_state_count = self.net.get_outcome_count(dn_handle)
                dn_states = [self.net.get_outcome_id(dn_handle, i) for i in range(dn_state_count)]
                
                f.write(f"--- NODO DE DECISIÓN: {dn_name.upper()} ---\n")
                eu_table = self.net.get_node_value(dn_handle)

                # --- LÓGICA PARA NODOS SIN PADRES ---
                if not parent_handles:
                    valid_utilities = eu_table[:dn_state_count]
                    
                    f.write("[Sin nodos padre / Decisión incondicional]\n\n")
                    header_actions = "".join([f"{state:>15}" for state in dn_states])
                    f.write(header_actions + "\n")
                    f.write("-" * (15 * len(dn_states)) + "\n")
                    row_vals = "".join([f"{val:>15.4f}" for val in valid_utilities])
                    f.write(row_vals + "\n\n")

                    # Evaluación global
                    for j, val in enumerate(valid_utilities):
                        if val > net_max_val:
                            net_max_val = val
                            net_max_info = {'dn': dn_name, 'action': dn_states[j], 'parents': {}}
                        if val < net_min_val:
                            net_min_val = val
                            net_min_info = {'dn': dn_name, 'action': dn_states[j], 'parents': {}}
                    continue

                # --- LÓGICA PARA NODOS CON PADRES ---
                parent_names = [self.net.get_node_id(p) for p in parent_handles]
                parent_state_lists = []
                
                for p in parent_handles:
                    count = self.net.get_outcome_count(p)
                    states = [self.net.get_outcome_id(p, i) for i in range(count)]
                    parent_state_lists.append(states)

                combinations = list(itertools.product(*parent_state_lists))
                header_parents = " | ".join(parent_names)
                pad_parents = max(len(header_parents), max([len(" | ".join(c)) for c in combinations]) if combinations else 0) + 2
                header_actions = "".join([f"{state:>12}" for state in dn_states])
                
                f.write(f"  # | {header_parents:<{pad_parents}} || {header_actions}\n")
                f.write("-" * (6 + pad_parents + 4 + (12 * len(dn_states))) + "\n")

                for i, combo in enumerate(combinations):
                    start_idx = i * dn_state_count
                    end_idx = start_idx + dn_state_count
                    action_utilities = eu_table[start_idx:end_idx]

                    combo_str = " | ".join(combo)
                    best_local_idx = int(np.argmax(action_utilities))
                    
                    utils_str_marked = "".join([
                        f"*{val:>11.4f}" if j == best_local_idx else f"{val:>12.4f}" 
                        for j, val in enumerate(action_utilities)
                    ])

                    f.write(f"{i+1:>3} | {combo_str:<{pad_parents}} || {utils_str_marked}\n")

                    # Evaluación global de cada valor individual en esta fila
                    for j, val in enumerate(action_utilities):
                        if val > net_max_val:
                            net_max_val = val
                            net_max_info = {'dn': dn_name, 'action': dn_states[j], 'parents': dict(zip(parent_names, combo))}
                        if val < net_min_val:
                            net_min_val = val
                            net_min_info = {'dn': dn_name, 'action': dn_states[j], 'parents': dict(zip(parent_names, combo))}

                f.write("\n" + "=" * 60 + "\n\n")

            # --- ESCRITURA EN TXT DE LOS EXTREMOS DE TODA LA RED ---
            f.write("============================================================\n")
            f.write("EXTREMOS GLOBALES ABSOLUTOS DE TODA LA RED\n")
            f.write("============================================================\n\n")
            
            f.write(f"[+] MÁXIMO ABSOLUTO: {net_max_val:.4f}\n")
            f.write(f"    - Nodo de Decisión: {net_max_info['dn']} (Acción: {net_max_info['action']})\n")
            if net_max_info['parents']:
                f.write("    - Nodos Condicionantes (Padres):\n")
                for p_name, p_state in net_max_info['parents'].items():
                    f.write(f"        * {p_name}: {p_state}\n")
            else:
                f.write("    - Nodos Condicionantes: Ninguno (Decisión incondicional)\n")

            f.write(f"\n[-] MÍNIMO ABSOLUTO: {net_min_val:.4f}\n")
            f.write(f"    - Nodo de Decisión: {net_min_info['dn']} (Acción: {net_min_info['action']})\n")
            if net_min_info['parents']:
                f.write("    - Nodos Condicionantes (Padres):\n")
                for p_name, p_state in net_min_info['parents'].items():
                    f.write(f"        * {p_name}: {p_state}\n")
            else:
                f.write("    - Nodos Condicionantes: Ninguno (Decisión incondicional)\n")

        # --- SALIDA POR PANTALLA ---
        print(f"[+] Tablas de utilidad exportadas en: {output_path}")
        
        print("\n" + "*" * 60)
        print(" EXTREMOS GLOBALES ABSOLUTOS DE TODA LA RED")
        print("*" * 60)
        
        print(f"\n[+] MÁXIMO ABSOLUTO: {net_max_val:.4f}")
        print(f"    Decisión tomada: {net_max_info['dn']} -> {net_max_info['action']}")
        print("    Valores de los nodos que definen esta utilidad:")
        if net_max_info['parents']:
            for p_name, p_state in net_max_info['parents'].items():
                print(f"      - {p_name}: {p_state}")
        else:
            print("      - (Sin nodos condicionantes)")

        print(f"\n[-] MÍNIMO ABSOLUTO: {net_min_val:.4f}")
        print(f"    Decisión tomada: {net_min_info['dn']} -> {net_min_info['action']}")
        print("    Valores de los nodos que definen esta utilidad:")
        if net_min_info['parents']:
            for p_name, p_state in net_min_info['parents'].items():
                print(f"      - {p_name}: {p_state}")
        else:
            print("      - (Sin nodos condicionantes)")
        print("\n" + "*" * 60 + "\n")

if __name__ == "__main__":
    try:
        gen = RuleGenerator(r"example\nhlv1\network-nhlv1.xdsl")
        gen.export_mappings(r"example\nhlv1\rule_mappings.txt")
        gen.export_utility_tables(r"example\nhlv1\utility_tables.txt")
        
        gen.generate_csv(n_rules=75, output_path=r"example\nhlv1\reglas_generadas.csv")
        
    except Exception as e:
        print(f"[!] Error crítico: {e}")