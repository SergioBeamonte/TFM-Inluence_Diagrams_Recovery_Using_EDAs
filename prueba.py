import pyAgrum as gum
import numpy as np
from EDAspy.optimization import UMDAd  # Algoritmo UMDA discreto para variables binarias

# ==========================================
# 1. CONSTRUCCIÓN DE LA RED BAYESIANA
# ==========================================
bn = gum.BayesNet("Bypass2_Optimization")

# Nodos de Azar (Chance)
hd = bn.add(gum.LabelizedVariable("HEARTDISEASE", "", ["ABSENT", "PRESENT"]))
pain = bn.add(gum.LabelizedVariable("PAIN", "", ["ABSENT", "PRESENT"]))
angio = bn.add(gum.LabelizedVariable("ANGIOGRAM", "", ["NEGATIVE", "POSITIVE"]))
early = bn.add(gum.LabelizedVariable("EARLYRESULTS", "", ["CoRem", "PaRem", "NoChng", "PrgrsvDisease"]))
life = bn.add(gum.LabelizedVariable("LIFEQ", "", ["DEAD", "LIVE2ALQ", "LIVE2AHQ"]))
eco = bn.add(gum.LabelizedVariable("ECONOMICALC", "", ["LOW", "MEDIUM", "HIGH"]))

# Nodos de Decisión transformados a nodos de Azar binarios (NO=0, YES=1)
surg = bn.add(gum.LabelizedVariable("HEARTSURGERY", "", ["NO", "YES"]))
pharma = bn.add(gum.LabelizedVariable("HEARTPHARMA", "", ["NO", "YES"]))

# Estructura de la red
bn.addArc(hd, pain)
bn.addArc(hd, angio)
bn.addArc(hd, life)

bn.addArc(pain, surg)
bn.addArc(angio, surg)

bn.addArc(angio, early)
bn.addArc(surg, early)

bn.addArc(pain, pharma)
bn.addArc(angio, pharma)
bn.addArc(early, pharma)

bn.addArc(surg, life)

bn.addArc(early, eco)
bn.addArc(surg, eco)
bn.addArc(pharma, eco)

# Llenado de CPTs exactas según el archivo R
bn.cpt(hd)[:] = [0.86, 0.14]

bn.cpt(pain)[{'HEARTDISEASE': 'ABSENT'}] = [0.80, 0.20]
bn.cpt(pain)[{'HEARTDISEASE': 'PRESENT'}] = [0.20, 0.80]

bn.cpt(angio)[{'HEARTDISEASE': 'ABSENT'}] = [0.95, 0.05]
bn.cpt(angio)[{'HEARTDISEASE': 'PRESENT'}] = [0.14, 0.86]

bn.cpt(early)[{'ANGIOGRAM': 'NEGATIVE', 'HEARTSURGERY': 'NO'}] = [0.97, 0.01, 0.01, 0.01]
bn.cpt(early)[{'ANGIOGRAM': 'NEGATIVE', 'HEARTSURGERY': 'YES'}] = [0.95, 0.03, 0.01, 0.01]
bn.cpt(early)[{'ANGIOGRAM': 'POSITIVE', 'HEARTSURGERY': 'NO'}] = [0.01, 0.04, 0.10, 0.80]
bn.cpt(early)[{'ANGIOGRAM': 'POSITIVE', 'HEARTSURGERY': 'YES'}] = [0.55, 0.20, 0.15, 0.10]

bn.cpt(life)[{'HEARTDISEASE': 'ABSENT', 'HEARTSURGERY': 'NO'}] = [0.01, 0.08, 0.91]
bn.cpt(life)[{'HEARTDISEASE': 'ABSENT', 'HEARTSURGERY': 'YES'}] = [0.05, 0.65, 0.30]
bn.cpt(life)[{'HEARTDISEASE': 'PRESENT', 'HEARTSURGERY': 'NO'}] = [0.45, 0.40, 0.15]
bn.cpt(life)[{'HEARTDISEASE': 'PRESENT', 'HEARTSURGERY': 'YES'}] = [0.20, 0.20, 0.60]

# Matriz ECONOMICALC (16 combinaciones)
eco_data = [
    0.90, 0.05, 0.05,  # CR, NO, NO
    0.80, 0.10, 0.10,  # CR, NO, YS
    0.50, 0.25, 0.25,  # CR, YS, NO
    0.45, 0.25, 0.30,  # CR, YS, YS
    0.85, 0.10, 0.05,  # PR, NO, NO
    0.95, 0.03, 0.02,  # PR, NO, YS
    0.45, 0.30, 0.25,  # PR, YS, NO
    0.40, 0.30, 0.30,  # PR, YS, YS
    0.80, 0.10, 0.10,  # NC, NO, NO
    0.90, 0.05, 0.05,  # NC, NO, YS
    0.40, 0.35, 0.25,  # NC, YS, NO
    0.35, 0.30, 0.35,  # NC, YS, YS
    0.75, 0.10, 0.15,  # PD, NO, NO
    0.85, 0.10, 0.05,  # PD, NO, YS
    0.35, 0.35, 0.30,  # PD, YS, NO
    0.30, 0.35, 0.35   # PD, YS, YS
]
bn.cpt(eco)[:] = np.array(eco_data).reshape(16, 3)

# Tabla de Utilidad (No es un nodo en la BN de inferencia, es un diccionario para evaluar)
utility_table = {
    ('DEAD', 'LOW'): 2.0, ('DEAD', 'MEDIUM'): 0.70, ('DEAD', 'HIGH'): 0.05,
    ('LIVE2ALQ', 'LOW'): 3.10, ('LIVE2ALQ', 'MEDIUM'): 2.90, ('LIVE2ALQ', 'HIGH'): 2.80,
    ('LIVE2AHQ', 'LOW'): 7.80, ('LIVE2AHQ', 'MEDIUM'): 4.80, ('LIVE2AHQ', 'HIGH'): 2.80
}

# ==========================================
# 2. DEFINICIÓN DE LA FUNCIÓN FITNESS (EDAspy)
# ==========================================
# El genoma tiene 20 bits: 
# - 4 bits para HEARTSURGERY (2 PAIN * 2 ANGIOGRAM)
# - 16 bits para HEARTPHARMA (2 PAIN * 2 ANGIOGRAM * 4 EARLYRESULTS)

def fitness_function(policy_vector):
    """
    Inyecta la política propuesta por el EDA en la Red Bayesiana, 
    calcula la probabilidad conjunta de LIFEQ y ECONOMICALC, 
    y devuelve la Utilidad Esperada.
    """
    surg_policy = policy_vector[0:4]
    pharma_policy = policy_vector[4:20]
    
    # Inyectar política determinista (0 o 1) en el nodo HEARTSURGERY
    idx = 0
    for p in ["ABSENT", "PRESENT"]:
        for a in ["NEGATIVE", "POSITIVE"]:
            decision = int(surg_policy[idx])
            prob = [1, 0] if decision == 0 else [0, 1]
            bn.cpt(surg)[{'PAIN': p, 'ANGIOGRAM': a}] = prob
            idx += 1
            
    # Inyectar política determinista en el nodo HEARTPHARMA
    idx = 0
    for p in ["ABSENT", "PRESENT"]:
        for a in ["NEGATIVE", "POSITIVE"]:
            for e in ["CoRem", "PaRem", "NoChng", "PrgrsvDisease"]:
                decision = int(pharma_policy[idx])
                prob = [1, 0] if decision == 0 else [0, 1]
                bn.cpt(pharma)[{'PAIN': p, 'ANGIOGRAM': a, 'EARLYRESULTS': e}] = prob
                idx += 1

    # Inferencia exacta
    ie = gum.LazyPropagation(bn)
    ie.makeInference()
    
    # Calcular Utilidad Esperada marginalizando sobre LIFEQ y ECONOMICALC
    # Como la BN asume que todas las variables están conectadas causalmente, 
    # extraemos la probabilidad conjunta de P(LIFEQ, ECONOMICALC)
    joint_pot = ie.jointPosterior({"LIFEQ", "ECONOMICALC"})
    
    expected_utility = 0.0
    for l_idx, l_val in enumerate(["DEAD", "LIVE2ALQ", "LIVE2AHQ"]):
        for e_idx, e_val in enumerate(["LOW", "MEDIUM", "HIGH"]):
            p = joint_pot[{'LIFEQ': l_val, 'ECONOMICALC': e_val}]
            u = utility_table[(l_val, e_val)]
            expected_utility += p * u
            
    # EDAspy minimiza por defecto, así que devolvemos el valor negativo
    return -expected_utility

# ==========================================
# 3. EJECUCIÓN DEL ALGORITMO EDA
# ==========================================
# Configuramos el espacio de búsqueda discreto (20 variables binarias)
n_variables = 20
# Dominio para cada variable: opciones 0 y 1
domain = [[0, 1] for _ in range(n_variables)]

# Instanciamos el modelo UMDA discreto
umda = UMDAd(size_gen=50, max_iter=100, dead_iter=10, n_variables=n_variables, domain=domain)

print("Iniciando evolución con EDAspy...")
eda_result = umda.minimize(fitness_function, True)

print("\n--- RESULTADOS ---")
print(f"Mejor Utilidad Esperada: {-eda_result.best_cost:.4f}")

# Extraer y mostrar las reglas de la mejor solución encontrada
best_policy = eda_result.best_ind

print("\nReglas óptimas para HEARTSURGERY:")
idx = 0
for p in ["ABSENT", "PRESENT"]:
    for a in ["NEGATIVE", "POSITIVE"]:
        decision = "YES" if best_policy[idx] == 1 else "NO"
        print(f"Si PAIN={p} y ANGIOGRAM={a} -> Cirugía: {decision}")
        idx += 1