# RUNBOOK — Experimento de varianza NHLv (segundo ordenador)

Este equipo corre **solo el experimento de NHLv**, en paralelo al de bypass que se
está ejecutando en la otra máquina. Cada uno escribe un CSV distinto, así que no
chocan.

## Qué hace

`explore_variance_nhlv.py` — descompone la varianza del *accuracy* de la
recuperación en NHLv (`utility_only`) según dos fuentes:
- **qué reglas** se eliciten (`rule_seed`), y
- **la semilla de inicialización** del EDA (`random_seed`).

Diseño (Plan B): 3 conjuntos de reglas × 3 inits × 5 fitness × {UMDA, EMNA} =
**90 runs**, 10% de reglas (7 de 67), parada `top90`, `min_iter=1`, cap 40 gen.
ETA ≈ 11–13 h. En `utility_only` las CPTs están fijas → **no hay MSE a
probabilidades**, solo varianza de accuracy.

## Setup desde cero (equipo que "no tiene nada")

Requisito: **Windows 64-bit + Python 3.13** (para el wheel vendorizado de pysmile).
Todo lo demás está en el repo (código, licencia de pysmile, ficheros de red, el
propio wheel de pysmile).

```
git clone https://github.com/SergioBeamonte/TFM.git
cd TFM
python -m venv .venv && .venv\Scripts\activate        # recomendado (evitar líos de OneDrive)
pip install -r requirements.txt
pip install vendor\pysmile-2.4.0-cp313-cp313-win_amd64.whl
```

Comprobación de que el entorno está OK (debe imprimir `OK`):
```
python -c "import pysmile_license, pysmile; from EDAspy.optimization import UMDAc, EGNA, EMNA, UnivariateKEDA; print('OK')"
```

Todo lo necesario ya está versionado y llega con el clone:
- `id_recovery.py` (con `rule_seed`), `explore_variance_nhlv.py`
- `pysmile_license.py` (la clave), `vendor\pysmile-...whl` (el binario)
- `example\nhlv1\network-nhlv1.xdsl`, `example\nhlv1\reglas_generadas.csv`

Notas:
- `pgmpy` es necesario aunque NHLv solo use UMDA/EMNA, porque `id_recovery` importa
  EGNA/KEDA al cargar el módulo.
- Si el equipo NO es Windows+Python3.13, el wheel vendorizado no sirve: hay que
  instalar el `pysmile` de BayesFusion para esa plataforma (necesita licencia).
- Si ya tenías el repo clonado, basta `git pull` en vez del clone.

## Ejecutar

```
python explore_variance_nhlv.py
```

- Es **reanudable**: escribe `example\explore_variance_nhlv.csv` tras cada run; si se
  corta, relanzar el mismo comando continúa donde iba.
- Ver progreso sin parar nada: contar filas de ese CSV (llega hasta 90).
- Al terminar, la descomposición:
  ```
  python explore_variance_nhlv.py --summary
  ```

## Devolver resultados a la máquina principal

```
git add example\explore_variance_nhlv.csv example\explore_variance_nhlv_summary.csv
git commit -m "nhlv variance: resultados del segundo equipo"
git push
```
(La máquina principal luego hace `git pull` y ya tiene los datos de NHLv.)

## Con Claude Code en este equipo

Abre Claude Code en la carpeta del repo y dile algo como:
> "Lee RUNBOOK_nhlv.md y ejecuta el experimento de NHLv que describe, en segundo
>  plano; avísame cuando termine."

## IMPORTANTE — no duplicar

NHLv debe correr en **una sola máquina**. En la principal se ha dejado
deliberadamente SIN lanzador automático para no ejecutarlo dos veces (dos procesos
escribiendo el mismo CSV vía OneDrive = conflicto). No arranques bypass aquí.
