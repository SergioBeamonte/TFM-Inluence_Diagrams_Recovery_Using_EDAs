"""Dashboard académico para exportar figuras a paper.

Diferencias con dashboard.py:
- matplotlib en vez de altair → figuras vectoriales (SVG/PDF) listas para LaTeX.
- Intervalos de confianza al 95 % (Student-t, N pequeñas) en lugar de mean ± σ.
- Tests pareados (Welch t-test, Bonferroni-corregido) para comparaciones.
- Botones de descarga por gráfico (SVG / PDF / PNG / TIFF a 300 DPI).
- Tablas con exportación a LaTeX (\\booktabs).
- Estilo serif + paleta viridis colorblind-safe.

Lanzar con:  streamlit run dashboard_academic.py
"""
from __future__ import annotations

import glob
import io
import os
from dataclasses import dataclass
from itertools import combinations

import altair as alt  # solo para tooltips de tabla, no figuras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st

# ─── PUBLICATION STYLE ────────────────────────────────────────────────────────

mpl.rcParams.update({
    # Tipografía
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':  'stix',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'legend.frameon':    False,
    # Ejes
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.grid':         True,
    'grid.linestyle':    ':',
    'grid.linewidth':    0.5,
    'grid.alpha':        0.6,
    # Otros
    'figure.dpi':        110,         # render en pantalla
    'savefig.dpi':       300,         # render para descarga
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
})

VIRIDIS = mpl.colormaps['viridis']
CB_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00']

st.set_page_config(page_title="EDA — Vista académica", page_icon="📐", layout="wide")
st.title("📐 EDA — Dashboard académico (exportable a paper)")
st.caption(
    "Intervalos de confianza al 95 % (Student-t), tests pareados con corrección "
    "de Bonferroni, figuras vectoriales y tablas en LaTeX."
)


# ─── DATA LOADING (igual que dashboard.py) ────────────────────────────────────

@st.cache_data(ttl=30)
def discover_models() -> dict:
    found = {}
    for f in glob.glob("**/grid_search_results_*.csv", recursive=True):
        parts = f.replace("\\", "/").split("/")
        if any(p.startswith(".") for p in parts):
            continue
        optimizer = os.path.basename(f).replace("grid_search_results_", "").replace(".csv", "")
        subdir    = os.path.basename(os.path.dirname(f))
        key       = f"{subdir} · {optimizer}"
        curves    = f.replace("results", "curves")
        found[key] = (
            os.path.normpath(f),
            os.path.normpath(curves) if os.path.exists(curves) else None,
        )
    return found


# Algunos CSVs antiguos (nhlv1/UMDA original) usan otro naming. Lo unificamos
# para que el resto del dashboard no tenga que ramificar por modelo.
_RESULTS_RENAME = {
    'best_fitness_mean':     'fitness_mean',
    'best_fitness_std':      'fitness_std',
    'best_accuracy_mean':    'accuracy_mean',
    'best_accuracy_std':     'accuracy_std',
    'best_accuracy_min':     'accuracy_min',
    'best_accuracy_max':     'accuracy_max',
    'best_mse_chance_mean':  'mse_chance_mean',
    'best_mse_chance_std':   'mse_chance_std',
    'best_mse_utility_mean': 'mse_utility_mean',
    'best_mse_utility_std':  'mse_utility_std',
    'best_entropy_norm_mean':'entropy_norm_mean',
    'best_entropy_norm_std': 'entropy_norm_std',
    'best_util_dev_mean':    'util_dev_mean',
    'best_util_dev_std':     'util_dev_std',
}
_RESULTS_DROP_LEGACY = {
    'mean_accuracy_mean', 'mean_accuracy_std',
    'mean_mse_chance_mean', 'mean_mse_chance_std',
    'mean_mse_chance_min', 'mean_mse_chance_max',
    'best_mse_chance_min', 'best_mse_chance_max',
    'mean_mse_utility_mean', 'mean_mse_utility_std',
    'mean_mse_utility_min', 'mean_mse_utility_max',
    'best_mse_utility_min', 'best_mse_utility_max',
    'mean_entropy_norm_mean', 'mean_entropy_norm_std',
    'mean_entropy_norm_min', 'mean_entropy_norm_max',
    'best_entropy_norm_min', 'best_entropy_norm_max',
    'mean_util_dev_mean', 'mean_util_dev_std',
    'mean_util_dev_min', 'mean_util_dev_max',
    'best_util_dev_min', 'best_util_dev_max',
    'mean_fitness',
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas históricas a las canónicas y rellena defaults."""
    df = df.rename(columns={k: v for k, v in _RESULTS_RENAME.items() if k in df.columns})
    df = df.drop(columns=[c for c in _RESULTS_DROP_LEGACY if c in df.columns])
    # Defaults para parámetros nuevos no presentes en CSVs antiguos.
    for col, default in [('mode', 'both'), ('sampling_mode', 'non_symmetric'),
                          ('chance_temperature', 1.0), ('utility_temperature', 1.0)]:
        if col not in df.columns:
            df[col] = default
    # Normaliza tipo de temperaturas a float (algunos CSVs las guardaron como int).
    for col in ('chance_temperature', 'utility_temperature'):
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1.0)
    return df


@st.cache_data
def load_all(models_frozen):
    res_list, cur_list = [], []
    for model, (rp, cp) in models_frozen:
        if rp and os.path.exists(rp):
            df = pd.read_csv(rp)
            if not df.empty:
                df.insert(0, "model", model)
                res_list.append(_normalize_columns(df))
        if cp and os.path.exists(cp):
            df = pd.read_csv(cp)
            if not df.empty:
                df.insert(0, "model", model)
                cur_list.append(_normalize_columns(df))
    df_r = pd.concat(res_list, ignore_index=True) if res_list else pd.DataFrame()
    df_c = pd.concat(cur_list, ignore_index=True) if cur_list else pd.DataFrame()
    return df_r, df_c


models = discover_models()
if not models:
    st.error("No se encontraron `grid_search_results_*.csv`.")
    st.stop()

df_r, df_c = load_all(tuple(sorted(models.items())))


# ─── STATISTICAL HELPERS ──────────────────────────────────────────────────────

def ci95_halfwidth(std: np.ndarray, n: int) -> np.ndarray:
    """Half-width of the 95% confidence interval of the mean (Student-t)."""
    if n < 2:
        return np.zeros_like(std)
    t_val = stats.t.ppf(0.975, n - 1)
    return t_val * std / np.sqrt(n)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d: standardized mean difference (pooled std)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan')
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return float('nan')
    return (a.mean() - b.mean()) / pooled


# ─── FIGURE DOWNLOAD ──────────────────────────────────────────────────────────

@dataclass
class FigBytes:
    svg: bytes
    pdf: bytes
    png: bytes


def fig_to_bytes(fig: plt.Figure) -> FigBytes:
    out = {}
    for fmt in ('svg', 'pdf', 'png'):
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt)
        buf.seek(0)
        out[fmt] = buf.getvalue()
    return FigBytes(**out)


def render_fig(fig: plt.Figure, basename: str, key_prefix: str):
    st.pyplot(fig, clear_figure=False)
    bts = fig_to_bytes(fig)
    c1, c2, c3 = st.columns(3)
    c1.download_button(f"⬇ {basename}.svg", bts.svg, file_name=f"{basename}.svg",
                       mime="image/svg+xml", key=f"{key_prefix}_svg", use_container_width=True)
    c2.download_button(f"⬇ {basename}.pdf", bts.pdf, file_name=f"{basename}.pdf",
                       mime="application/pdf", key=f"{key_prefix}_pdf", use_container_width=True)
    c3.download_button(f"⬇ {basename}.png", bts.png, file_name=f"{basename}.png",
                       mime="image/png", key=f"{key_prefix}_png", use_container_width=True)
    plt.close(fig)


# ─── SIDEBAR FILTERS ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🎛️ Filtros")

    ref = df_r if not df_r.empty else df_c

    all_models  = sorted(ref["model"].unique())
    all_mode    = sorted(ref["mode"].dropna().unique())
    all_samp    = sorted(ref["sampling_mode"].dropna().unique())
    all_fitness = sorted(ref["fitness_type"].dropna().unique())
    all_pct     = sorted(ref["n_decision_rules_pct"].dropna().unique())
    all_tc      = sorted(ref["chance_temperature"].dropna().unique())
    all_tu      = sorted(ref["utility_temperature"].dropna().unique())

    sel_models  = st.multiselect("Modelos", all_models, default=all_models)
    sel_mode    = st.multiselect("Modo descomposición", all_mode, default=all_mode)
    sel_samp    = st.multiselect("Muestreo", all_samp, default=all_samp)
    sel_fitness = st.multiselect("Fitness", all_fitness, default=all_fitness)
    sel_pct     = st.multiselect("% reglas decisión", all_pct, default=all_pct)
    sel_tc      = st.multiselect("T softmax (CPTs)", all_tc, default=all_tc)
    sel_tu      = st.multiselect("T sigmoid (Util.)", all_tu, default=all_tu)

    st.divider()
    st.subheader("📐 Exportación")
    n_reps = st.number_input("N repeticiones (para CI 95 %)", min_value=2, max_value=50,
                             value=5, step=1,
                             help="Si tu grid usa otro N_REPETITIONS, cámbialo aquí.")
    fig_width = st.slider("Ancho figura (in)", 3.0, 12.0, 6.0, 0.5)
    fig_height = st.slider("Alto figura (in)", 2.0, 10.0, 4.0, 0.5)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[
        df["model"].isin(sel_models)
        & df["mode"].isin(sel_mode)
        & df["sampling_mode"].isin(sel_samp)
        & df["fitness_type"].isin(sel_fitness)
        & df["n_decision_rules_pct"].isin(sel_pct)
        & df["chance_temperature"].isin(sel_tc)
        & df["utility_temperature"].isin(sel_tu)
    ].copy()


df_rf = apply_filters(df_r)
df_cf = apply_filters(df_c)

# Filas con accuracy_mean NaN no son útiles para nada — las quitamos para evitar
# romper agregaciones y plots (KEDA con mode=both abortado deja NaN).
n_drop = df_rf['accuracy_mean'].isna().sum() if 'accuracy_mean' in df_rf.columns else 0
if n_drop:
    st.info(f"Excluyendo {n_drop} filas con `accuracy_mean=NaN` (corridas abortadas).")
    df_rf = df_rf.dropna(subset=['accuracy_mean']).reset_index(drop=True)

if df_rf.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

# Pre-cálculo: CI95 a partir de std + n_reps
df_rf['ci95'] = ci95_halfwidth(df_rf['accuracy_std'].fillna(0).values, n_reps)


# ─── TABS ─────────────────────────────────────────────────────────────────────

tab_bars, tab_box, tab_heat, tab_curves, tab_tables = st.tabs([
    "📊 Comparativos (CI 95 %)",
    "📦 Distribuciones",
    "🔥 Heatmaps 2D",
    "📈 Convergencia",
    "📋 Tablas + LaTeX",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BARS WITH CI
# ══════════════════════════════════════════════════════════════════════════════

with tab_bars:
    st.subheader("Accuracy por configuración con intervalo de confianza al 95 %")

    cgrp1, cgrp2 = st.columns(2)
    group_by = cgrp1.selectbox("Agrupar por (eje x)",
                                ['fitness_type', 'mode', 'sampling_mode', 'n_decision_rules_pct',
                                 'chance_temperature', 'utility_temperature'])
    color_by = cgrp2.selectbox("Color (segundo factor)",
                                ['(ninguno)', 'model', 'mode', 'sampling_mode', 'fitness_type'],
                                index=2)

    color_arg = None if color_by == '(ninguno)' else color_by
    if color_arg == group_by:
        st.warning("El eje x y el color son la misma variable. Cambia uno.")
        st.stop()
    agg_cols = [group_by] + ([color_arg] if color_arg else [])
    agg = df_rf.groupby(agg_cols, dropna=False).agg(
        mean=('accuracy_mean', 'mean'),
        std=('accuracy_std', 'mean'),
        n=('accuracy_mean', 'count'),
    ).reset_index()
    agg = agg.dropna(subset=['mean'])  # cells vacías → no plotear
    if agg.empty:
        st.warning("No hay datos para esta combinación de ejes.")
        st.stop()
    # CI conservador: t(n_reps) sobre la std agregada por celda
    agg['ci'] = ci95_halfwidth(agg['std'].fillna(0).values, n_reps)

    # Ancho auto si hay muchas categorías
    n_x = agg[group_by].nunique()
    n_c = agg[color_arg].nunique() if color_arg else 1
    auto_w = max(fig_width, n_x * max(1.0, n_c * 0.45) + 1.5)
    fig, ax = plt.subplots(figsize=(auto_w, fig_height))

    if color_arg is None:
        x_labels = agg[group_by].astype(str).tolist()
        x = np.arange(len(x_labels))
        ax.bar(x, agg['mean'], yerr=agg['ci'], capsize=3,
               color=CB_COLORS[0], edgecolor='black', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=0)
    else:
        groups = sorted(agg[group_by].astype(str).unique())
        subs   = sorted(agg[color_arg].astype(str).unique())
        x = np.arange(len(groups))
        w = 0.8 / max(len(subs), 1)
        # Index por (group, color) para lookup limpio
        agg_idx = agg.set_index([agg[group_by].astype(str), agg[color_arg].astype(str)])
        for i, sub in enumerate(subs):
            means, cis = [], []
            for g in groups:
                key = (g, sub)
                if key in agg_idx.index:
                    row = agg_idx.loc[key]
                    # Si hay duplicados, .loc devuelve DataFrame: tomamos la media
                    if isinstance(row, pd.DataFrame):
                        means.append(row['mean'].mean())
                        cis.append(row['ci'].mean())
                    else:
                        means.append(row['mean'])
                        cis.append(row['ci'])
                else:
                    means.append(np.nan)
                    cis.append(0)
            # NaN → no se dibuja, evitamos barras a 0 engañosas
            xs = x + i * w - 0.4 + w / 2
            mask = ~np.isnan(means)
            ax.bar(np.array(xs)[mask], np.array(means)[mask], w,
                   yerr=np.array(cis)[mask], capsize=2,
                   color=CB_COLORS[i % len(CB_COLORS)], edgecolor='black', linewidth=0.5,
                   label=str(sub))
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=0)
        # Leyenda DEBAJO del plot, horizontal, fuera del área de ejes.
        # Así no tapa barras altas (problema típico con accuracy ≈ 100 %).
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.18),
            ncol=min(len(subs), 4),
            title=color_arg,
            frameon=False,
            borderaxespad=0.0,
        )

    ax.set_xlabel(group_by.replace('_', ' '))
    ax.set_ylabel(r'Accuracy media (\%)')
    ax.set_title(f'Comparación por {group_by} (CI 95 %, N={n_reps})')
    # Y-lim seguro frente a NaN/empty
    y_low = agg['mean'].min()
    if pd.notna(y_low):
        ax.set_ylim(bottom=max(0, y_low - 10))
    # tight_layout no recorta artistas externos; savefig usa bbox='tight' (rcParams).
    fig.tight_layout()

    render_fig(fig, f"bars_{group_by}_{color_arg or 'none'}", key_prefix="bars")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab_box:
    st.subheader("Distribución de accuracy: box + scatter")
    st.caption("Cada punto es una combinación del grid; el box resume el conjunto seleccionado.")

    c0a, c0b = st.columns([2, 1])
    box_by = c0a.selectbox("Factor del eje x",
                            ['fitness_type', 'mode', 'sampling_mode', 'n_decision_rules_pct',
                             'chance_temperature', 'utility_temperature', 'model'])
    box_style = c0b.radio("Estilo", ['box + jitter', 'violin + jitter'], horizontal=True)

    cats = sorted(df_rf[box_by].astype(str).unique())
    data = [df_rf.loc[df_rf[box_by].astype(str) == c, 'accuracy_mean'].values for c in cats]
    n_cats = len(cats)

    # Ancho auto: ~1.6 in por categoría con un mínimo razonable. Ignora el slider
    # global porque el boxplot necesita más aire que las otras figuras.
    auto_w = max(fig_width, n_cats * 1.6 + 1.5)
    auto_h = max(fig_height, 4.5)
    fig, ax = plt.subplots(figsize=(auto_w, auto_h))

    if box_style == 'box + jitter':
        bp = ax.boxplot(data, labels=cats, widths=0.55, showmeans=True,
                        medianprops={'color': '#D55E00', 'linewidth': 1.6},
                        meanprops={'marker': 'D', 'markerfacecolor': 'white',
                                   'markeredgecolor': 'black', 'markersize': 5},
                        boxprops={'linewidth': 0.9, 'facecolor': '#E8F0F8'},
                        whiskerprops={'linewidth': 0.9},
                        capprops={'linewidth': 0.9},
                        flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4},
                        patch_artist=True)
    else:
        parts = ax.violinplot(data, positions=range(1, n_cats + 1), widths=0.7,
                              showmeans=False, showmedians=True, showextrema=True)
        for body in parts['bodies']:
            body.set_facecolor('#E8F0F8')
            body.set_edgecolor('#0072B2')
            body.set_alpha(0.85)
            body.set_linewidth(0.9)
        for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
            if key in parts:
                parts[key].set_color('#0072B2')
                parts[key].set_linewidth(0.9)
        if 'cmedians' in parts:
            parts['cmedians'].set_color('#D55E00')
            parts['cmedians'].set_linewidth(1.6)
        ax.set_xticks(range(1, n_cats + 1))
        ax.set_xticklabels(cats)

    # Jitter scatter encima — mayor dispersión horizontal con boxplots más anchos
    rng = np.random.default_rng(0)
    jitter = 0.10
    for i, d in enumerate(data, start=1):
        if len(d):
            ax.scatter(rng.normal(i, jitter, len(d)), d, alpha=0.45, s=18,
                       color=CB_COLORS[0], edgecolor='white', linewidths=0.3, zorder=3)

    # N por categoría debajo del label
    ax.set_xticklabels([f"{c}\n(n={len(d)})" for c, d in zip(cats, data)])

    # Padding lateral para que la jitter no toque los bordes
    ax.set_xlim(0.4, n_cats + 0.6)

    ax.set_xlabel(box_by.replace('_', ' '))
    ax.set_ylabel(r'Accuracy media (\%)')
    ax.set_title(f'Distribución por {box_by}  (rombo = media, línea = mediana)')

    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    fig.tight_layout()

    render_fig(fig, f"box_{box_by}_{box_style.split()[0]}", key_prefix="box")

    # ── Tests pareados (Welch t-test, Bonferroni) ─────────────────────────────
    st.markdown("##### Tests pareados (Welch *t*-test, Bonferroni-corregido)")
    pairs = list(combinations(cats, 2))
    if not pairs:
        st.info("No hay pares para comparar (selecciona >1 categoría).")
    else:
        n_pairs = len(pairs)
        rows = []
        for a, b in pairs:
            xa = df_rf.loc[df_rf[box_by].astype(str) == a, 'accuracy_mean'].values
            xb = df_rf.loc[df_rf[box_by].astype(str) == b, 'accuracy_mean'].values
            if len(xa) < 2 or len(xb) < 2:
                continue
            t, p = stats.ttest_ind(xa, xb, equal_var=False)
            p_corr = min(p * n_pairs, 1.0)
            rows.append({
                'A': a, 'B': b,
                f'mean A': xa.mean(), f'mean B': xb.mean(),
                'Δ (A−B)': xa.mean() - xb.mean(),
                "Cohen's d": cohens_d(xa, xb),
                't': t, 'p (raw)': p, 'p (Bonf.)': p_corr,
                'sig.': '***' if p_corr < 0.001 else '**' if p_corr < 0.01 else '*' if p_corr < 0.05 else 'ns'
            })
        if rows:
            tbl = pd.DataFrame(rows)
            st.dataframe(
                tbl.style.format({
                    'mean A': '{:.2f}', 'mean B': '{:.2f}', 'Δ (A−B)': '{:+.2f}',
                    "Cohen's d": '{:+.2f}', 't': '{:+.2f}',
                    'p (raw)': '{:.4f}', 'p (Bonf.)': '{:.4f}'
                }),
                use_container_width=True, hide_index=True,
            )
            with st.expander("Exportar tabla a LaTeX"):
                st.code(tbl.to_latex(index=False, float_format='%.3f',
                                      caption=f'Pairwise comparisons by {box_by}',
                                      label=f'tab:pairwise_{box_by}'),
                        language='latex')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════

with tab_heat:
    st.subheader("Heatmap 2D de accuracy media")

    c1, c2, c3 = st.columns(3)
    x_ax = c1.selectbox("Eje X", ['chance_temperature', 'utility_temperature',
                                    'n_decision_rules_pct', 'fitness_type'],
                         key='heat_x')
    y_ax = c2.selectbox("Eje Y", ['utility_temperature', 'chance_temperature',
                                    'mode', 'sampling_mode', 'fitness_type'],
                         key='heat_y')
    facet = c3.selectbox("Faceta (subplots)", ['(ninguna)', 'sampling_mode', 'mode', 'model'],
                          key='heat_facet')

    metric = st.radio("Métrica",
                       ['accuracy_mean', 'mse_chance_mean', 'mse_utility_mean', 'stop_gen_mean'],
                       horizontal=True)

    # Validaciones de los ejes elegidos
    if x_ax == y_ax:
        st.warning("Eje X e Y son iguales; elige columnas distintas.")
        st.stop()
    if metric not in df_rf.columns:
        st.warning(f"La métrica `{metric}` no está disponible en los datos filtrados.")
        st.stop()

    series_metric = pd.to_numeric(df_rf[metric], errors='coerce')
    if series_metric.dropna().empty:
        st.warning(f"`{metric}` está vacía o no es numérica para los filtros actuales.")
        st.stop()

    if facet == '(ninguna)':
        facets = [None]
        sub_dfs = [df_rf]
    else:
        facets = sorted(df_rf[facet].dropna().astype(str).unique())
        sub_dfs = [df_rf[df_rf[facet].astype(str) == f] for f in facets]
        if not facets:
            st.warning(f"No hay valores no-nulos en `{facet}` para facetar.")
            st.stop()

    fig, axes = plt.subplots(1, len(facets), figsize=(fig_width * len(facets), fig_height),
                              squeeze=False, sharey=True)
    vmin, vmax = float(series_metric.min()), float(series_metric.max())
    if vmin == vmax:        # evita división por 0 al normalizar colores de anotación
        vmax = vmin + 1e-9

    im = None
    for ax, sub, fname in zip(axes[0], sub_dfs, facets):
        if sub.empty or sub[metric].dropna().empty:
            ax.set_title(f"{facet}={fname} (vacío)" if fname else "(vacío)")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        pv = sub.pivot_table(values=metric, index=y_ax, columns=x_ax, aggfunc='mean')
        im = ax.imshow(pv.values, aspect='auto', cmap=VIRIDIS, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pv.columns)))
        ax.set_xticklabels([str(c) for c in pv.columns], rotation=0)
        ax.set_yticks(range(len(pv.index)))
        ax.set_yticklabels([str(c) for c in pv.index])
        # Anotaciones
        for i in range(len(pv.index)):
            for j in range(len(pv.columns)):
                v = pv.values[i, j]
                if pd.notna(v):
                    color = 'white' if (v - vmin) / (vmax - vmin + 1e-9) < 0.5 else 'black'
                    ax.text(j, i, f"{v:.2f}", ha='center', va='center', color=color, fontsize=8)
        ax.set_xlabel(x_ax.replace('_', ' '))
        if ax is axes[0, 0]:
            ax.set_ylabel(y_ax.replace('_', ' '))
        if fname is not None:
            ax.set_title(f"{facet}={fname}")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.8, pad=0.02)
        cbar.set_label(metric.replace('_', ' '))
    fig.suptitle(f'{metric} por {y_ax} × {x_ax}', y=1.02)
    render_fig(fig, f"heatmap_{metric}_{x_ax}_{y_ax}_{facet}", key_prefix="heat")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CONVERGENCE CURVES
# ══════════════════════════════════════════════════════════════════════════════

with tab_curves:
    st.subheader("Curvas de convergencia")
    st.caption("Las curvas ya están agregadas sobre las repeticiones. El sombreado representa ±1 σ entre configuraciones del grupo (no entre repeticiones). Para CI por rep haría falta guardar histories per-rep.")

    if df_cf.empty:
        st.info("No hay curvas para los filtros seleccionados.")
    else:
        metric_map = {
            "Accuracy (%)":     "mean_accuracy",
            "Fitness":          "mean_fitness",
            "MSE Chance":       "mean_error_chance",
            "MSE Utility":      "mean_error_utility",
            "Entropía Norm.":   "mean_entropy_norm",
            "Util Dev":         "mean_util_dev",
        }
        avail = {k: v for k, v in metric_map.items() if v in df_cf.columns}
        if not avail:
            st.warning("No se encontraron métricas estándar en el CSV de curvas filtrado.")
            st.stop()
        c1, c2 = st.columns(2)
        sel_metric_label = c1.selectbox("Métrica", list(avail.keys()))
        sel_metric = avail[sel_metric_label]
        color_curve = c2.selectbox("Color por",
                                    ['model', 'mode', 'sampling_mode', 'fitness_type',
                                     'n_decision_rules_pct'],
                                    index=1)

        df_cf_v = df_cf.dropna(subset=[sel_metric]).copy()
        if df_cf_v.empty:
            st.warning(f"`{sel_metric}` está vacía para los filtros actuales.")
            st.stop()

        agg = df_cf_v.groupby([color_curve, 'generation'])[sel_metric].agg(['mean', 'std']).reset_index()
        cats = sorted(agg[color_curve].dropna().astype(str).unique())

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        for i, c in enumerate(cats):
            d = agg[agg[color_curve].astype(str) == c].sort_values('generation')
            if d.empty:
                continue
            x = d['generation'].values
            m = d['mean'].values
            s = d['std'].fillna(0).values
            color = CB_COLORS[i % len(CB_COLORS)]
            ax.plot(x, m, color=color, linewidth=1.4, label=str(c))
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)

        ax.set_xlabel("Generación")
        ax.set_ylabel(sel_metric_label)
        ax.set_title(f"Convergencia — {sel_metric_label}")
        if cats:
            # Leyenda a la derecha del plot, fuera del área de ejes.
            # 'best' a veces se sienta encima de las curvas; fuera no compite con datos.
            ax.legend(
                title=color_curve,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                borderaxespad=0.0,
            )
        fig.tight_layout()
        render_fig(fig, f"curve_{sel_metric}_{color_curve}", key_prefix="curve")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TABLES + LATEX
# ══════════════════════════════════════════════════════════════════════════════

with tab_tables:
    st.subheader("Mejor configuración por (modelo × modo)")

    # idxmax falla si hay grupos con todos NaN. Filtramos antes.
    df_for_idx = df_rf.dropna(subset=['accuracy_mean']).copy()
    if df_for_idx.empty:
        st.warning("No hay filas con accuracy_mean válido tras filtrar.")
        st.stop()
    idx = df_for_idx.groupby(['model', 'mode'])['accuracy_mean'].idxmax()
    cols_keep = ['model', 'mode', 'sampling_mode', 'fitness_type',
                 'chance_temperature', 'utility_temperature',
                 'n_decision_rules_pct', 'accuracy_mean', 'accuracy_std',
                 'accuracy_min', 'accuracy_max', 'ci95',
                 'mse_chance_mean', 'mse_utility_mean']
    cols_keep = [c for c in cols_keep if c in df_for_idx.columns]
    best = df_for_idx.loc[idx, cols_keep].sort_values('accuracy_mean', ascending=False)

    st.dataframe(
        best.style.format({
            'chance_temperature': '{:.1f}', 'utility_temperature': '{:.1f}',
            'accuracy_mean': '{:.2f}', 'accuracy_std': '{:.2f}',
            'accuracy_min': '{:.0f}', 'accuracy_max': '{:.0f}',
            'ci95': '{:.2f}',
            'mse_chance_mean': '{:.3f}', 'mse_utility_mean': '{:.3f}',
        }),
        use_container_width=True, hide_index=True,
    )

    with st.expander("Exportar tabla a LaTeX (booktabs)"):
        # Construye un acc ± CI legible
        latex_df = best.copy()
        latex_df['Accuracy (% ± CI95)'] = latex_df.apply(
            lambda r: f"{r['accuracy_mean']:.2f} $\\pm$ {r['ci95']:.2f}", axis=1)
        cols = ['model', 'mode', 'sampling_mode', 'fitness_type',
                'chance_temperature', 'utility_temperature',
                'n_decision_rules_pct', 'Accuracy (% ± CI95)',
                'mse_chance_mean', 'mse_utility_mean']
        st.code(latex_df[cols].to_latex(
            index=False,
            escape=False,
            float_format='%.3f',
            caption='Best configuration per (model, decomposition mode). Accuracy reported as mean $\\pm$ 95\\% confidence interval (Student-$t$, $N$ runs).',
            label='tab:best_configs'),
            language='latex')

    st.divider()
    st.subheader("Top 10 configuraciones absolutas")

    top = df_rf.nlargest(10, 'accuracy_mean')[[
        'model', 'mode', 'sampling_mode', 'fitness_type',
        'chance_temperature', 'utility_temperature',
        'n_decision_rules_pct', 'accuracy_mean', 'ci95', 'mse_utility_mean'
    ]]
    st.dataframe(
        top.style.format({
            'chance_temperature': '{:.1f}', 'utility_temperature': '{:.1f}',
            'accuracy_mean': '{:.2f}', 'ci95': '{:.2f}',
            'mse_utility_mean': '{:.3f}',
        }),
        use_container_width=True, hide_index=True,
    )
