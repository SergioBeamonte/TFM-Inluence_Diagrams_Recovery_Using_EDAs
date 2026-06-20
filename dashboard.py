import streamlit as st
import pandas as pd
import altair as alt
import glob
import os

st.set_page_config(
    page_title="EDA Grid Search Explorer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 EDA Grid Search — Comparación Multi-Modelo")
st.caption("Barrido paramétrico para recuperación de IDs. Carga automática de todos los modelos disponibles.")

# ─── DISCOVERY ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def discover_models():
    found = {}
    for f in glob.glob("**/grid_search_results_*.csv", recursive=True):
        # Ignorar worktrees de git y carpetas ocultas
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
    # CSVs del estudio min/max utilidad: formato crudo (1 fila/rep), se agregan
    # al vuelo al esquema de grid_search_results (ver _load_minmax).
    for f in glob.glob("**/explore_minmax_ut.csv", recursive=True):
        parts = f.replace("\\", "/").split("/")
        if any(p.startswith(".") for p in parts):
            continue
        subdir = os.path.basename(os.path.dirname(f))
        key    = f"{subdir} · minmax_ut"
        found[key] = (os.path.normpath(f), None)
    return found


_RESULTS_RENAME = {
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
_RESULTS_DROP = {
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


def _normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in _RESULTS_RENAME.items() if k in df.columns})
    df = df.drop(columns=[c for c in _RESULTS_DROP if c in df.columns])
    # Defaults retrocompatibles: filas antiguas sin las dimensiones nuevas se
    # interpretan como el comportamiento clásico para que sigan apareciendo en
    # los filtros sin estallar.
    if 'mode' not in df.columns:
        df['mode'] = 'both'
    if 'sampling_mode' not in df.columns:
        df['sampling_mode'] = 'non_symmetric'
    if 'chance_temperature' not in df.columns:
        df['chance_temperature'] = 1.0
    if 'utility_temperature' not in df.columns:
        df['utility_temperature'] = 1.0
    # size_gen (tamaño de población) lo introduce explore_size_gen.py; en el
    # resto de CSVs queda como NaN y se ignora en los filtros.
    if 'size_gen' not in df.columns:
        df['size_gen'] = pd.NA
    # min_max_ut solo lo trae el estudio explore_minmax_ut; en el resto, NaN.
    if 'min_max_ut' not in df.columns:
        df['min_max_ut'] = pd.NA
    return df


def _fill_new_param_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena columnas de los parámetros nuevos para CSVs anteriores al cambio."""
    if 'mode' not in df.columns:
        df['mode'] = 'both'
    if 'sampling_mode' not in df.columns:
        df['sampling_mode'] = 'non_symmetric'
    if 'chance_temperature' not in df.columns:
        df['chance_temperature'] = 1.0
    if 'utility_temperature' not in df.columns:
        df['utility_temperature'] = 1.0
    if 'size_gen' not in df.columns:
        df['size_gen'] = pd.NA
    if 'min_max_ut' not in df.columns:
        df['min_max_ut'] = pd.NA
    return df


# total_rules por red (no viene en el CSV crudo de min/max); ver explore_minmax_ut.py.
_MINMAX_TOTAL_RULES = {'bypass2': 20, 'nhlv1': 67}


def _load_minmax(path: str) -> pd.DataFrame:
    """Agrega un CSV crudo de explore_minmax_ut (1 fila por repetición) al esquema
    de grid_search_results, añadiendo `min_max_ut` como característica extra.

    Las dimensiones fijas del estudio (fitness binary, stop top50, muestreo no
    simétrico, T=1, 10% de reglas) se rellenan con sus valores de diseño para que
    encajen con los filtros del dashboard (ver COMMON en explore_minmax_ut.py)."""
    raw = pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame()
    cfg = ['net', 'optimizer', 'min_max_ut', 'size_gen', 'n_decision_rules', 'mode']
    g = raw.groupby(cfg, dropna=False)
    out = g.agg(
        accuracy_mean=('best_accuracy', 'mean'),
        accuracy_std=('best_accuracy', 'std'),
        accuracy_min=('best_accuracy', 'min'),
        accuracy_max=('best_accuracy', 'max'),
        mse_chance_mean=('mse_chance', 'mean'),
        mse_chance_std=('mse_chance', 'std'),
        mse_utility_mean=('mse_utility', 'mean'),
        mse_utility_std=('mse_utility', 'std'),
        stop_gen_mean=('stop_generation', 'mean'),
        stop_gen_std=('stop_generation', 'std'),
        stop_gen_min=('stop_generation', 'min'),
        stop_gen_max=('stop_generation', 'max'),
        best_fitness_mean=('best_fitness', 'mean'),
        best_fitness_std=('best_fitness', 'std'),
    ).reset_index()
    # Tiempos de CPU (solo en algunos CSVs): cpu_per_gen → gen_time, cpu_total → wall_time.
    if 'cpu_per_gen' in raw.columns:
        cpu = g.agg(
            gen_time_mean=('cpu_per_gen', 'mean'),
            gen_time_std=('cpu_per_gen', 'std'),
            wall_time_mean=('cpu_total', 'mean'),
            wall_time_std=('cpu_total', 'std'),
        ).reset_index()
        out = out.merge(cpu, on=cfg)
    out['fitness_type']        = 'binary'
    out['stop_mode']           = 'top50'
    out['sampling_mode']       = 'non_symmetric'
    out['chance_temperature']  = 1.0
    out['utility_temperature'] = 1.0
    out['n_decision_rules_pct'] = 10
    out['total_rules'] = out['net'].map(_MINMAX_TOTAL_RULES)
    out['model'] = out['net'] + ' · ' + out['optimizer'].astype(str).str.upper() + ' · mmu'
    return out


@st.cache_data
def load_all(model_dict_frozen):
    res_list, cur_list = [], []
    for model, (results_path, curves_path) in model_dict_frozen:
        if results_path and os.path.exists(results_path):
            if os.path.basename(results_path).startswith("explore_minmax_ut"):
                df = _load_minmax(results_path)
                if not df.empty:
                    res_list.append(_normalize_results(df))
            else:
                df = pd.read_csv(results_path)
                if not df.empty:
                    df.insert(0, "model", model)
                    res_list.append(_normalize_results(df))
        if curves_path and os.path.exists(curves_path):
            df = pd.read_csv(curves_path)
            if not df.empty:
                df.insert(0, "model", model)
                cur_list.append(_fill_new_param_defaults(df))
    df_r = pd.concat(res_list, ignore_index=True) if res_list else pd.DataFrame()
    df_c = pd.concat(cur_list, ignore_index=True) if cur_list else pd.DataFrame()
    return df_r, df_c


model_dict = discover_models()

if not model_dict:
    st.error("No se encontraron archivos `grid_search_results_MODELO.csv` en ningún subdirectorio.")
    st.stop()

# ─── SELECCIÓN DE CSVs ────────────────────────────────────────────────────────
# Evita cargar todos los CSVs al abrir: el usuario elige cuáles antes de cargar.

_available_keys = sorted(model_dict.keys())

if "loaded_models" not in st.session_state:
    st.session_state.loaded_models = []

# Limpia selecciones obsoletas (CSVs que ya no existen).
st.session_state.loaded_models = [
    k for k in st.session_state.loaded_models if k in model_dict
]

with st.sidebar:
    st.header("📂 Datos a Cargar")
    st.caption(f"{len(_available_keys)} CSVs detectados. Elige cuáles cargar.")

    _default_sel = st.session_state.loaded_models or []
    _picker = st.multiselect(
        "CSVs disponibles",
        _available_keys,
        default=_default_sel,
        key="csv_picker",
    )
    _c1, _c2 = st.columns(2)
    _load_btn = _c1.button("📥 Cargar", type="primary", width='stretch')
    _all_btn  = _c2.button("Todos",    width='stretch')

    if _load_btn:
        st.session_state.loaded_models = list(_picker)
        st.rerun()
    if _all_btn:
        st.session_state.loaded_models = list(_available_keys)
        st.rerun()

if not st.session_state.loaded_models:
    st.info(
        f"📂 **{len(_available_keys)} CSVs detectados.** "
        "Selecciona en la barra lateral los que quieras cargar y pulsa **Cargar** "
        "(o **Todos** para cargarlos todos)."
    )
    with st.expander("Ver lista de CSVs disponibles", expanded=True):
        for _k in _available_keys:
            _res, _cur = model_dict[_k]
            st.write(f"- **{_k}** — `{_res}`" + ("" if _cur else "  _(sin curvas)_"))
    st.stop()

_selected_dict = {k: model_dict[k] for k in st.session_state.loaded_models}
df_results_all, df_curves_all = load_all(tuple(sorted(_selected_dict.items())))

if df_results_all.empty and df_curves_all.empty:
    models_found = ", ".join(sorted(model_dict.keys()))
    st.info(
        f"Los archivos para **{models_found}** están vacíos — el grid search probablemente "
        "aún está en ejecución. Recarga la página cuando termine."
    )
    st.stop()

MODEL_COLORS   = alt.Scale(scheme="tableau10")
FITNESS_COLORS = alt.Scale(scheme="set2")


def _register_theme():
    def _theme():
        return {
            "config": {
                "background": "white",
                "axis": {
                    "domainColor": "#d1d5db",
                    "gridColor":   "#f3f4f6",
                    "gridWidth":   1,
                    "labelColor":  "#6b7280",
                    "labelFontSize": 11,
                    "tickColor":   "#d1d5db",
                    "tickSize":    4,
                    "titleColor":  "#374151",
                    "titleFontSize": 12,
                    "titleFontWeight": "normal",
                    "titlePadding": 8,
                },
                "legend": {
                    "labelColor":       "#6b7280",
                    "labelFontSize":    11,
                    "titleColor":       "#374151",
                    "titleFontSize":    12,
                    "titleFontWeight":  "normal",
                    "padding":          4,
                },
                "title": {
                    "color":       "#111827",
                    "fontSize":    13,
                    "fontWeight":  "600",
                    "anchor":      "start",
                    "offset":      8,
                },
                "view": {"strokeWidth": 0},
            }
        }
    alt.themes.register("tfm_pro", _theme)
    alt.themes.enable("tfm_pro")


_register_theme()

_CATEGORICAL_COLS = [
    "model", "mode", "sampling_mode", "fitness_type", "stop_mode",
    "n_decision_rules", "n_decision_rules_pct", "total_rules",
    "chance_temperature", "utility_temperature",
]
# size_gen solo aparece si algún CSV cargado lo trae con datos (explore_size_gen).
_HAS_SIZE_GEN = (
    not df_results_all.empty
    and "size_gen" in df_results_all.columns
    and df_results_all["size_gen"].notna().any()
)
if _HAS_SIZE_GEN:
    _CATEGORICAL_COLS.append("size_gen")
# min_max_ut solo aparece si se carga el estudio explore_minmax_ut.
_HAS_MINMAX = (
    not df_results_all.empty
    and "min_max_ut" in df_results_all.columns
    and df_results_all["min_max_ut"].notna().any()
)
if _HAS_MINMAX:
    _CATEGORICAL_COLS.append("min_max_ut")

_NUMERIC_COLS = (
    [c for c in df_results_all.columns
     if c not in _CATEGORICAL_COLS
        and c != "size_gen"
        and pd.api.types.is_numeric_dtype(df_results_all[c])]
    if not df_results_all.empty else []
)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Filtros Globales")
    st.caption("Se aplican a ambas pestañas.")

    ref = df_results_all if not df_results_all.empty else df_curves_all

    all_models  = sorted(ref["model"].unique())
    all_mode    = sorted(ref["mode"].unique())
    all_samp    = sorted(ref["sampling_mode"].unique())
    all_fitness = sorted(ref["fitness_type"].unique())
    all_stop    = sorted(ref["stop_mode"].unique())
    all_ndr     = (
        sorted(int(v) for v in ref["n_decision_rules"].dropna().unique())
        if "n_decision_rules" in ref.columns else []
    )
    all_pct     = sorted(ref["n_decision_rules_pct"].unique())
    all_tr      = (
        sorted(int(v) for v in ref["total_rules"].dropna().unique())
        if "total_rules" in ref.columns else []
    )
    all_ct      = sorted(ref["chance_temperature"].unique())
    all_ut      = sorted(ref["utility_temperature"].unique())
    all_sg      = (
        sorted(int(v) for v in ref["size_gen"].dropna().unique())
        if "size_gen" in ref.columns else []
    )
    all_mmu     = (
        sorted(bool(v) for v in ref["min_max_ut"].dropna().unique())
        if "min_max_ut" in ref.columns and ref["min_max_ut"].notna().any() else []
    )

    sel_models  = st.multiselect("Modelos",              all_models,  default=all_models)
    sel_mode    = st.multiselect("Modo (both/util/cpt)", all_mode,    default=all_mode)
    sel_samp    = st.multiselect("Muestreo (sym/non)",   all_samp,    default=all_samp)
    sel_fitness = st.multiselect("Tipo de Fitness",      all_fitness, default=all_fitness)
    sel_stop    = st.multiselect("Modo de Parada",       all_stop,    default=all_stop)
    sel_ndr     = st.multiselect("Nº Reglas de Decisión", all_ndr, default=all_ndr) if all_ndr else None
    sel_pct     = st.multiselect("% Reglas de Decisión", all_pct,     default=all_pct)
    sel_tr      = st.multiselect("Total de Reglas",      all_tr, default=all_tr) if all_tr else None
    sel_ct      = st.multiselect("T softmax (CPTs)",     all_ct,      default=all_ct)
    sel_ut      = st.multiselect("T sigmoid (Util.)",    all_ut,      default=all_ut)
    if all_sg:
        sel_sg = st.multiselect("Tamaño de Generación", all_sg, default=all_sg)
    else:
        sel_sg = None
    if all_mmu:
        sel_mmu = st.multiselect("Min/Max Utilidad fijado", all_mmu, default=all_mmu)
    else:
        sel_mmu = None

    st.divider()
    st.subheader("📊 Explorador de Variables")
    st.caption("Oculta características del explorador interactivo.")
    excluded_num = st.multiselect("Ocultar métricas numéricas",     _NUMERIC_COLS,     default=[])
    excluded_cat = st.multiselect("Ocultar características categ.", _CATEGORICAL_COLS, default=[])

avail_num = [c for c in _NUMERIC_COLS     if c not in excluded_num]
avail_cat = [c for c in _CATEGORICAL_COLS if c not in excluded_cat]


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (
        df["model"].isin(sel_models)
        & df["mode"].isin(sel_mode)
        & df["sampling_mode"].isin(sel_samp)
        & df["fitness_type"].isin(sel_fitness)
        & df["stop_mode"].isin(sel_stop)
        & df["n_decision_rules_pct"].isin(sel_pct)
        & df["chance_temperature"].isin(sel_ct)
        & df["utility_temperature"].isin(sel_ut)
    )
    # n_decision_rules / total_rules: presentes en resultados (total_rules no en
    # curvas). Filtra solo las filas que sí tienen la columna; el resto pasan.
    if sel_ndr is not None and "n_decision_rules" in df.columns:
        ndr_num = pd.to_numeric(df["n_decision_rules"], errors="coerce")
        mask &= ndr_num.isna() | ndr_num.isin(sel_ndr)
    if sel_tr is not None and "total_rules" in df.columns:
        tr_num = pd.to_numeric(df["total_rules"], errors="coerce")
        mask &= tr_num.isna() | tr_num.isin(sel_tr)
    # size_gen: filtra solo las filas que sí lo tienen; las que no, pasan.
    if sel_sg is not None and "size_gen" in df.columns:
        sg_num = pd.to_numeric(df["size_gen"], errors="coerce")
        mask &= sg_num.isna() | sg_num.isin(sel_sg)
    # min_max_ut: solo lo tiene el estudio minmax; el resto de filas pasan.
    if sel_mmu is not None and "min_max_ut" in df.columns:
        mmu = df["min_max_ut"]
        mask &= mmu.isna() | mmu.isin(sel_mmu)
    return df[mask].copy()


df_r = apply_filters(df_results_all)
df_c = apply_filters(df_curves_all)

# ─── TABS ─────────────────────────────────────────────────────────────────────

tab_results, tab_curves = st.tabs(["📋 Resultados por Modelo", "📈 Curvas de Convergencia"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

with tab_results:
    if df_r.empty:
        st.info("Sin datos de resultados para los filtros seleccionados.")
    else:
        # ── KPIs ──────────────────────────────────────────────────────────────
        best = df_r.loc[df_r["accuracy_mean"].idxmax()]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Configuraciones totales", len(df_r))
        _best_sg = best.get("size_gen") if "size_gen" in best.index else None
        _sg_part = f" · sg={int(_best_sg)}" if pd.notna(_best_sg) else ""
        k2.metric(
            "Mejor Accuracy (media)",
            f"{best['accuracy_mean']:.1f}%",
            delta=(
                f"{best['model']} · mode={best['mode']} · samp={best['sampling_mode']} · "
                f"{best['fitness_type']} · {best['stop_mode']} · {best['n_decision_rules_pct']}% · "
                f"Tc={best['chance_temperature']} · Tu={best['utility_temperature']}{_sg_part}"
            ),
        )
        k3.metric(
            "Accuracy Global Máximo",
            f"{df_r['accuracy_max'].max():.1f}%",
        )
        k4.metric(
            "Menor MSE Chance (media)",
            f"{df_r['mse_chance_mean'].min():.3f}",
        )
        k5.metric("Gen. Media de Parada", f"{df_r['stop_gen_mean'].mean():.1f}")

        st.divider()

        # ── EXPLORADOR INTERACTIVO ────────────────────────────────────────────
        st.subheader("🔬 Explorador de Variables")

        _sel_row = st.columns([2, 2, 1])
        sel_exp_num = _sel_row[0].multiselect(
            "Variables numéricas (eje X / Y)",
            avail_num,
            default=avail_num[:2] if len(avail_num) >= 2 else avail_num,
            key="exp_num",
        )
        sel_exp_cat = _sel_row[1].multiselect(
            "Variables categóricas (color / forma)",
            avail_cat,
            default=[],
            key="exp_cat",
        )

        n_num = len(sel_exp_num)
        n_cat = len(sel_exp_cat)
        total = n_num + n_cat

        _chart_type = "Scatter"
        if n_num == 2:
            _chart_type = _sel_row[2].radio(
                "Tipo",
                ["Scatter", "Boxplot"],
                key="chart_type_toggle",
            )

        def _et(col):
            return "N" if col in _CATEGORICAL_COLS else "Q"

        if total < 2:
            st.info("Selecciona al menos 2 características.")
        elif n_num > 2:
            st.warning("Máximo 2 variables numéricas.")
        elif n_cat > 2:
            st.warning("Máximo 2 variables categóricas.")
        elif n_num == 0:
            st.warning("Selecciona al menos 1 variable numérica.")
        else:
            _exp_chart = None
            all_sel = sel_exp_num + sel_exp_cat
            _tooltip = [alt.Tooltip(f"{c}:{_et(c)}", title=c) for c in all_sel]
            _base = alt.Chart(df_r)

            def _bar_y_scale(agg, col, cats):
                """Slider para acercar el eje Y de las barras (no arrancar en 0).
                Por defecto se ajusta al rango de los datos para que diferencias
                pequeñas (p.ej. 98 vs 100) se aprecien."""
                vals = pd.to_numeric(agg[col], errors="coerce").dropna()
                if vals.empty:
                    return alt.Scale(zero=False)
                dmin, dmax = float(vals.min()), float(vals.max())
                span = dmax - dmin
                pad = span * 0.1 if span > 0 else (abs(dmax) * 0.05 or 1.0)
                lo_bound = round(min(0.0, dmin - pad), 4)
                hi_bound = round(dmax + pad, 4)
                default_lo = round(max(lo_bound, dmin - pad), 4)
                if lo_bound >= hi_bound:  # datos constantes degenerados
                    return alt.Scale(zero=False)
                lo, hi = st.slider(
                    f"Rango eje Y — {col}",
                    min_value=float(lo_bound), max_value=float(hi_bound),
                    value=(float(default_lo), float(hi_bound)),
                    key=f"yzoom_{col}_{'_'.join(cats)}_{lo_bound}_{hi_bound}",
                )
                return alt.Scale(domain=[lo, hi])

            # ── BOXPLOT ──────────────────────────────────────────────────────
            if _chart_type == "Boxplot":
                if n_cat != 1:
                    st.warning("El boxplot requiere exactamente 1 variable categórica para agrupar.")
                else:
                    # Altair hconcat ignora use_container_width: fijamos ancho explícito
                    # por sub-gráfico, escalando con el nº de categorías para que las
                    # cajas tengan aire.
                    _n_cats = df_r[sel_exp_cat[0]].nunique()
                    _sub_w = max(450, _n_cats * 75)

                    def _boxplot(y_col):
                        return (
                            alt.Chart(df_r)
                            .mark_boxplot(extent="min-max", size=42)
                            .encode(
                                x=alt.X(f"{sel_exp_cat[0]}:N", title=sel_exp_cat[0],
                                        sort=alt.EncodingSortField(f"{y_col}", op="median", order="descending"),
                                        axis=alt.Axis(labelAngle=-25, labelLimit=180)),
                                y=alt.Y(f"{y_col}:Q", title=y_col, scale=alt.Scale(zero=False)),
                                color=alt.Color(f"{sel_exp_cat[0]}:N", scale=MODEL_COLORS, legend=None),
                                tooltip=[
                                    alt.Tooltip(f"{sel_exp_cat[0]}:N", title=sel_exp_cat[0]),
                                    alt.Tooltip(f"{y_col}:Q", title=y_col, format=".4f"),
                                ],
                            )
                            .properties(width=_sub_w, height=440, title=y_col)
                        )
                    _exp_chart = alt.hconcat(
                        _boxplot(sel_exp_num[0]),
                        _boxplot(sel_exp_num[1]),
                        spacing=40,
                    )

            # ── SCATTER / BARRAS ─────────────────────────────────────────────
            elif n_num == 2 and n_cat == 0:
                _exp_chart = (
                    _base.mark_point(size=80, opacity=0.75, filled=True)
                    .encode(
                        x=alt.X(f"{sel_exp_num[0]}:Q", title=sel_exp_num[0], scale=alt.Scale(zero=False)),
                        y=alt.Y(f"{sel_exp_num[1]}:Q", title=sel_exp_num[1], scale=alt.Scale(zero=False)),
                        tooltip=_tooltip,
                    )
                    .properties(height=440)
                    .interactive()
                )

            elif n_num == 2 and n_cat == 1:
                _exp_chart = (
                    _base.mark_point(size=80, opacity=0.75, filled=True)
                    .encode(
                        x=alt.X(f"{sel_exp_num[0]}:Q", title=sel_exp_num[0], scale=alt.Scale(zero=False)),
                        y=alt.Y(f"{sel_exp_num[1]}:Q", title=sel_exp_num[1], scale=alt.Scale(zero=False)),
                        color=alt.Color(f"{sel_exp_cat[0]}:N", title=sel_exp_cat[0], scale=MODEL_COLORS),
                        tooltip=_tooltip,
                    )
                    .properties(height=440)
                    .interactive()
                )

            elif n_num == 2 and n_cat == 2:
                _exp_chart = (
                    _base.mark_point(size=80, opacity=0.75, filled=True)
                    .encode(
                        x=alt.X(f"{sel_exp_num[0]}:Q", title=sel_exp_num[0], scale=alt.Scale(zero=False)),
                        y=alt.Y(f"{sel_exp_num[1]}:Q", title=sel_exp_num[1], scale=alt.Scale(zero=False)),
                        color=alt.Color(f"{sel_exp_cat[0]}:N", title=sel_exp_cat[0], scale=MODEL_COLORS),
                        shape=alt.Shape(f"{sel_exp_cat[1]}:N", title=sel_exp_cat[1]),
                        tooltip=_tooltip,
                    )
                    .properties(height=440)
                    .interactive()
                )

            elif n_num == 1 and n_cat == 1:
                _agg = df_r.groupby(sel_exp_cat[0], as_index=False)[sel_exp_num[0]].mean()
                _yscale = _bar_y_scale(_agg, sel_exp_num[0], sel_exp_cat)
                _exp_chart = (
                    alt.Chart(_agg)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X(f"{sel_exp_cat[0]}:N", title=sel_exp_cat[0], sort="-y"),
                        y=alt.Y(f"{sel_exp_num[0]}:Q", title=f"Media — {sel_exp_num[0]}",
                                scale=_yscale),
                        color=alt.Color(f"{sel_exp_cat[0]}:N", title=sel_exp_cat[0],
                                        scale=MODEL_COLORS, legend=None),
                        tooltip=[
                            alt.Tooltip(f"{sel_exp_cat[0]}:N", title=sel_exp_cat[0]),
                            alt.Tooltip(f"{sel_exp_num[0]}:Q", title=f"Media {sel_exp_num[0]}", format=".4f"),
                        ],
                    )
                    .properties(height=440)
                )

            elif n_num == 1 and n_cat == 2:
                _agg = df_r.groupby(sel_exp_cat, as_index=False)[sel_exp_num[0]].mean()
                _yscale = _bar_y_scale(_agg, sel_exp_num[0], sel_exp_cat)
                _exp_chart = (
                    alt.Chart(_agg)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X(f"{sel_exp_cat[0]}:N", title=sel_exp_cat[0], sort="-y"),
                        y=alt.Y(f"{sel_exp_num[0]}:Q", title=f"Media — {sel_exp_num[0]}",
                                scale=_yscale),
                        color=alt.Color(f"{sel_exp_cat[1]}:N", title=sel_exp_cat[1], scale=MODEL_COLORS),
                        xOffset=f"{sel_exp_cat[1]}:N",
                        tooltip=[
                            alt.Tooltip(f"{sel_exp_cat[0]}:N", title=sel_exp_cat[0]),
                            alt.Tooltip(f"{sel_exp_cat[1]}:N", title=sel_exp_cat[1]),
                            alt.Tooltip(f"{sel_exp_num[0]}:Q", title=f"Media {sel_exp_num[0]}", format=".4f"),
                        ],
                    )
                    .properties(height=440)
                )

            else:
                st.warning("Combinación no soportada.")

            if _exp_chart is not None:
                st.altair_chart(_exp_chart, width='stretch')

        # ── TABLA COMPLETA ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("📋 Tabla Completa — Todos los Modelos y Configuraciones")

        ordered_cols = [
            "model", "mode", "sampling_mode", "fitness_type", "stop_mode", "n_decision_rules_pct",
            "min_max_ut", "chance_temperature", "utility_temperature", "size_gen",
            "accuracy_mean", "accuracy_std", "accuracy_min", "accuracy_max",
            "mse_chance_mean", "mse_chance_std",
            "mse_utility_mean", "mse_utility_std",
            "entropy_norm_mean", "entropy_norm_std",
            "util_dev_mean", "util_dev_std",
            "stop_gen_mean", "stop_gen_std", "stop_gen_min", "stop_gen_max",
            "gen_time_mean", "gen_time_std", "wall_time_mean", "wall_time_std",
            "best_fitness_mean", "best_fitness_std",
            "total_rules", "n_decision_rules",
        ]
        show_cols = [c for c in ordered_cols if c in df_r.columns]
        table = df_r[show_cols].sort_values(
            ["accuracy_mean", "model"], ascending=[False, True]
        ).copy()
        if "min_max_ut" in table.columns:
            table["min_max_ut"] = (
                table["min_max_ut"].map({True: "Sí", False: "No"}).fillna("—")
            )

        st.dataframe(
            table,
            width='stretch',
            hide_index=True,
            column_config={
                "model":               st.column_config.TextColumn("Modelo"),
                "mode":                st.column_config.TextColumn("Modo"),
                "sampling_mode":       st.column_config.TextColumn("Muestreo"),
                "fitness_type":        st.column_config.TextColumn("Fitness"),
                "stop_mode":           st.column_config.TextColumn("Stop Mode"),
                "n_decision_rules_pct":st.column_config.NumberColumn("% Reglas",       format="%.0f %%"),
                "min_max_ut":          st.column_config.TextColumn("Min/Max Ut"),
                "chance_temperature":  st.column_config.NumberColumn("T softmax",      format="%.2f"),
                "utility_temperature": st.column_config.NumberColumn("T sigmoid",      format="%.2f"),
                "size_gen":            st.column_config.NumberColumn("Tamaño Gen.",    format="%.0f"),
                "accuracy_mean":       st.column_config.NumberColumn("Accuracy Media",  format="%.1f %%"),
                "accuracy_std":        st.column_config.NumberColumn("Accuracy σ",      format="%.2f"),
                "accuracy_min":        st.column_config.NumberColumn("Accuracy Min",    format="%.1f %%"),
                "accuracy_max":        st.column_config.NumberColumn("Accuracy Max",    format="%.1f %%"),
                "mse_chance_mean":     st.column_config.NumberColumn("MSE Chance",      format="%.3f"),
                "mse_chance_std":      st.column_config.NumberColumn("MSE Chance σ",    format="%.3f"),
                "mse_utility_mean":    st.column_config.NumberColumn("MSE Utility",     format="%.3f"),
                "mse_utility_std":     st.column_config.NumberColumn("MSE Utility σ",   format="%.3f"),
                "entropy_norm_mean":   st.column_config.NumberColumn("Entropía Norm.",  format="%.4f"),
                "entropy_norm_std":    st.column_config.NumberColumn("Entropía σ",      format="%.4f"),
                "util_dev_mean":       st.column_config.NumberColumn("Util Dev",        format="%.4f"),
                "util_dev_std":        st.column_config.NumberColumn("Util Dev σ",      format="%.4f"),
                "stop_gen_mean":       st.column_config.NumberColumn("Gen. Media",      format="%.1f"),
                "stop_gen_std":        st.column_config.NumberColumn("Gen. σ",          format="%.2f"),
                "stop_gen_min":        st.column_config.NumberColumn("Gen. Min",        format="%.0f"),
                "stop_gen_max":        st.column_config.NumberColumn("Gen. Max",        format="%.0f"),
                "gen_time_mean":       st.column_config.NumberColumn("CPU Time/Gen (s)", format="%.3f"),
                "gen_time_std":        st.column_config.NumberColumn("CPU Time/Gen σ",  format="%.3f"),
                "wall_time_mean":      st.column_config.NumberColumn("CPU Time Total (s)", format="%.2f"),
                "wall_time_std":       st.column_config.NumberColumn("CPU Time Total σ", format="%.2f"),
                "best_fitness_mean":   st.column_config.NumberColumn("Best Fitness",    format="%.4f"),
                "best_fitness_std":    st.column_config.NumberColumn("Best Fitness σ",  format="%.4f"),
                "total_rules":         st.column_config.NumberColumn("Total Reglas"),
                "n_decision_rules":    st.column_config.NumberColumn("N Reglas"),
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CURVES
# ══════════════════════════════════════════════════════════════════════════════

CURVE_METRICS = {
    "Accuracy (%)":    "mean_accuracy",
    "Fitness":         "mean_fitness",
    "MSE Chance":      "mean_error_chance",
    "MSE Utility":     "mean_error_utility",
    "Entropía Norm.":  "mean_entropy_norm",
    "Util Dev":        "mean_util_dev",
    "CPU Time/Gen (s)":"mean_gen_time",
}

with tab_curves:
    if df_c.empty:
        st.info("Sin datos de curvas para los filtros seleccionados — puede que el grid search aún esté corriendo.")
    else:
        df_c["config"] = (
            df_c["mode"] + " | "
            + df_c["sampling_mode"] + " | "
            + df_c["fitness_type"] + " | "
            + df_c["stop_mode"] + " | "
            + df_c["n_decision_rules_pct"].astype(str) + "% | "
            + "Tc=" + df_c["chance_temperature"].astype(str)
            + " Tu=" + df_c["utility_temperature"].astype(str)
        )
        df_c["model_config"] = df_c["model"] + " · " + df_c["config"]

        # ── CONTROLES ─────────────────────────────────────────────────────────
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)

        _avail_metrics = {lbl: m for lbl, m in CURVE_METRICS.items() if m in df_c.columns}
        sel_metric_label = ctrl1.selectbox("Métrica principal", list(_avail_metrics.keys()))
        sel_metric = _avail_metrics[sel_metric_label]

        max_gen  = int(df_c["generation"].max())
        gen_range = ctrl2.slider("Rango de generaciones", 1, max_gen, (1, max_gen))

        _COLOR_OPTIONS = {
            "Modelo":           "model",
            "Modo":             "mode",
            "Muestreo":         "sampling_mode",
            "Fitness Type":     "fitness_type",
            "Stop Mode":        "stop_mode",
            "% Reglas":         "n_decision_rules_pct",
            "T softmax":        "chance_temperature",
            "T sigmoid":        "utility_temperature",
        }
        sel_color_label = ctrl3.selectbox("Color por", list(_COLOR_OPTIONS.keys()))
        sel_color_col   = _COLOR_OPTIONS[sel_color_label]
        _cenc = "O" if sel_color_col in ("n_decision_rules_pct", "chance_temperature", "utility_temperature") else "N"

        curve_fitness = ctrl4.multiselect(
            "Fitness (filtro adicional curvas)",
            sorted(df_c["fitness_type"].unique()),
            default=sorted(df_c["fitness_type"].unique()),
        )

        df_cf = df_c[
            (df_c["generation"] >= gen_range[0])
            & (df_c["generation"] <= gen_range[1])
            & (df_c["fitness_type"].isin(curve_fitness))
        ].copy()

        if df_cf.empty:
            st.warning("No hay curvas para esta selección.")
            st.stop()

        # ── SUB-TABS ──────────────────────────────────────────────────────────
        ctab1, ctab2, ctab3 = st.tabs([
            "🌐 Todas las Curvas",
            "📉 Promedios y Factores",
            "🔬 Drilldown por Config",
        ])

        # ── CTAB 1: Todas las curvas ───────────────────────────────────────────
        with ctab1:
            st.subheader(f"{sel_metric_label} — todas las configuraciones")
            st.caption(f"Color por **{sel_color_label}**. Clic en la leyenda para resaltar.")

            sel = alt.selection_point(fields=[sel_color_col], bind="legend")

            base = alt.Chart(df_cf).encode(
                x=alt.X("generation:Q", title="Generación"),
                y=alt.Y(f"{sel_metric}:Q", title=sel_metric_label,
                        scale=alt.Scale(zero=False)),
                color=alt.Color(f"{sel_color_col}:{_cenc}", title=sel_color_label,
                                scale=MODEL_COLORS),
                detail="model_config:N",
                opacity=alt.condition(sel, alt.value(0.8), alt.value(0.06)),
                tooltip=[
                    "model:N", "fitness_type:N", "stop_mode:N",
                    alt.Tooltip("n_decision_rules_pct:O", title="% Reglas"),
                    alt.Tooltip(f"{sel_metric}:Q", format=".4f"),
                    "generation:Q",
                ],
            )

            all_curves = (
                base.mark_line(strokeWidth=1.2)
                + base.mark_circle(size=1).add_params(sel)
            ).properties(height=420).interactive()

            st.altair_chart(all_curves, width='stretch')

            # Mini-charts de las 6 métricas en paralelo (2 filas × 3)
            st.subheader(f"Curva promedio por {sel_color_label} — todas las métricas")
            # Forward-fill cada curva individual hasta la gen. máxima antes de promediar,
            # para que las curvas cortas (convergencia temprana) no arrastren la media a 0.
            _all_gens = sorted(df_cf["generation"].unique())
            _metric_cols = [c for c in CURVE_METRICS.values() if c in df_cf.columns]
            _cat_cols_present = [c for c in _CATEGORICAL_COLS if c in df_cf.columns]
            _fill_pieces = []
            for _cfg, _grp in df_cf.groupby("model_config"):
                _cat_vals = _grp[_cat_cols_present].iloc[0].to_dict()
                _sub = (
                    _grp.set_index("generation")[_metric_cols]
                    .reindex(_all_gens)
                    .ffill()
                    .reset_index()
                )
                for _k, _v in _cat_vals.items():
                    _sub[_k] = _v
                _fill_pieces.append(_sub)
            _df_mini = pd.concat(_fill_pieces, ignore_index=True) if _fill_pieces else df_cf

            _metrics_items = [(lbl, m) for lbl, m in CURVE_METRICS.items() if m in df_cf.columns]
            for _row_items in [_metrics_items[:3], _metrics_items[3:]]:
                if not _row_items:
                    continue
                _row_cols = st.columns(len(_row_items))
                for col, (lbl, m) in zip(_row_cols, _row_items):
                    avg = _df_mini.groupby([sel_color_col, "generation"], as_index=False)[m].mean()
                    mini = (
                        alt.Chart(avg)
                        .mark_line(strokeWidth=2.5)
                        .encode(
                            x=alt.X("generation:Q", title="Gen."),
                            y=alt.Y(f"{m}:Q", title=lbl, scale=alt.Scale(zero=False)),
                            color=alt.Color(f"{sel_color_col}:{_cenc}", legend=None,
                                            scale=MODEL_COLORS),
                            tooltip=[f"{sel_color_col}:{_cenc}", "generation:Q",
                                     alt.Tooltip(f"{m}:Q", format=".4f")],
                        )
                        .properties(height=210, title=lbl)
                        .interactive()
                    )
                    col.altair_chart(mini, width='stretch')

        # ── CTAB 2: Promedios por factor ───────────────────────────────────────
        with ctab2:
            st.subheader(f"Promedio de {sel_metric_label} por {sel_color_label}")

            avg_model = df_cf.groupby([sel_color_col, "generation"], as_index=False)[sel_metric].mean()
            ch_model = (
                alt.Chart(avg_model)
                .mark_line(strokeWidth=3)
                .encode(
                    x=alt.X("generation:Q", title="Generación"),
                    y=alt.Y(f"{sel_metric}:Q", title=sel_metric_label,
                            scale=alt.Scale(zero=False)),
                    color=alt.Color(f"{sel_color_col}:{_cenc}", title=sel_color_label,
                                    scale=MODEL_COLORS),
                    tooltip=[f"{sel_color_col}:{_cenc}", "generation:Q",
                             alt.Tooltip(f"{sel_metric}:Q", format=".4f")],
                )
                .properties(height=320)
                .interactive()
            )
            st.altair_chart(ch_model, width='stretch')

            f1, f2 = st.columns(2)

            with f1:
                st.subheader("Por Stop Mode")
                avg_stop = df_cf.groupby(["stop_mode", "generation"], as_index=False)[sel_metric].mean()
                ch_stop = (
                    alt.Chart(avg_stop)
                    .mark_line(strokeWidth=2)
                    .encode(
                        x=alt.X("generation:Q", title="Generación"),
                        y=alt.Y(f"{sel_metric}:Q", title=sel_metric_label,
                                scale=alt.Scale(zero=False)),
                        color=alt.Color("stop_mode:N", title="Stop Mode",
                                        scale=alt.Scale(scheme="set2")),
                        tooltip=["stop_mode:N", "generation:Q",
                                 alt.Tooltip(f"{sel_metric}:Q", format=".4f")],
                    )
                    .properties(height=270)
                    .interactive()
                )
                st.altair_chart(ch_stop, width='stretch')

            with f2:
                st.subheader("Por % Reglas")
                avg_pct = df_cf.groupby(["n_decision_rules_pct", "generation"], as_index=False)[sel_metric].mean()
                avg_pct["pct_label"] = avg_pct["n_decision_rules_pct"].astype(str) + "%"
                ch_pct = (
                    alt.Chart(avg_pct)
                    .mark_line(strokeWidth=2)
                    .encode(
                        x=alt.X("generation:Q", title="Generación"),
                        y=alt.Y(f"{sel_metric}:Q", title=sel_metric_label,
                                scale=alt.Scale(zero=False)),
                        color=alt.Color("pct_label:N", title="% Reglas",
                                        sort=[f"{p}%" for p in sorted(all_pct)],
                                        scale=alt.Scale(scheme="oranges")),
                        tooltip=["pct_label:N", "generation:Q",
                                 alt.Tooltip(f"{sel_metric}:Q", format=".4f")],
                    )
                    .properties(height=270)
                    .interactive()
                )
                st.altair_chart(ch_pct, width='stretch')

            st.subheader("Por Fitness Type")
            avg_ft = df_cf.groupby(["fitness_type", "generation"], as_index=False)[sel_metric].mean()
            ch_ft = (
                alt.Chart(avg_ft)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("generation:Q", title="Generación"),
                    y=alt.Y(f"{sel_metric}:Q", title=sel_metric_label,
                            scale=alt.Scale(zero=False)),
                    color=alt.Color("fitness_type:N", title="Fitness Type",
                                    scale=FITNESS_COLORS),
                    tooltip=["fitness_type:N", "generation:Q",
                             alt.Tooltip(f"{sel_metric}:Q", format=".4f")],
                )
                .properties(height=280)
                .interactive()
            )
            st.altair_chart(ch_ft, width='stretch')

        # ── CTAB 3: Drilldown ──────────────────────────────────────────────────
        with ctab3:
            st.subheader("Fija una configuración y compara modelos")
            st.caption("Selecciona fitness type, stop mode y % reglas para ver cómo evoluciona cada modelo.")

            d1, d2, d3 = st.columns(3)

            avail_ft   = sorted(df_cf["fitness_type"].unique())
            d_ft       = d1.selectbox("Fitness Type", avail_ft)

            avail_stop = sorted(df_cf[df_cf["fitness_type"] == d_ft]["stop_mode"].unique())
            d_stop     = d2.selectbox("Stop Mode", avail_stop)

            avail_pct  = sorted(
                df_cf[(df_cf["fitness_type"] == d_ft) & (df_cf["stop_mode"] == d_stop)]
                ["n_decision_rules_pct"].unique()
            )
            d_pct = d3.selectbox("% Reglas", avail_pct)

            drill = df_cf[
                (df_cf["fitness_type"] == d_ft)
                & (df_cf["stop_mode"] == d_stop)
                & (df_cf["n_decision_rules_pct"] == d_pct)
            ].copy()

            if drill.empty:
                st.info("No hay datos para esta combinación.")
            else:
                models_present = sorted(drill["model"].unique())
                st.caption(
                    f"**{d_ft}** · **{d_stop}** · **{d_pct}%** reglas — "
                    f"Modelos: {', '.join(models_present)}"
                )

                def drill_chart(y_field, y_title):
                    return (
                        alt.Chart(drill)
                        .mark_line(strokeWidth=2.5)
                        .encode(
                            x=alt.X("generation:Q", title="Generación"),
                            y=alt.Y(f"{y_field}:Q", title=y_title,
                                    scale=alt.Scale(zero=False)),
                            color=alt.Color("model:N", title="Modelo", scale=MODEL_COLORS),
                            tooltip=["model:N", "generation:Q",
                                     alt.Tooltip(f"{y_field}:Q", format=".4f")],
                        )
                        .properties(height=240)
                        .interactive()
                    )

                # Grid de métricas: siempre 2 columnas, filas según disponibilidad
                row1_c1, row1_c2 = st.columns(2)
                row2_c1, row2_c2 = st.columns(2)

                row1_c1.subheader("Accuracy (%)")
                row1_c1.altair_chart(drill_chart("mean_accuracy",      "Accuracy (%)"),    width='stretch')
                row1_c2.subheader("Fitness")
                row1_c2.altair_chart(drill_chart("mean_fitness",       "Fitness"),         width='stretch')
                row2_c1.subheader("MSE Chance")
                row2_c1.altair_chart(drill_chart("mean_error_chance",  "MSE Chance"),      width='stretch')
                row2_c2.subheader("MSE Utility")
                row2_c2.altair_chart(drill_chart("mean_error_utility", "MSE Utility"),     width='stretch')

                _has_ent = "mean_entropy_norm" in drill.columns
                _has_dev = "mean_util_dev" in drill.columns
                if _has_ent or _has_dev:
                    row3_c1, row3_c2 = st.columns(2)
                    if _has_ent:
                        row3_c1.subheader("Entropía Norm.")
                        row3_c1.altair_chart(drill_chart("mean_entropy_norm", "Entropía Norm."), width='stretch')
                    if _has_dev:
                        row3_c2.subheader("Util Dev")
                        row3_c2.altair_chart(drill_chart("mean_util_dev",     "Util Dev"),        width='stretch')

                # Velocidad de convergencia
                st.subheader("Velocidad de convergencia — gen. donde se alcanza el 98% del accuracy final")

                conv_rows = []
                for model in models_present:
                    md = drill[drill["model"] == model].sort_values("generation")
                    if len(md) < 3:
                        continue
                    final_acc = md["mean_accuracy"].iloc[-1]
                    hit = md[md["mean_accuracy"] >= final_acc * 0.98]
                    row = {
                        "Modelo":             model,
                        "Gen. Conv. (98%)":   int(hit["generation"].min()) if not hit.empty else None,
                        "Accuracy Final":     round(final_acc, 2),
                        "Fitness Final":      round(md["mean_fitness"].iloc[-1], 4),
                        "MSE Chance Final":   round(md["mean_error_chance"].iloc[-1], 4),
                        "MSE Utility Final":  round(md["mean_error_utility"].iloc[-1], 4),
                    }
                    if "mean_entropy_norm" in md.columns:
                        row["Entropía Norm. Final"] = round(md["mean_entropy_norm"].iloc[-1], 4)
                    if "mean_util_dev" in md.columns:
                        row["Util Dev Final"] = round(md["mean_util_dev"].iloc[-1], 4)
                    conv_rows.append(row)

                if conv_rows:
                    conv_df = pd.DataFrame(conv_rows)
                    st.dataframe(conv_df, width='stretch', hide_index=True)

                    conv_bar = (
                        alt.Chart(conv_df.dropna(subset=["Gen. Conv. (98%)"]))
                        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                        .encode(
                            x=alt.X("Modelo:N"),
                            y=alt.Y("Gen. Conv. (98%):Q", title="Generación de Convergencia"),
                            color=alt.Color("Modelo:N", scale=MODEL_COLORS),
                            tooltip=["Modelo:N", "Gen. Conv. (98%):Q",
                                     alt.Tooltip("Accuracy Final:Q", format=".2f")],
                        )
                        .properties(height=230)
                    )
                    st.altair_chart(conv_bar, width='stretch')