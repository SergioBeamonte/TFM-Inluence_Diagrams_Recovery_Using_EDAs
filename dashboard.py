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

@st.cache_data
def discover_models():
    found = {}
    for f in glob.glob("**/grid_search_results_*.csv", recursive=True):
        model = os.path.basename(f).replace("grid_search_results_", "").replace(".csv", "")
        curves = f.replace("results", "curves")
        found[model] = (
            os.path.normpath(f),
            os.path.normpath(curves) if os.path.exists(curves) else None,
        )
    return found


@st.cache_data
def load_all(model_dict_frozen):
    res_list, cur_list = [], []
    for model, (results_path, curves_path) in model_dict_frozen:
        if results_path and os.path.exists(results_path):
            df = pd.read_csv(results_path)
            if not df.empty:
                df.insert(0, "model", model)
                res_list.append(df)
        if curves_path and os.path.exists(curves_path):
            df = pd.read_csv(curves_path)
            if not df.empty:
                df.insert(0, "model", model)
                cur_list.append(df)
    df_r = pd.concat(res_list, ignore_index=True) if res_list else pd.DataFrame()
    df_c = pd.concat(cur_list, ignore_index=True) if cur_list else pd.DataFrame()
    return df_r, df_c


model_dict = discover_models()

if not model_dict:
    st.error("No se encontraron archivos `grid_search_results_MODELO.csv` en ningún subdirectorio.")
    st.stop()

df_results_all, df_curves_all = load_all(tuple(sorted(model_dict.items())))

if df_results_all.empty and df_curves_all.empty:
    models_found = ", ".join(sorted(model_dict.keys()))
    st.info(
        f"Los archivos para **{models_found}** están vacíos — el grid search probablemente "
        "aún está en ejecución. Recarga la página cuando termine."
    )
    st.stop()

MODEL_COLORS   = alt.Scale(scheme="tableau10")
FITNESS_COLORS = alt.Scale(scheme="set2")

_CATEGORICAL_COLS = ["model", "fitness_type", "stop_mode", "n_decision_rules_pct"]
_NUMERIC_COLS = (
    [c for c in df_results_all.columns
     if c not in _CATEGORICAL_COLS and pd.api.types.is_numeric_dtype(df_results_all[c])]
    if not df_results_all.empty else []
)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Filtros Globales")
    st.caption("Se aplican a ambas pestañas.")

    ref = df_results_all if not df_results_all.empty else df_curves_all

    all_models  = sorted(ref["model"].unique())
    all_fitness = sorted(ref["fitness_type"].unique())
    all_stop    = sorted(ref["stop_mode"].unique())
    all_pct     = sorted(ref["n_decision_rules_pct"].unique())

    sel_models  = st.multiselect("Modelos",              all_models,  default=all_models)
    sel_fitness = st.multiselect("Tipo de Fitness",      all_fitness, default=all_fitness)
    sel_stop    = st.multiselect("Modo de Parada",       all_stop,    default=all_stop)
    sel_pct     = st.multiselect("% Reglas de Decisión", all_pct,     default=all_pct)

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
    return df[
        df["model"].isin(sel_models)
        & df["fitness_type"].isin(sel_fitness)
        & df["stop_mode"].isin(sel_stop)
        & df["n_decision_rules_pct"].isin(sel_pct)
    ].copy()


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
        best = df_r.loc[df_r["best_accuracy_mean"].idxmax()]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Configuraciones totales", len(df_r))
        k2.metric(
            "Mejor Accuracy (best_mean)",
            f"{best['best_accuracy_mean']:.1f}%",
            delta=f"{best['model']} · {best['fitness_type']} · {best['stop_mode']} · {best['n_decision_rules_pct']}%",
        )
        k3.metric(
            "Accuracy Global Máximo",
            f"{df_r['best_accuracy_max'].max():.1f}%",
        )
        k4.metric(
            "Menor MSE Chance (media)",
            f"{df_r['mean_mse_chance_mean'].min():.3f}",
        )
        k5.metric("Gen. Media de Parada", f"{df_r['stop_gen_mean'].mean():.1f}")

        st.divider()

        # ── EXPLORADOR INTERACTIVO ────────────────────────────────────────────
        st.subheader("🔬 Explorador de Variables")
        st.caption(
            "Selecciona 2–4 características (máx. 2 numéricas · máx. 2 categóricas). "
            "2 numéricas → scatter · numérica + categórica → barras · "
            "añade categóricas para separar por color y forma."
        )

        exp_c1, exp_c2 = st.columns(2)
        sel_exp_num = exp_c1.multiselect(
            "Variables numéricas (eje X / Y)",
            avail_num,
            default=avail_num[:2] if len(avail_num) >= 2 else avail_num,
            key="exp_num",
        )
        sel_exp_cat = exp_c2.multiselect(
            "Variables categóricas (color / forma)",
            avail_cat,
            default=[],
            key="exp_cat",
        )

        n_num = len(sel_exp_num)
        n_cat = len(sel_exp_cat)
        total = n_num + n_cat

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

            if n_num == 2 and n_cat == 0:
                _exp_chart = (
                    _base.mark_point(size=90, opacity=0.8, filled=True)
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
                    _base.mark_point(size=90, opacity=0.8, filled=True)
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
                    _base.mark_point(size=90, opacity=0.8, filled=True)
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
                _exp_chart = (
                    alt.Chart(_agg)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X(f"{sel_exp_cat[0]}:N", title=sel_exp_cat[0], sort="-y"),
                        y=alt.Y(f"{sel_exp_num[0]}:Q", title=f"Media {sel_exp_num[0]}",
                                scale=alt.Scale(zero=False)),
                        color=alt.Color(f"{sel_exp_cat[0]}:N", title=sel_exp_cat[0],
                                        scale=MODEL_COLORS, legend=None),
                        tooltip=[
                            alt.Tooltip(f"{sel_exp_cat[0]}:N", title=sel_exp_cat[0]),
                            alt.Tooltip(f"{sel_exp_num[0]}:Q", title=f"Media {sel_exp_num[0]}", format=".3f"),
                        ],
                    )
                    .properties(height=440)
                )

            elif n_num == 1 and n_cat == 2:
                _agg = df_r.groupby(sel_exp_cat, as_index=False)[sel_exp_num[0]].mean()
                _exp_chart = (
                    alt.Chart(_agg)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X(f"{sel_exp_cat[0]}:N", title=sel_exp_cat[0], sort="-y"),
                        y=alt.Y(f"{sel_exp_num[0]}:Q", title=f"Media {sel_exp_num[0]}",
                                scale=alt.Scale(zero=False)),
                        color=alt.Color(f"{sel_exp_cat[1]}:N", title=sel_exp_cat[1], scale=MODEL_COLORS),
                        xOffset=f"{sel_exp_cat[1]}:N",
                        tooltip=[
                            alt.Tooltip(f"{sel_exp_cat[0]}:N", title=sel_exp_cat[0]),
                            alt.Tooltip(f"{sel_exp_cat[1]}:N", title=sel_exp_cat[1]),
                            alt.Tooltip(f"{sel_exp_num[0]}:Q", title=f"Media {sel_exp_num[0]}", format=".3f"),
                        ],
                    )
                    .properties(height=440)
                )

            else:
                st.warning("Combinación no soportada.")

            if _exp_chart is not None:
                st.altair_chart(_exp_chart, use_container_width=True)

        # ── TABLA COMPLETA ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("📋 Tabla Completa — Todos los Modelos y Configuraciones")

        ordered_cols = [
            "model", "fitness_type", "stop_mode", "n_decision_rules_pct",
            "best_accuracy_mean",  "best_accuracy_std",  "best_accuracy_min",  "best_accuracy_max",
            "mean_accuracy_mean",  "mean_accuracy_std",
            "mean_mse_chance_mean","mean_mse_chance_std",
            "mean_mse_utility_mean","mean_mse_utility_std",
            "best_mse_chance_mean","best_mse_utility_mean",
            "stop_gen_mean",       "stop_gen_std",        "stop_gen_min",       "stop_gen_max",
            "best_fitness_mean",   "best_fitness_std",
            "total_rules",         "n_decision_rules",
        ]
        show_cols = [c for c in ordered_cols if c in df_r.columns]
        table = df_r[show_cols].sort_values(
            ["best_accuracy_mean", "model"], ascending=[False, True]
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "model":                   st.column_config.TextColumn("Modelo"),
                "fitness_type":            st.column_config.TextColumn("Fitness"),
                "stop_mode":               st.column_config.TextColumn("Stop Mode"),
                "n_decision_rules_pct":    st.column_config.NumberColumn("% Reglas",         format="%d %%"),
                "best_accuracy_mean":      st.column_config.NumberColumn("Best Acc. Mean",   format="%.1f %%"),
                "best_accuracy_std":       st.column_config.NumberColumn("Best Acc. σ",      format="%.2f"),
                "best_accuracy_min":       st.column_config.NumberColumn("Best Acc. Min",    format="%.1f %%"),
                "best_accuracy_max":       st.column_config.NumberColumn("Best Acc. Max",    format="%.1f %%"),
                "mean_accuracy_mean":      st.column_config.NumberColumn("Mean Acc. Mean",   format="%.1f %%"),
                "mean_accuracy_std":       st.column_config.NumberColumn("Mean Acc. σ",      format="%.2f"),
                "mean_mse_chance_mean":    st.column_config.NumberColumn("MSE Chance (med)", format="%.3f"),
                "mean_mse_chance_std":     st.column_config.NumberColumn("MSE Chance σ",     format="%.3f"),
                "mean_mse_utility_mean":   st.column_config.NumberColumn("MSE Utility (med)",format="%.3f"),
                "mean_mse_utility_std":    st.column_config.NumberColumn("MSE Utility σ",    format="%.3f"),
                "best_mse_chance_mean":    st.column_config.NumberColumn("Best MSE Chance",  format="%.3f"),
                "best_mse_utility_mean":   st.column_config.NumberColumn("Best MSE Utility", format="%.3f"),
                "stop_gen_mean":           st.column_config.NumberColumn("Gen. Media",        format="%.1f"),
                "stop_gen_std":            st.column_config.NumberColumn("Gen. σ",            format="%.2f"),
                "stop_gen_min":            st.column_config.NumberColumn("Gen. Min",          format="%.0f"),
                "stop_gen_max":            st.column_config.NumberColumn("Gen. Max",          format="%.0f"),
                "best_fitness_mean":       st.column_config.NumberColumn("Best Fitness Mean", format="%.4f"),
                "best_fitness_std":        st.column_config.NumberColumn("Best Fitness σ",    format="%.4f"),
                "total_rules":             st.column_config.NumberColumn("Total Reglas"),
                "n_decision_rules":        st.column_config.NumberColumn("N Reglas"),
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CURVES
# ══════════════════════════════════════════════════════════════════════════════

CURVE_METRICS = {
    "Accuracy (%)":   "mean_accuracy",
    "Fitness":        "mean_fitness",
    "MSE Chance":     "mean_error_chance",
    "MSE Utility":    "mean_error_utility",
}

with tab_curves:
    if df_c.empty:
        st.info("Sin datos de curvas para los filtros seleccionados — puede que el grid search aún esté corriendo.")
    else:
        df_c["config"] = (
            df_c["fitness_type"] + " | "
            + df_c["stop_mode"] + " | "
            + df_c["n_decision_rules_pct"].astype(str) + "%"
        )
        df_c["model_config"] = df_c["model"] + " · " + df_c["config"]

        # ── CONTROLES ─────────────────────────────────────────────────────────
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)

        sel_metric_label = ctrl1.selectbox("Métrica principal", list(CURVE_METRICS.keys()))
        sel_metric = CURVE_METRICS[sel_metric_label]

        max_gen  = int(df_c["generation"].max())
        gen_range = ctrl2.slider("Rango de generaciones", 1, max_gen, (1, max_gen))

        _COLOR_OPTIONS = {
            "Modelo":           "model",
            "Fitness Type":     "fitness_type",
            "Stop Mode":        "stop_mode",
            "% Reglas":         "n_decision_rules_pct",
        }
        sel_color_label = ctrl3.selectbox("Color por", list(_COLOR_OPTIONS.keys()))
        sel_color_col   = _COLOR_OPTIONS[sel_color_label]
        _cenc = "O" if sel_color_col == "n_decision_rules_pct" else "N"

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

            st.altair_chart(all_curves, use_container_width=True)

            # Mini-charts de las 4 métricas en paralelo
            st.subheader(f"Curva promedio por {sel_color_label} — las cuatro métricas")
            # Forward-fill cada curva individual hasta la gen. máxima antes de promediar,
            # para que las curvas cortas (convergencia temprana) no arrastren la media a 0.
            _all_gens = sorted(df_cf["generation"].unique())
            _metric_cols = list(CURVE_METRICS.values())
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

            m_cols = st.columns(4)
            for col, (lbl, m) in zip(m_cols, CURVE_METRICS.items()):
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
                col.altair_chart(mini, use_container_width=True)

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
            st.altair_chart(ch_model, use_container_width=True)

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
                st.altair_chart(ch_stop, use_container_width=True)

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
                st.altair_chart(ch_pct, use_container_width=True)

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
            st.altair_chart(ch_ft, use_container_width=True)

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

                # 2×2 grid con las 4 métricas
                row1_c1, row1_c2 = st.columns(2)
                row2_c1, row2_c2 = st.columns(2)

                row1_c1.subheader("Accuracy (%)")
                row1_c1.altair_chart(drill_chart("mean_accuracy",     "Accuracy (%)"),   use_container_width=True)
                row1_c2.subheader("Fitness")
                row1_c2.altair_chart(drill_chart("mean_fitness",      "Fitness"),        use_container_width=True)
                row2_c1.subheader("MSE Chance")
                row2_c1.altair_chart(drill_chart("mean_error_chance",  "MSE Chance"),    use_container_width=True)
                row2_c2.subheader("MSE Utility")
                row2_c2.altair_chart(drill_chart("mean_error_utility", "MSE Utility"),   use_container_width=True)

                # Velocidad de convergencia
                st.subheader("Velocidad de convergencia — gen. donde se alcanza el 98% del accuracy final")

                conv_rows = []
                for model in models_present:
                    md = drill[drill["model"] == model].sort_values("generation")
                    if len(md) < 3:
                        continue
                    final_acc = md["mean_accuracy"].iloc[-1]
                    hit = md[md["mean_accuracy"] >= final_acc * 0.98]
                    conv_rows.append({
                        "Modelo":                  model,
                        "Gen. Conv. (98%)":        int(hit["generation"].min()) if not hit.empty else None,
                        "Accuracy Final":          round(final_acc, 2),
                        "Fitness Final":           round(md["mean_fitness"].iloc[-1], 4),
                        "MSE Chance Final":        round(md["mean_error_chance"].iloc[-1], 4),
                        "MSE Utility Final":       round(md["mean_error_utility"].iloc[-1], 4),
                    })

                if conv_rows:
                    conv_df = pd.DataFrame(conv_rows)
                    st.dataframe(conv_df, use_container_width=True, hide_index=True)

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
                    st.altair_chart(conv_bar, use_container_width=True)