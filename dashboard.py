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

        # ── CHART 1: Scatter accuracy vs gen. parada ───────────────────────────
        st.subheader("1 · Accuracy vs Generación de Parada")
        st.caption(
            "Cada punto = una configuración (fitness × stop mode × % reglas). "
            "Color = fitness type · Forma = modelo."
        )

        scatter = (
            alt.Chart(df_r)
            .mark_point(size=90, opacity=0.8, filled=True)
            .encode(
                x=alt.X("stop_gen_mean:Q", title="Generación Media de Parada"),
                y=alt.Y(
                    "best_accuracy_mean:Q",
                    title="Mejor Accuracy Medio (%)",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color("fitness_type:N", title="Fitness Type", scale=FITNESS_COLORS),
                shape=alt.Shape("model:N", title="Modelo"),
                tooltip=[
                    alt.Tooltip("model:N",               title="Modelo"),
                    alt.Tooltip("fitness_type:N",         title="Fitness"),
                    alt.Tooltip("stop_mode:N",            title="Stop Mode"),
                    alt.Tooltip("n_decision_rules_pct:Q", title="% Reglas"),
                    alt.Tooltip("best_accuracy_mean:Q",   title="Best Acc. Mean",  format=".1f"),
                    alt.Tooltip("mean_accuracy_mean:Q",   title="Mean Acc. Mean",  format=".1f"),
                    alt.Tooltip("stop_gen_mean:Q",        title="Gen. Parada",     format=".1f"),
                    alt.Tooltip("mean_mse_chance_mean:Q", title="MSE Chance",      format=".4f"),
                    alt.Tooltip("mean_mse_utility_mean:Q",title="MSE Utility",     format=".4f"),
                ],
            )
            .properties(height=400)
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)

        st.divider()

        # ── CHARTS 2 & 3: Por fitness type y por % reglas ─────────────────────
        st.subheader("2 · Accuracy Medio por Fitness Type y por % Reglas")

        ft_data = (
            df_r.groupby(["model", "fitness_type"], as_index=False)
            .agg(
                best_accuracy_mean=("best_accuracy_mean", "mean"),
                mean_accuracy_mean=("mean_accuracy_mean", "mean"),
                stop_gen_mean=("stop_gen_mean", "mean"),
            )
        )

        pct_data = (
            df_r.groupby(["model", "n_decision_rules_pct"], as_index=False)
            .agg(
                best_accuracy_mean=("best_accuracy_mean", "mean"),
                mean_accuracy_mean=("mean_accuracy_mean", "mean"),
                best_accuracy_max=("best_accuracy_max",   "max"),
            )
        )

        c2 = (
            alt.Chart(ft_data)
            .mark_bar()
            .encode(
                x=alt.X("fitness_type:N", title="Fitness Type", sort="-y"),
                y=alt.Y("best_accuracy_mean:Q", title="Best Accuracy Medio (%)",
                        scale=alt.Scale(zero=False)),
                color=alt.Color("model:N", title="Modelo", scale=MODEL_COLORS),
                xOffset="model:N",
                tooltip=[
                    "model:N", "fitness_type:N",
                    alt.Tooltip("best_accuracy_mean:Q",  title="Best Acc. Mean",  format=".2f"),
                    alt.Tooltip("mean_accuracy_mean:Q",  title="Mean Acc. Mean",  format=".2f"),
                    alt.Tooltip("stop_gen_mean:Q",       title="Gen. Media",      format=".1f"),
                ],
            )
            .properties(height=300, title="Por Fitness Type")
        )

        c3 = (
            alt.Chart(pct_data)
            .mark_bar()
            .encode(
                x=alt.X("n_decision_rules_pct:O", title="% Reglas de Decisión"),
                y=alt.Y("best_accuracy_mean:Q", title="Best Accuracy Medio (%)",
                        scale=alt.Scale(zero=False)),
                color=alt.Color("model:N", title="Modelo", scale=MODEL_COLORS),
                xOffset="model:N",
                tooltip=[
                    "model:N",
                    alt.Tooltip("n_decision_rules_pct:O",  title="% Reglas"),
                    alt.Tooltip("best_accuracy_mean:Q",    format=".2f"),
                    alt.Tooltip("best_accuracy_max:Q",     title="Máx. alcanzado", format=".2f"),
                    alt.Tooltip("mean_accuracy_mean:Q",    title="Mean Acc. Mean", format=".2f"),
                ],
            )
            .properties(height=300, title="Por % Reglas")
        )

        col_a, col_b = st.columns(2)
        col_a.altair_chart(c2, use_container_width=True)
        col_b.altair_chart(c3, use_container_width=True)

        st.divider()

        # ── CHART 4: Heatmap fitness × stop_mode por modelo ───────────────────
        st.subheader("3 · Mapa de Calor — Best Accuracy por Fitness × Stop Mode")
        st.caption("Facetado por modelo.")

        heat_data = (
            df_r.groupby(["model", "fitness_type", "stop_mode"], as_index=False)
            ["best_accuracy_mean"].mean()
        )

        heatmap = (
            alt.Chart(heat_data)
            .mark_rect(stroke="white", strokeWidth=1)
            .encode(
                x=alt.X("stop_mode:N",    title="Stop Mode",    sort=all_stop),
                y=alt.Y("fitness_type:N", title="Fitness Type", sort=all_fitness),
                color=alt.Color(
                    "best_accuracy_mean:Q",
                    title="Best Acc. Medio (%)",
                    scale=alt.Scale(scheme="blues", domain=[
                        heat_data["best_accuracy_mean"].min() - 2,
                        heat_data["best_accuracy_mean"].max(),
                    ]),
                ),
                facet=alt.Facet("model:N", columns=max(1, len(sel_models))),
                tooltip=[
                    "model:N", "fitness_type:N", "stop_mode:N",
                    alt.Tooltip("best_accuracy_mean:Q", title="Best Acc. Medio (%)", format=".1f"),
                ],
            )
            .properties(width=200, height=150)
        )
        st.altair_chart(heatmap)

        st.divider()

        # ── CHART 5: Best vs Mean accuracy por modelo ─────────────────────────
        st.subheader("4 · Best Accuracy vs Mean Accuracy por Modelo")
        st.caption(
            "Diferencia entre el mejor individuo encontrado (best) y la media de accuracy "
            "del run completo (mean). Mayor separación = el EDA encuentra buenas soluciones "
            "puntualmente pero la media del run es más baja."
        )

        bvm_data = (
            df_r.groupby("model", as_index=False)
            .agg(
                Best=("best_accuracy_mean",  "mean"),
                Mean=("mean_accuracy_mean",  "mean"),
            )
            .melt(id_vars="model", var_name="stat", value_name="accuracy")
        )

        ch5 = (
            alt.Chart(bvm_data)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("model:N", title="Modelo"),
                y=alt.Y("accuracy:Q", title="Accuracy (%)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "stat:N",
                    title="Estadístico",
                    scale=alt.Scale(
                        domain=["Best", "Mean"],
                        range=["#2ecc71", "#3498db"],
                    ),
                ),
                xOffset="stat:N",
                tooltip=["model:N", "stat:N", alt.Tooltip("accuracy:Q", format=".2f")],
            )
            .properties(height=300)
        )
        st.altair_chart(ch5, use_container_width=True)

        st.divider()

        # ── CHART 6: MSE Chance vs MSE Utility ───────────────────────────────
        st.subheader("5 · Error: MSE Chance vs MSE Utility")
        st.caption(
            "Scatter de los dos componentes del error. Cada punto es una configuración. "
            "Color = modelo · Forma = fitness type."
        )

        mse_scatter = (
            alt.Chart(df_r)
            .mark_point(size=80, opacity=0.75, filled=True)
            .encode(
                x=alt.X("mean_mse_chance_mean:Q",   title="MSE Chance (media)",   scale=alt.Scale(zero=False)),
                y=alt.Y("mean_mse_utility_mean:Q",  title="MSE Utility (media)",  scale=alt.Scale(zero=False)),
                color=alt.Color("model:N",       title="Modelo",      scale=MODEL_COLORS),
                shape=alt.Shape("fitness_type:N", title="Fitness Type"),
                tooltip=[
                    "model:N", "fitness_type:N", "stop_mode:N",
                    alt.Tooltip("n_decision_rules_pct:Q",  title="% Reglas"),
                    alt.Tooltip("mean_mse_chance_mean:Q",  title="MSE Chance",   format=".4f"),
                    alt.Tooltip("mean_mse_utility_mean:Q", title="MSE Utility",  format=".4f"),
                    alt.Tooltip("best_accuracy_mean:Q",    title="Best Acc.",    format=".1f"),
                ],
            )
            .properties(height=380)
            .interactive()
        )
        st.altair_chart(mse_scatter, use_container_width=True)

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
        ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])

        sel_metric_label = ctrl1.selectbox("Métrica principal", list(CURVE_METRICS.keys()))
        sel_metric = CURVE_METRICS[sel_metric_label]

        max_gen  = int(df_c["generation"].max())
        gen_range = ctrl2.slider("Rango de generaciones", 1, max_gen, (1, max_gen))

        curve_fitness = ctrl3.multiselect(
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
            st.caption("Clic en el modelo de la leyenda para resaltar sus curvas.")

            sel = alt.selection_point(fields=["model"], bind="legend")

            base = alt.Chart(df_cf).encode(
                x=alt.X("generation:Q", title="Generación"),
                y=alt.Y(f"{sel_metric}:Q", title=sel_metric_label,
                        scale=alt.Scale(zero=False)),
                color=alt.Color("model:N", title="Modelo", scale=MODEL_COLORS),
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
            st.subheader("Curva promedio por modelo — las cuatro métricas")
            m_cols = st.columns(4)
            for col, (lbl, m) in zip(m_cols, CURVE_METRICS.items()):
                avg = df_cf.groupby(["model", "generation"], as_index=False)[m].mean()
                mini = (
                    alt.Chart(avg)
                    .mark_line(strokeWidth=2.5)
                    .encode(
                        x=alt.X("generation:Q", title="Gen."),
                        y=alt.Y(f"{m}:Q", title=lbl, scale=alt.Scale(zero=False)),
                        color=alt.Color("model:N", legend=None, scale=MODEL_COLORS),
                        tooltip=["model:N", "generation:Q",
                                 alt.Tooltip(f"{m}:Q", format=".4f")],
                    )
                    .properties(height=210, title=lbl)
                    .interactive()
                )
                col.altair_chart(mini, use_container_width=True)

        # ── CTAB 2: Promedios por factor ───────────────────────────────────────
        with ctab2:
            st.subheader(f"Promedio de {sel_metric_label} por Modelo")

            avg_model = df_cf.groupby(["model", "generation"], as_index=False)[sel_metric].mean()
            ch_model = (
                alt.Chart(avg_model)
                .mark_line(strokeWidth=3)
                .encode(
                    x=alt.X("generation:Q", title="Generación"),
                    y=alt.Y(f"{sel_metric}:Q", title=sel_metric_label,
                            scale=alt.Scale(zero=False)),
                    color=alt.Color("model:N", title="Modelo", scale=MODEL_COLORS),
                    tooltip=["model:N", "generation:Q",
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