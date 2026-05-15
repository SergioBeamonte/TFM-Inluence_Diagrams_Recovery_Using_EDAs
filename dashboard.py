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
st.caption("Barrido paramétrico para recuperación de IDs con EDAs. Carga automática de todos los modelos disponibles.")

# ─── DISCOVERY ────────────────────────────────────────────────────────────────

@st.cache_data
def discover_models():
    found = {}
    for f in glob.glob("**/grid_search_results_*.csv", recursive=True):
        model = os.path.basename(f).replace("grid_search_results_", "").replace(".csv", "")
        curves = f.replace("results", "curves")
        found[model] = {
            "results": os.path.normpath(f),
            "curves": os.path.normpath(curves) if os.path.exists(curves) else None,
        }
    return found


@st.cache_data
def load_all(model_dict_frozen):
    res_list, cur_list = [], []
    for model, (results_path, curves_path) in model_dict_frozen:
        if results_path and os.path.exists(results_path):
            df = pd.read_csv(results_path)
            df.insert(0, "model", model)
            res_list.append(df)
        if curves_path and os.path.exists(curves_path):
            df = pd.read_csv(curves_path)
            df.insert(0, "model", model)
            cur_list.append(df)
    df_r = pd.concat(res_list, ignore_index=True) if res_list else pd.DataFrame()
    df_c = pd.concat(cur_list, ignore_index=True) if cur_list else pd.DataFrame()
    return df_r, df_c


model_dict = discover_models()

if not model_dict:
    st.error(
        "No se encontraron archivos `grid_search_results_MODELO.csv` en ningún subdirectorio. "
        "Asegúrate de ejecutar el grid search primero."
    )
    st.stop()

# Freeze dict for cache key (must be hashable)
df_results_all, df_curves_all = load_all(tuple(sorted(
    (k, (v["results"], v["curves"])) for k, v in model_dict.items()
)))

# ─── SIDEBAR: FILTROS GLOBALES ─────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Filtros Globales")
    st.caption(f"Modelos disponibles: {', '.join(sorted(model_dict.keys()))}")

    all_models = sorted(df_results_all["model"].unique())
    sel_models = st.multiselect("Modelos", all_models, default=all_models)

    all_fitness = sorted(df_results_all["fitness_type"].unique())
    sel_fitness = st.multiselect("Tipo de Fitness", all_fitness, default=all_fitness)

    all_stop = sorted(df_results_all["stop_mode"].unique())
    sel_stop = st.multiselect("Modo de Parada", all_stop, default=all_stop)

    all_pct = sorted(df_results_all["n_decision_rules_pct"].unique())
    sel_pct = st.multiselect("% Reglas de Decisión", all_pct, default=all_pct)

    st.divider()
    st.caption("Los filtros se aplican a ambas pestañas.")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["model"].isin(sel_models)
        & df["fitness_type"].isin(sel_fitness)
        & df["stop_mode"].isin(sel_stop)
        & df["n_decision_rules_pct"].isin(sel_pct)
    ].copy()


df_r = apply_filters(df_results_all)
df_c = apply_filters(df_curves_all) if not df_curves_all.empty else pd.DataFrame()

if df_r.empty:
    st.warning("No hay datos para la combinación de filtros seleccionada.")
    st.stop()

# ─── TABS ─────────────────────────────────────────────────────────────────────

tab_results, tab_curves = st.tabs(["📋 Resultados por Modelo", "📈 Curvas de Convergencia"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

with tab_results:
    # ── KPIs ──────────────────────────────────────────────────────────────────
    best_row = df_r.loc[df_r["accuracy_media"].idxmax()]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Configuraciones", len(df_r))
    k2.metric("Mejor Accuracy Medio", f"{best_row['accuracy_media']:.1f}%",
              delta=f"{best_row['model']} · {best_row['fitness_type']} · {best_row['stop_mode']} · {best_row['n_decision_rules_pct']}%")
    k3.metric("Mejor Accuracy Puntual", f"{df_r['accuracy_mejor'].max():.1f}%")
    k4.metric("Menor Error Medio", f"{df_r['error_media'].min():.4f}")
    k5.metric("Gen. Media de Parada", f"{df_r['stop_gen_media'].mean():.1f}")

    st.divider()

    # ── CHART 1: Accuracy media por modelo × % reglas ─────────────────────────
    st.subheader("1 · Accuracy Medio por % Reglas y Modelo")

    bar_pct = (
        df_r.groupby(["model", "n_decision_rules_pct"], as_index=False)
        .agg(accuracy_media=("accuracy_media", "mean"),
             accuracy_mejor=("accuracy_mejor", "max"),
             error_media=("error_media", "mean"))
    )

    ch1 = (
        alt.Chart(bar_pct)
        .mark_bar()
        .encode(
            x=alt.X("n_decision_rules_pct:O", title="% Reglas de Decisión"),
            y=alt.Y("accuracy_media:Q", title="Accuracy Medio (%)", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title="Modelo", scale=alt.Scale(scheme="tableau10")),
            xOffset=alt.XOffset("model:N"),
            tooltip=[
                alt.Tooltip("model:N", title="Modelo"),
                alt.Tooltip("n_decision_rules_pct:O", title="% Reglas"),
                alt.Tooltip("accuracy_media:Q", title="Accuracy Medio", format=".2f"),
                alt.Tooltip("accuracy_mejor:Q", title="Mejor Accuracy", format=".2f"),
                alt.Tooltip("error_media:Q", title="Error Medio", format=".4f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(ch1, use_container_width=True)

    # ── CHART 2: Accuracy por fitness_type × modelo ───────────────────────────
    st.subheader("2 · Accuracy y Error por Fitness Type")

    ft_data = (
        df_r.groupby(["model", "fitness_type"], as_index=False)
        .agg(accuracy_media=("accuracy_media", "mean"),
             error_media=("error_media", "mean"))
    )

    c2a = (
        alt.Chart(ft_data)
        .mark_bar()
        .encode(
            x=alt.X("fitness_type:N", title="Fitness Type"),
            y=alt.Y("accuracy_media:Q", title="Accuracy Medio (%)", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title="Modelo", scale=alt.Scale(scheme="tableau10")),
            xOffset="model:N",
            tooltip=["model:N", "fitness_type:N",
                     alt.Tooltip("accuracy_media:Q", format=".2f")],
        )
        .properties(height=300, title="Accuracy Medio por Fitness Type")
    )

    c2b = (
        alt.Chart(ft_data)
        .mark_bar()
        .encode(
            x=alt.X("fitness_type:N", title="Fitness Type"),
            y=alt.Y("error_media:Q", title="Error Medio (MSE)", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title="Modelo", scale=alt.Scale(scheme="tableau10")),
            xOffset="model:N",
            tooltip=["model:N", "fitness_type:N",
                     alt.Tooltip("error_media:Q", format=".4f")],
        )
        .properties(height=300, title="Error Medio por Fitness Type")
    )

    ca, cb = st.columns(2)
    ca.altair_chart(c2a, use_container_width=True)
    cb.altair_chart(c2b, use_container_width=True)

    # ── CHART 3: Stop mode × accuracy ─────────────────────────────────────────
    st.subheader("3 · Impacto del Stop Mode por Modelo")

    sm_data = (
        df_r.groupby(["model", "stop_mode"], as_index=False)
        .agg(accuracy_media=("accuracy_media", "mean"),
             stop_gen_media=("stop_gen_media", "mean"))
    )

    c3a = (
        alt.Chart(sm_data)
        .mark_bar()
        .encode(
            x=alt.X("stop_mode:N", title="Stop Mode"),
            y=alt.Y("accuracy_media:Q", title="Accuracy Medio (%)", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", scale=alt.Scale(scheme="tableau10")),
            xOffset="model:N",
            tooltip=["model:N", "stop_mode:N",
                     alt.Tooltip("accuracy_media:Q", format=".2f")],
        )
        .properties(height=280, title="Accuracy por Stop Mode")
    )

    c3b = (
        alt.Chart(sm_data)
        .mark_bar()
        .encode(
            x=alt.X("stop_mode:N", title="Stop Mode"),
            y=alt.Y("stop_gen_media:Q", title="Generación Media de Parada"),
            color=alt.Color("model:N", scale=alt.Scale(scheme="tableau10")),
            xOffset="model:N",
            tooltip=["model:N", "stop_mode:N",
                     alt.Tooltip("stop_gen_media:Q", format=".1f")],
        )
        .properties(height=280, title="Generación de Parada por Stop Mode")
    )

    cc, cd = st.columns(2)
    cc.altair_chart(c3a, use_container_width=True)
    cd.altair_chart(c3b, use_container_width=True)

    # ── CHART 4: Scatter accuracy vs stop_gen ─────────────────────────────────
    st.subheader("4 · Accuracy vs Generación de Parada (todas las configs)")

    sc4 = (
        alt.Chart(df_r)
        .mark_circle(size=80, opacity=0.75)
        .encode(
            x=alt.X("stop_gen_media:Q", title="Generación Media de Parada"),
            y=alt.Y("accuracy_media:Q", title="Accuracy Medio (%)",
                    scale=alt.Scale(zero=False)),
            color=alt.Color("fitness_type:N", title="Fitness Type",
                            scale=alt.Scale(scheme="tableau10")),
            shape=alt.Shape("model:N", title="Modelo"),
            tooltip=[
                "model:N", "fitness_type:N", "stop_mode:N",
                alt.Tooltip("n_decision_rules_pct:Q", title="% Reglas"),
                alt.Tooltip("accuracy_media:Q", format=".2f"),
                alt.Tooltip("stop_gen_media:Q", format=".1f"),
                alt.Tooltip("error_media:Q", format=".4f"),
            ],
        )
        .properties(height=380)
        .interactive()
    )
    st.altair_chart(sc4, use_container_width=True)

    # ── CHART 5: Mapa de calor accuracy por fitness × stop_mode ──────────────
    st.subheader("5 · Mapa de Calor — Accuracy por Fitness × Stop Mode")

    heat_data = (
        df_r.groupby(["model", "fitness_type", "stop_mode"], as_index=False)
        ["accuracy_media"].mean()
    )

    heatmap = (
        alt.Chart(heat_data)
        .mark_rect()
        .encode(
            x=alt.X("stop_mode:N", title="Stop Mode"),
            y=alt.Y("fitness_type:N", title="Fitness Type"),
            color=alt.Color("accuracy_media:Q", title="Accuracy Medio (%)",
                            scale=alt.Scale(scheme="blues")),
            facet=alt.Facet("model:N", columns=max(1, len(sel_models))),
            tooltip=[
                "model:N", "fitness_type:N", "stop_mode:N",
                alt.Tooltip("accuracy_media:Q", format=".2f"),
            ],
        )
        .properties(width=180, height=130)
    )
    st.altair_chart(heatmap)

    # ── CHART 6: Distribución mejor / media / peor por modelo ─────────────────
    st.subheader("6 · Distribución de Accuracy por Modelo (Mejor / Media / Peor)")

    box_data = (
        df_r.groupby("model", as_index=False)
        .agg(
            Mejor=("accuracy_mejor", "mean"),
            Media=("accuracy_media", "mean"),
            Peor=("accuracy_peor", "mean"),
        )
        .melt(id_vars="model", var_name="stat", value_name="accuracy")
    )

    ch6 = (
        alt.Chart(box_data)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title="Modelo"),
            y=alt.Y("accuracy:Q", title="Accuracy (%)", scale=alt.Scale(zero=False)),
            color=alt.Color("stat:N", title="Estadístico",
                            scale=alt.Scale(
                                domain=["Mejor", "Media", "Peor"],
                                range=["#2ecc71", "#3498db", "#e74c3c"],
                            )),
            xOffset="stat:N",
            tooltip=["model:N", "stat:N",
                     alt.Tooltip("accuracy:Q", format=".2f")],
        )
        .properties(height=300)
    )
    st.altair_chart(ch6, use_container_width=True)

    # ── TABLA COMPLETA ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Tabla Completa — Todos los Modelos y Configuraciones")

    ordered_cols = [
        "model", "fitness_type", "stop_mode", "n_decision_rules_pct",
        "accuracy_mejor", "accuracy_media", "accuracy_peor", "accuracy_std",
        "error_mejor", "error_media", "error_peor", "error_std",
        "stop_gen_mejor", "stop_gen_media", "stop_gen_peor", "stop_gen_std",
        "fitness_mejor", "fitness_media", "fitness_peor", "fitness_std",
        "total_rules", "n_decision_rules",
    ]
    show_cols = [c for c in ordered_cols if c in df_r.columns]
    table = df_r[show_cols].sort_values(["accuracy_media", "model"], ascending=[False, True])

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "model":                  st.column_config.TextColumn("Modelo"),
            "fitness_type":           st.column_config.TextColumn("Fitness"),
            "stop_mode":              st.column_config.TextColumn("Stop Mode"),
            "n_decision_rules_pct":   st.column_config.NumberColumn("% Reglas", format="%d %%"),
            "accuracy_mejor":         st.column_config.NumberColumn("Acc. Mejor", format="%.1f %%"),
            "accuracy_media":         st.column_config.NumberColumn("Acc. Media", format="%.1f %%"),
            "accuracy_peor":          st.column_config.NumberColumn("Acc. Peor", format="%.1f %%"),
            "accuracy_std":           st.column_config.NumberColumn("Acc. σ",    format="%.2f"),
            "error_mejor":            st.column_config.NumberColumn("Err. Mejor", format="%.4f"),
            "error_media":            st.column_config.NumberColumn("Err. Media", format="%.4f"),
            "error_peor":             st.column_config.NumberColumn("Err. Peor",  format="%.4f"),
            "error_std":              st.column_config.NumberColumn("Err. σ",     format="%.4f"),
            "stop_gen_mejor":         st.column_config.NumberColumn("Gen. Min",   format="%.0f"),
            "stop_gen_media":         st.column_config.NumberColumn("Gen. Media", format="%.1f"),
            "stop_gen_peor":          st.column_config.NumberColumn("Gen. Max",   format="%.0f"),
            "stop_gen_std":           st.column_config.NumberColumn("Gen. σ",     format="%.2f"),
            "fitness_mejor":          st.column_config.NumberColumn("Fit. Mejor", format="%.4f"),
            "fitness_media":          st.column_config.NumberColumn("Fit. Media", format="%.4f"),
            "fitness_peor":           st.column_config.NumberColumn("Fit. Peor",  format="%.4f"),
            "fitness_std":            st.column_config.NumberColumn("Fit. σ",     format="%.4f"),
            "total_rules":            st.column_config.NumberColumn("Total Reglas"),
            "n_decision_rules":       st.column_config.NumberColumn("N Reglas"),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CURVES
# ══════════════════════════════════════════════════════════════════════════════

with tab_curves:
    if df_c.empty:
        st.info("No hay curvas disponibles para los modelos/filtros seleccionados.")
        st.stop()

    # Etiqueta de config para leyendas
    df_c["config"] = (
        df_c["fitness_type"] + " | "
        + df_c["stop_mode"] + " | "
        + df_c["n_decision_rules_pct"].astype(str) + "%"
    )
    df_c["model_config"] = df_c["model"] + " · " + df_c["config"]

    # ── CONTROLES LOCALES ──────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])

    metric_map = {"Accuracy (%)": "accuracy", "Fitness": "fitness", "MSE": "mse"}
    sel_metric_label = ctrl1.selectbox("Métrica principal", list(metric_map.keys()))
    sel_metric = metric_map[sel_metric_label]

    max_gen = int(df_c["generation"].max())
    gen_range = ctrl2.slider("Rango de generaciones", 1, max_gen, (1, max_gen))

    # Añadir multiselect de fitness_type específico para curvas
    curve_fitness = ctrl3.multiselect(
        "Filtro adicional de Fitness (curvas)",
        sorted(df_c["fitness_type"].unique()),
        default=sorted(df_c["fitness_type"].unique()),
    )

    df_c_f = df_c[
        (df_c["generation"] >= gen_range[0])
        & (df_c["generation"] <= gen_range[1])
        & (df_c["fitness_type"].isin(curve_fitness))
    ].copy()

    if df_c_f.empty:
        st.warning("No hay curvas para esta selección.")
        st.stop()

    # ── SUB-TABS ───────────────────────────────────────────────────────────────
    ctab1, ctab2, ctab3 = st.tabs([
        "🌐 Comparación Global",
        "📉 Promedio por Modelo",
        "🔬 Drilldown por Config",
    ])

    # ── CTAB 1: Todas las curvas ───────────────────────────────────────────────
    with ctab1:
        st.subheader(f"{sel_metric_label} — Todas las configuraciones")
        st.caption("Haz clic en una serie de la leyenda para resaltarla.")

        highlight = alt.selection_point(fields=["model"], bind="legend")

        base = alt.Chart(df_c_f).encode(
            x=alt.X("generation:Q", title="Generación"),
            y=alt.Y(f"{sel_metric}:Q", title=sel_metric_label,
                    scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title="Modelo",
                            scale=alt.Scale(scheme="tableau10")),
            detail="model_config:N",
            opacity=alt.condition(highlight, alt.value(0.85), alt.value(0.08)),
            tooltip=[
                "model:N", "fitness_type:N", "stop_mode:N",
                alt.Tooltip("n_decision_rules_pct:O", title="% Reglas"),
                alt.Tooltip(f"{sel_metric}:Q", format=".4f"),
                "generation:Q",
            ],
        )

        chart_all = (
            base.mark_line(strokeWidth=1.2)
            + base.mark_circle(size=1).add_params(highlight)
        ).properties(height=420).interactive()

        st.altair_chart(chart_all, use_container_width=True)

        # Añadir también las tres métricas en paralelo para contexto general
        st.subheader("Todas las métricas — vista rápida")
        metrics_all = ["accuracy", "fitness", "mse"]
        labels_all = ["Accuracy (%)", "Fitness", "MSE"]

        m_cols = st.columns(3)
        for col, m, lbl in zip(m_cols, metrics_all, labels_all):
            avg_all = df_c_f.groupby(["model", "generation"], as_index=False)[m].mean()
            mini = (
                alt.Chart(avg_all)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("generation:Q", title="Gen."),
                    y=alt.Y(f"{m}:Q", title=lbl, scale=alt.Scale(zero=False)),
                    color=alt.Color("model:N", legend=None,
                                    scale=alt.Scale(scheme="tableau10")),
                    tooltip=["model:N", "generation:Q",
                             alt.Tooltip(f"{m}:Q", format=".4f")],
                )
                .properties(height=220, title=lbl)
                .interactive()
            )
            col.altair_chart(mini, use_container_width=True)

    # ── CTAB 2: Promedio por modelo ────────────────────────────────────────────
    with ctab2:
        st.subheader(f"{sel_metric_label} promediado sobre todas las configs — por Modelo")

        avg_model = df_c_f.groupby(["model", "generation"], as_index=False)[sel_metric].mean()

        ch_avg = (
            alt.Chart(avg_model)
            .mark_line(strokeWidth=3)
            .encode(
                x=alt.X("generation:Q", title="Generación"),
                y=alt.Y(f"{sel_metric}:Q", title=f"{sel_metric_label} (media)",
                        scale=alt.Scale(zero=False)),
                color=alt.Color("model:N", title="Modelo",
                                scale=alt.Scale(scheme="tableau10")),
                tooltip=["model:N", "generation:Q",
                         alt.Tooltip(f"{sel_metric}:Q", format=".4f")],
            )
            .properties(height=380)
            .interactive()
        )
        st.altair_chart(ch_avg, use_container_width=True)

        # Comparación por stop_mode promediada
        st.subheader(f"{sel_metric_label} por Stop Mode (promedio sobre modelos y configs)")

        avg_stop = df_c_f.groupby(["stop_mode", "generation"], as_index=False)[sel_metric].mean()

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
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(ch_stop, use_container_width=True)

        # Comparación por % reglas promediada
        st.subheader(f"{sel_metric_label} por % Reglas (promedio sobre modelos y configs)")

        avg_pct = df_c_f.groupby(["n_decision_rules_pct", "generation"], as_index=False)[sel_metric].mean()
        avg_pct["pct_label"] = avg_pct["n_decision_rules_pct"].astype(str) + "%"

        ch_pct = (
            alt.Chart(avg_pct)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("generation:Q", title="Generación"),
                y=alt.Y(f"{sel_metric}:Q", title=sel_metric_label,
                        scale=alt.Scale(zero=False)),
                color=alt.Color("pct_label:N", title="% Reglas",
                                scale=alt.Scale(scheme="oranges")),
                tooltip=["pct_label:N", "generation:Q",
                         alt.Tooltip(f"{sel_metric}:Q", format=".4f")],
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(ch_pct, use_container_width=True)

    # ── CTAB 3: Drilldown ─────────────────────────────────────────────────────
    with ctab3:
        st.subheader("🔬 Fija una configuración y compara modelos")

        d1, d2, d3 = st.columns(3)

        avail_ft = sorted(df_c_f["fitness_type"].unique())
        d_ft = d1.selectbox("Fitness Type", avail_ft, key="d_ft")

        avail_stop = sorted(
            df_c_f[df_c_f["fitness_type"] == d_ft]["stop_mode"].unique()
        )
        d_stop = d2.selectbox("Stop Mode", avail_stop, key="d_stop")

        avail_pct = sorted(
            df_c_f[
                (df_c_f["fitness_type"] == d_ft) & (df_c_f["stop_mode"] == d_stop)
            ]["n_decision_rules_pct"].unique()
        )
        d_pct = d3.selectbox("% Reglas", avail_pct, key="d_pct")

        drill = df_c_f[
            (df_c_f["fitness_type"] == d_ft)
            & (df_c_f["stop_mode"] == d_stop)
            & (df_c_f["n_decision_rules_pct"] == d_pct)
        ].copy()

        if drill.empty:
            st.info("No hay datos para esta combinación.")
        else:
            st.caption(
                f"Config: **{d_ft}** · **{d_stop}** · **{d_pct}%** reglas  "
                f"— Modelos: {', '.join(sorted(drill['model'].unique()))}"
            )

            def make_drill_chart(data, y_field, y_title):
                return (
                    alt.Chart(data)
                    .mark_line(strokeWidth=2.5)
                    .encode(
                        x=alt.X("generation:Q", title="Generación"),
                        y=alt.Y(f"{y_field}:Q", title=y_title,
                                scale=alt.Scale(zero=False)),
                        color=alt.Color("model:N", title="Modelo",
                                        scale=alt.Scale(scheme="tableau10")),
                        tooltip=["model:N", "generation:Q",
                                 alt.Tooltip(f"{y_field}:Q", format=".4f")],
                    )
                    .properties(height=270)
                    .interactive()
                )

            m1, m2, m3 = st.columns(3)
            m1.subheader("Accuracy (%)")
            m1.altair_chart(make_drill_chart(drill, "accuracy", "Accuracy (%)"),
                            use_container_width=True)
            m2.subheader("Fitness")
            m2.altair_chart(make_drill_chart(drill, "fitness", "Fitness"),
                            use_container_width=True)
            m3.subheader("MSE")
            m3.altair_chart(make_drill_chart(drill, "mse", "MSE"),
                            use_container_width=True)

            # Velocidad de convergencia
            st.subheader("⚡ Velocidad de Convergencia — Gen. donde se alcanza el 98% del Accuracy final")

            conv_rows = []
            for model in sorted(drill["model"].unique()):
                m_data = drill[drill["model"] == model].sort_values("generation")
                if len(m_data) >= 3:
                    final_acc = m_data["accuracy"].iloc[-1]
                    threshold = final_acc * 0.98
                    conv_gen_row = m_data[m_data["accuracy"] >= threshold]
                    conv_gen = int(conv_gen_row["generation"].min()) if not conv_gen_row.empty else None
                    final_fit = m_data["fitness"].iloc[-1]
                    final_mse = m_data["mse"].iloc[-1]
                    conv_rows.append({
                        "Modelo": model,
                        "Gen. Convergencia (98%)": conv_gen,
                        "Accuracy Final": round(final_acc, 2),
                        "Fitness Final": round(final_fit, 4),
                        "MSE Final": round(final_mse, 4),
                    })

            if conv_rows:
                conv_df = pd.DataFrame(conv_rows)
                st.dataframe(conv_df, use_container_width=True, hide_index=True)

                conv_bar = (
                    alt.Chart(conv_df.dropna(subset=["Gen. Convergencia (98%)"]))
                    .mark_bar()
                    .encode(
                        x=alt.X("Modelo:N", title="Modelo"),
                        y=alt.Y("Gen. Convergencia (98%):Q",
                                title="Generación de Convergencia"),
                        color=alt.Color("Modelo:N",
                                        scale=alt.Scale(scheme="tableau10")),
                        tooltip=["Modelo:N", "Gen. Convergencia (98%):Q",
                                 alt.Tooltip("Accuracy Final:Q", format=".2f")],
                    )
                    .properties(height=250)
                )
                st.altair_chart(conv_bar, use_container_width=True)
