import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Dashboard Frío vs Caliente", layout="wide")

st.title("Inicialización en frío vs en caliente (capacidad de la población)")
st.caption(
    "Para cada nº de reglas de training k=1..10 se entrena el EDA y se observa qué "
    "fracción de la POBLACIÓN recupera TODAS las reglas reales (`pct_pop_full`). "
    "**frío** = init aleatorio en cada k (hasta 100 gens); **caliente** = se reajusta "
    "a la población final de k−1 (hasta 30 gens). La estrella marca la generación en "
    "que la población cruza el 50%."
)

csv_path = r"example\explore_init_coldwarm.csv"

if not os.path.exists(csv_path):
    st.error(f"No se encuentra {csv_path}. Ejecuta explore_capacity.py primero.")
    st.stop()


@st.cache_data(ttl=30)
def load_data():
    df = pd.read_csv(csv_path)
    df["star_cross50"] = (
        df["star_cross50"].astype(str).str.strip().str.lower().isin(["true", "1"])
    )
    return df


df = load_data()
if df.empty or "method" not in df.columns:
    st.warning("El CSV aún no tiene datos del estudio frío/caliente (puede estar generándose).")
    st.stop()

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.header("Filtros")

methods = sorted(df["method"].unique().tolist())
sel_methods = st.sidebar.multiselect("Método", methods, default=methods)

optimizers = sorted(df["optimizer"].unique().tolist())
sel_opts = st.sidebar.multiselect("Optimizador", optimizers, default=optimizers)

fitness_types = sorted(df["fitness_type"].unique().tolist())
sel_fit = st.sidebar.multiselect("Función Fitness", fitness_types, default=fitness_types)

reps = ["Todas (Promedio)"] + sorted(df["rep"].unique().tolist())
sel_rep = st.sidebar.selectbox("Repetición (Seed)", reps)

x_options = {
    "Generación acumulada": "cum_gen",
    "Nº reglas de training": "n_train_rules",
    "Generación dentro del nº de reglas": "stage_gen",
}
x_label = st.sidebar.selectbox("Eje X", list(x_options.keys()))
x_col = x_options[x_label]

metrics = ["pct_pop_full", "max_accuracy", "mean_accuracy", "n_rules_correct"]
metrics = [m for m in metrics if m in df.columns]
sel_metric = st.sidebar.selectbox("Métrica Eje Y", metrics, index=0)

color_options = {
    "Método (frío/caliente)": "method",
    "Optimizador": "optimizer",
    "Función Fitness": "fitness_type",
    "Experimento (método+opt+fitness)": "Experiment",
}
color_label = st.sidebar.selectbox("Colorear por", list(color_options.keys()))
color_col = color_options[color_label]

show_stars = st.sidebar.checkbox("Mostrar estrellas (cruce del 50%)", value=True)

# ─── FILTRADO ───────────────────────────────────────────────────────────────────
base = df[
    df["method"].isin(sel_methods)
    & df["optimizer"].isin(sel_opts)
    & df["fitness_type"].isin(sel_fit)
].copy()
if base.empty:
    st.warning("No hay datos para la selección.")
    st.stop()
if sel_rep != "Todas (Promedio)":
    base = base[base["rep"] == sel_rep].copy()

base["Experiment"] = base["method"] + " · " + base["optimizer"] + " · " + base["fitness_type"]
ID = ["method", "optimizer", "fitness_type"]

# Curva: por generación (cum_gen/stage_gen) o colapsada por nº de reglas (pico por k).
if x_col == "n_train_rules":
    per_rep = base.groupby(ID + ["rep", x_col], as_index=False)[sel_metric].max()
    if sel_rep == "Todas (Promedio)":
        curve = per_rep.groupby(ID + [x_col], as_index=False)[sel_metric].mean()
    else:
        curve = per_rep
else:
    if sel_rep == "Todas (Promedio)":
        curve = base.groupby(ID + [x_col], as_index=False)[sel_metric].mean()
    else:
        curve = base.copy()
curve["Experiment"] = curve["method"] + " · " + curve["optimizer"] + " · " + curve["fitness_type"]
curve = curve.sort_values(ID + [x_col])

# ─── GRÁFICO PRINCIPAL ──────────────────────────────────────────────────────────
fig = px.line(curve, x=x_col, y=sel_metric, color=color_col, line_group="Experiment",
              title=f"{sel_metric} vs {x_label}",
              labels={x_col: x_label, sel_metric: sel_metric},
              markers=(x_col == "n_train_rules"))

# Línea de referencia del 50% cuando la métrica es la fracción de población perfecta.
if sel_metric == "pct_pop_full":
    fig.add_hline(y=50, line_dash="dot", line_color="gray",
                  annotation_text="50%", annotation_position="top left")

# Estrellas: generaciones donde la población cruza el 50% (color por método).
if show_stars:
    stars = base[base["star_cross50"]].copy()
    if not stars.empty:
        _METHOD_COLOR = {"cold": "#1f77b4", "warm": "#d62728"}
        for m, sub in stars.groupby("method"):
            fig.add_trace(go.Scatter(
                x=sub[x_col], y=sub[sel_metric], mode="markers",
                marker=dict(size=12, symbol="star",
                            color=_METHOD_COLOR.get(m, "#444"),
                            line=dict(width=1, color=_METHOD_COLOR.get(m, "#444"))),
                name=f"Cruce 50% ({m})",
                hovertext=("opt=" + sub["optimizer"] + " · fit=" + sub["fitness_type"]
                           + " · k=" + sub["n_train_rules"].astype(str)),
                hoverinfo="text+x+y",
            ))

st.plotly_chart(fig, use_container_width=True)

# ─── RESUMEN: ¿cuándo cruza el 50%? frío vs caliente ────────────────────────────
st.subheader("¿Cuándo cruza la población el 50%? — frío vs caliente")
st.caption("Primer cruce del 50% por corrida: con cuántas reglas (k) y en qué generación acumulada.")

cross = (base[base["star_cross50"]]
         .sort_values("cum_gen")
         .groupby(["method", "optimizer", "fitness_type", "rep"], as_index=False)
         .head(1))

if cross.empty:
    st.info("Todavía ninguna corrida cruza el 50% en los datos cargados.")
else:
    c1, c2, c3 = st.columns(3)
    for col, m in zip((c1, c2), ["cold", "warm"]):
        sub = cross[cross["method"] == m]
        if not sub.empty:
            col.metric(
                f"{m}: reglas medias hasta 50%",
                f"{sub['n_train_rules'].mean():.1f}",
                delta=f"{sub['cum_gen'].mean():.0f} gens acum.",
                delta_color="off",
            )
    # % de corridas que llegan al 95% (parada)
    reached = (base.groupby(["method", "optimizer", "fitness_type", "rep"])["pct_pop_full"]
               .max().reset_index())
    reached["95"] = reached["pct_pop_full"] >= 95
    pct95 = reached.groupby("method")["95"].mean().mul(100).round(0)
    c3.metric("Corridas que superan 95%",
              " / ".join(f"{m}: {int(pct95.get(m, 0))}%" for m in methods))

    bar = px.bar(
        cross.groupby(["method", "optimizer"], as_index=False)["n_train_rules"].mean(),
        x="optimizer", y="n_train_rules", color="method", barmode="group",
        title="Reglas de training medias hasta cruzar el 50% (frío vs caliente)",
        labels={"n_train_rules": "Reglas hasta 50% (media)", "optimizer": "Optimizador",
                "method": "Método"},
    )
    st.plotly_chart(bar, use_container_width=True)

st.subheader("Datos (filtrados)")
st.dataframe(base, use_container_width=True, hide_index=True)
