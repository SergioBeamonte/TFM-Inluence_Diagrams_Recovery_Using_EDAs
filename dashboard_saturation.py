import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Dashboard Saturación", layout="wide")

st.title("Visualización de Saturación (reglas necesarias por estancamiento)")
st.caption(
    "Se arranca con 1 regla y se añade otra cada vez que la accuracy real se "
    "estanca. El estudio mide **cuántas reglas se necesitan** para que el 50% de "
    "la población cumpla todas las reglas reales. Las estrellas marcan cada nueva regla."
)

csv_path = r"example\explore_saturation.csv"

if not os.path.exists(csv_path):
    st.error(f"No se encuentra el archivo: {csv_path}. Ejecuta explore_saturation.py primero.")
    st.stop()


@st.cache_data
def load_data():
    df = pd.read_csv(csv_path)
    df["rule_added_after_gen"] = (
        df["rule_added_after_gen"].astype(str).str.strip().str.lower().isin(["true", "1"])
    )
    return df


df = load_data()

# ─── SIDEBAR / FILTROS ──────────────────────────────────────────────────────────
st.sidebar.header("Filtros")

nets = df["net"].unique().tolist()
selected_nets = st.sidebar.multiselect("Red (net)", nets, default=nets)

optimizers = df["optimizer"].unique().tolist()
selected_opts = st.sidebar.multiselect("Optimizador", optimizers, default=optimizers)

fitness_types = df["fitness_type"].unique().tolist()
selected_fit = st.sidebar.multiselect("Función Fitness", fitness_types, default=fitness_types)

reps = ["Todas (Promedio)"] + sorted(df["rep"].unique().tolist())
selected_rep = st.sidebar.selectbox("Repetición (Seed)", reps)

# Métrica del eje Y. n_train_rules (reglas en el pool) es la curva de saturación.
metrics = ["n_train_rules", "max_accuracy", "mean_accuracy",
           "pct_success_indv", "n_rules_correct", "gen_cpu_time"]
metrics = [m for m in metrics if m in df.columns]
selected_metric = st.sidebar.selectbox("Métrica Eje Y", metrics, index=0)

# Característica por la que agrupar/colorear las curvas.
color_options = {
    "Experimento (net+opt+fitness+modo)": "Experiment",
    "Modo": "mode",
    "Función Fitness": "fitness_type",
    "Optimizador": "optimizer",
    "Red (net)": "net",
}
color_label = st.sidebar.selectbox("Colorear curvas por", list(color_options.keys()))
color_col = color_options[color_label]

# Estrellas = generaciones donde se añadió una regla. Se pueden ocultar para
# ver mejor las curvas.
show_stars = st.sidebar.checkbox("Mostrar estrellas (nuevas reglas)", value=True)

# ─── FILTRADO ───────────────────────────────────────────────────────────────────
base_df = df[
    (df["net"].isin(selected_nets))
    & (df["optimizer"].isin(selected_opts))
    & (df["fitness_type"].isin(selected_fit))
].copy()

if base_df.empty:
    st.warning("No hay datos para la combinación seleccionada.")
    st.stop()

# Datos para las curvas: una repetición concreta o el promedio sobre todas.
if selected_rep != "Todas (Promedio)":
    filtered_df = base_df[base_df["rep"] == selected_rep].copy()
else:
    groupby_cols = ["net", "optimizer", "fitness_type", "mode", "gen"]
    avg_df = base_df.groupby(groupby_cols).agg({
        selected_metric: "mean",
        "n_train_rules": "max",
    }).reset_index()
    # Marca una generación como "nueva regla" si cualquier repetición la añadió ahí.
    events_df = base_df.groupby(groupby_cols)["rule_added_after_gen"].sum().reset_index()
    events_df["rule_added_after_gen"] = events_df["rule_added_after_gen"] > 0
    filtered_df = pd.merge(avg_df, events_df, on=groupby_cols)

filtered_df["Experiment"] = (
    filtered_df["net"] + " - " + filtered_df["optimizer"] + " - "
    + filtered_df["fitness_type"] + " - " + filtered_df["mode"]
)

# ─── GRÁFICO DE CURVAS ──────────────────────────────────────────────────────────
fig = px.line(filtered_df, x="gen", y=selected_metric, color=color_col, line_group="Experiment",
              title=f"Evolución de {selected_metric} por Generación",
              labels={"gen": "Generación (gen)", selected_metric: selected_metric},
              markers=False)

rule_added_df = filtered_df[filtered_df["rule_added_after_gen"] == True]
if show_stars and not rule_added_df.empty:
    for exp in rule_added_df["Experiment"].unique():
        exp_df = rule_added_df[rule_added_df["Experiment"] == exp]
        fig.add_trace(go.Scatter(
            x=exp_df["gen"],
            y=exp_df[selected_metric],
            mode="markers",
            marker=dict(size=10, symbol="star", line=dict(width=1, color="DarkSlateGrey")),
            name=f"Nueva Regla ({exp})",
            hovertext="Nº reglas en pool: " + exp_df["n_train_rules"].astype(str),
        ))

st.plotly_chart(fig, use_container_width=True)

# ─── REGLAS NECESARIAS (resultado clave del estudio) ────────────────────────────
st.subheader("Reglas necesarias por corrida")
st.caption("Nº de reglas en el pool al terminar cada corrida (n_train_rules de la última generación).")

# Última generación de cada (net, optimizer, fitness, rep) → reglas necesarias.
needed = (base_df.sort_values("gen")
          .groupby(["net", "optimizer", "fitness_type", "rep"], as_index=False)
          .tail(1))

c1, c2, c3 = st.columns(3)
c1.metric("Reglas necesarias (media)", f"{needed['n_train_rules'].mean():.1f}")
c2.metric("Máximo", f"{needed['n_train_rules'].max():.0f}")
c3.metric("Mínimo", f"{needed['n_train_rules'].min():.0f}")

summary = (needed.groupby(["optimizer", "fitness_type"])["n_train_rules"]
           .agg(media="mean", desv="std", n="count").reset_index())
summary["desv"] = summary["desv"].fillna(0.0)

bar = px.bar(
    summary, x="optimizer", y="media", color="fitness_type", barmode="group",
    error_y="desv",
    title="Reglas necesarias (media ± σ) por optimizador y fitness",
    labels={"media": "Reglas necesarias (media)", "optimizer": "Optimizador",
            "fitness_type": "Función Fitness"},
)
st.plotly_chart(bar, use_container_width=True)

# ─── TABLAS ─────────────────────────────────────────────────────────────────────
with st.expander("Tabla — reglas necesarias (media por optimizador × fitness)"):
    pivot = summary.pivot(index="optimizer", columns="fitness_type", values="media").round(1)
    st.dataframe(pivot)

st.subheader("Datos tabulares (curvas filtradas)")
st.dataframe(filtered_df)
