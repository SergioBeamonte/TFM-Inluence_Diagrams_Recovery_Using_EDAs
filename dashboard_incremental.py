import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Dashboard Curvas Incrementales", layout="wide")

st.title("Visualización de Curvas Incrementales (EDA)")

csv_path = r"example\explore_incremental.csv"

if not os.path.exists(csv_path):
    st.error(f"No se encuentra el archivo: {csv_path}. Asegúrate de ejecutar explore_incremental.py primero.")
    st.stop()


@st.cache_data
def load_data():
    df = pd.read_csv(csv_path)
    # rule_added_after_gen puede venir como bool o como texto "True"/"False".
    df["rule_added_after_gen"] = (
        df["rule_added_after_gen"].astype(str).str.strip().str.lower().isin(["true", "1"])
    )
    return df


df = load_data()

# ─── PALETA Y UTILIDADES DE COLOR ───────────────────────────────────────────────
PALETTE = px.colors.qualitative.D3


def _hex_to_rgba(color: str, alpha: float) -> str:
    """Convierte '#rrggbb' o 'rgb(...)' a 'rgba(r,g,b,alpha)'."""
    if color.startswith("rgb"):
        nums = color[color.find("(") + 1:color.find(")")].split(",")[:3]
        r, g, b = (int(float(n)) for n in nums)
    else:
        c = color.lstrip("#")
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─── SIDEBAR / FILTROS ──────────────────────────────────────────────────────────
st.sidebar.header("Filtros")

FILTER_COLS = {
    "net": "Red (net)",
    "optimizer": "Optimizador",
    "fitness_type": "Función Fitness",
    "size_gen": "Tamaño de Generación",
    "mode": "Modo",
}

selected_filters = {}
for col, label in FILTER_COLS.items():
    if col not in df.columns:
        continue
    opts = sorted(df[col].dropna().unique().tolist())
    selected_filters[col] = st.sidebar.multiselect(label, opts, default=opts)

# Métrica del eje Y
metrics = ["max_accuracy", "mean_accuracy", "pct_success_indv", "n_train_rules", "gen_cpu_time"]
metrics = [m for m in metrics if m in df.columns]
selected_metric = st.sidebar.selectbox("Métrica Eje Y", metrics, index=0)

st.sidebar.divider()

# Agrupación: qué columnas definen una "curva"/color (el experimento).
_group_candidates = [c for c in FILTER_COLS if c in df.columns]
_default_group = [c for c in ["net", "optimizer", "fitness_type"] if c in _group_candidates]
group_cols = st.sidebar.multiselect(
    "Definir experimento por (color)",
    _group_candidates,
    default=_default_group,
    help="Elige una o varias características. Cada combinación distinta es un color/curva. "
         "Selecciona una sola para pintar todas las curvas en función de esa característica.",
)
if not group_cols:
    st.sidebar.warning("Selecciona al menos una característica para agrupar.")
    group_cols = _default_group

st.sidebar.divider()

# Modo de visualización
view_mode = st.sidebar.radio(
    "Modo de visualización",
    ["Promedio ± dispersión", "Todas las curvas + promedio", "Repetición concreta"],
)

selected_rep = None
if view_mode == "Repetición concreta":
    reps = sorted(df["rep"].dropna().unique().tolist())
    selected_rep = st.sidebar.selectbox("Repetición (Seed)", reps)

# ─── FILTRADO ───────────────────────────────────────────────────────────────────
mask = pd.Series(True, index=df.index)
for col, sel in selected_filters.items():
    mask &= df[col].isin(sel)
filtered_df = df[mask].copy()

if view_mode == "Repetición concreta":
    filtered_df = filtered_df[filtered_df["rep"] == selected_rep]

if filtered_df.empty:
    st.warning("No hay datos para la combinación seleccionada.")
    st.stop()

# Etiqueta de grupo (color) y de curva individual (rep + resto de características).
filtered_df["__group"] = filtered_df[group_cols].astype(str).agg(" · ".join, axis=1)
_id_cols = [c for c in ["net", "optimizer", "fitness_type", "size_gen", "mode", "rep"]
            if c in filtered_df.columns]
filtered_df["__curve"] = filtered_df[_id_cols].astype(str).agg(" | ".join, axis=1)

groups = sorted(filtered_df["__group"].unique())
color_map = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}

# ─── GRÁFICO ────────────────────────────────────────────────────────────────────
fig = go.Figure()


def _add_stars(curve_df, base_color, legendgroup, name):
    """Marca con estrellas las generaciones donde se añadió una regla.
    Las estrellas usan el mismo color que la curva."""
    ev = curve_df[curve_df["rule_added_after_gen"]]
    if ev.empty:
        return
    fig.add_trace(go.Scatter(
        x=ev["gen"],
        y=ev[selected_metric],
        mode="markers",
        marker=dict(size=11, symbol="star", color=base_color,
                    line=dict(width=1, color=base_color)),
        name=name,
        legendgroup=legendgroup,
        showlegend=False,
        hovertext="Nº reglas en pool: " + ev["n_train_rules"].astype(str),
        hoverinfo="text+x+y",
    ))


for g in groups:
    base = color_map[g]
    g_df = filtered_df[filtered_df["__group"] == g]

    if view_mode == "Promedio ± dispersión":
        # Media y σ entre todas las curvas (reps · resto de características) por generación.
        agg = (g_df.groupby("gen")[selected_metric]
               .agg(["mean", "std"]).reset_index().sort_values("gen"))
        agg["std"] = agg["std"].fillna(0.0)
        upper = agg["mean"] + agg["std"]
        lower = agg["mean"] - agg["std"]

        # Banda de dispersión (±1σ) — sin estrella.
        fig.add_trace(go.Scatter(
            x=agg["gen"], y=upper, mode="lines", line=dict(width=0),
            legendgroup=g, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=agg["gen"], y=lower, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=_hex_to_rgba(base, 0.15),
            legendgroup=g, showlegend=False, hoverinfo="skip",
        ))
        # Curva promedio.
        fig.add_trace(go.Scatter(
            x=agg["gen"], y=agg["mean"], mode="lines",
            line=dict(color=base, width=3), name=g, legendgroup=g,
        ))

    elif view_mode == "Todas las curvas + promedio":
        # Todas las curvas individuales en el mismo color, más claro.
        light = _hex_to_rgba(base, 0.30)
        for cid, c_df in g_df.groupby("__curve"):
            c_df = c_df.sort_values("gen")
            fig.add_trace(go.Scatter(
                x=c_df["gen"], y=c_df[selected_metric], mode="lines",
                line=dict(color=light, width=1),
                name=g, legendgroup=g, showlegend=False,
                hovertext=cid, hoverinfo="text+x+y",
            ))
            _add_stars(c_df, base, g, f"Nueva regla ({cid})")
        # Promedio por encima, en color pleno (sin estrella).
        agg = g_df.groupby("gen")[selected_metric].mean().reset_index().sort_values("gen")
        fig.add_trace(go.Scatter(
            x=agg["gen"], y=agg[selected_metric], mode="lines",
            line=dict(color=base, width=3), name=f"{g} (promedio)", legendgroup=g,
        ))

    else:  # Repetición concreta
        for cid, c_df in g_df.groupby("__curve"):
            c_df = c_df.sort_values("gen")
            fig.add_trace(go.Scatter(
                x=c_df["gen"], y=c_df[selected_metric], mode="lines",
                line=dict(color=base, width=2.5), name=g, legendgroup=g,
            ))
            _add_stars(c_df, base, g, f"Nueva regla ({cid})")

fig.update_layout(
    title=f"Evolución de {selected_metric} por Generación",
    xaxis_title="Generación (gen)",
    yaxis_title=selected_metric,
    hovermode="closest",
    legend_title="Experimento",
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Datos tabulares (filtrados)")
st.dataframe(filtered_df.drop(columns=["__group", "__curve"]))
