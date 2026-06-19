import streamlit as st
import pandas as pd
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
    return pd.read_csv(csv_path)

df = load_data()

st.sidebar.header("Filtros")

# Selectors
nets = df['net'].unique().tolist()
selected_nets = st.sidebar.multiselect("Red (net)", nets, default=nets)

optimizers = df['optimizer'].unique().tolist()
selected_opts = st.sidebar.multiselect("Optimizador", optimizers, default=optimizers)

fitness_types = df['fitness_type'].unique().tolist()
selected_fit = st.sidebar.multiselect("Función Fitness", fitness_types, default=fitness_types)

reps = ["Todas (Promedio)"] + sorted(df['rep'].unique().tolist())
selected_rep = st.sidebar.selectbox("Repetición (Seed)", reps)

# Elegir la métrica principal
metrics = ['max_accuracy', 'mean_accuracy', 'pct_success_indv', 'n_train_rules', 'gen_cpu_time']
selected_metric = st.sidebar.selectbox("Métrica Eje Y", metrics, index=0)

# Característica por la que agrupar/colorear las curvas
color_options = {
    "Experimento (net+opt+fitness+modo)": "Experiment",
    "Modo": "mode",
    "Función Fitness": "fitness_type",
    "Optimizador": "optimizer",
    "Red (net)": "net",
}
color_label = st.sidebar.selectbox("Colorear curvas por", list(color_options.keys()))
color_col = color_options[color_label]

# Filtrado básico
filtered_df = df[
    (df['net'].isin(selected_nets)) &
    (df['optimizer'].isin(selected_opts)) &
    (df['fitness_type'].isin(selected_fit))
].copy()

if selected_rep != "Todas (Promedio)":
    filtered_df = filtered_df[filtered_df['rep'] == selected_rep]
else:
    # Si se selecciona "Promedio", agrupamos por todas las combinaciones y generación
    groupby_cols = ['net', 'optimizer', 'fitness_type', 'mode', 'gen']
    
    # Calcular promedios
    avg_df = filtered_df.groupby(groupby_cols).agg({
        selected_metric: 'mean',
        'n_train_rules': 'max',
    }).reset_index()
    
    # Para visualizar "dónde se añadieron reglas", cogeremos los puntos donde "cualquier" rep añadió una regla,
    # aunque promediando pierde un poco de precisión exacta en qué gen se añadió, da una idea visual.
    # Alternativamente, podemos contar cuántas repeticiones añadieron regla en cada gen
    events_df = filtered_df.groupby(groupby_cols)['rule_added_after_gen'].sum().reset_index()
    events_df['rule_added_after_gen'] = events_df['rule_added_after_gen'] > 0
    
    filtered_df = pd.merge(avg_df, events_df, on=groupby_cols)

if filtered_df.empty:
    st.warning("No hay datos para la combinación seleccionada.")
    st.stop()

# Crear columna identificadora de cada curva (net+opt+fitness+modo)
filtered_df['Experiment'] = (
    filtered_df['net'] + " - " + filtered_df['optimizer'] + " - "
    + filtered_df['fitness_type'] + " - " + filtered_df['mode']
)

# Pintar gráfico principal de líneas. El color se asigna según la característica
# elegida en la barra lateral; line_group mantiene cada curva separada aunque
# varias compartan color.
fig = px.line(filtered_df, x='gen', y=selected_metric, color=color_col, line_group='Experiment',
              title=f"Evolución de {selected_metric} por Generación",
              labels={'gen': 'Generación (gen)', selected_metric: selected_metric},
              markers=False)

# Añadir marcadores donde se añaden reglas (puntos gordos / estrellas)
rule_added_df = filtered_df[filtered_df['rule_added_after_gen'] == True]

if not rule_added_df.empty:
    for exp in rule_added_df['Experiment'].unique():
        exp_df = rule_added_df[rule_added_df['Experiment'] == exp]
        fig.add_trace(go.Scatter(
            x=exp_df['gen'], 
            y=exp_df[selected_metric],
            mode='markers',
            marker=dict(size=10, symbol='star', line=dict(width=1, color='DarkSlateGrey')),
            name=f'Nueva Regla ({exp})',
            hovertext="Nº reglas en pool: " + exp_df['n_train_rules'].astype(str)
        ))

st.plotly_chart(fig, use_container_width=True)

st.subheader("Datos tabulares (filtrados)")
st.dataframe(filtered_df)
