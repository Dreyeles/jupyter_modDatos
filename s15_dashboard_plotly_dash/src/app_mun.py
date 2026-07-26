from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- 1. CARGA DE DATOS ---
file_path_train = '../datasets/train.csv'
file_path_test = '../datasets/test.csv'

df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)

# --- 2. PREPARACIÓN DEL MODELO ---
features = ['fifa_rank_pre_tournament', 'fifa_points_pre_tournament', 
            'squad_total_market_value_eur', 'goals_scored_last_4y', 
            'wins_last_4y', 'squad_avg_age']

# Llenamos nulos con la mediana para evitar errores
X_train = df_train[features].fillna(df_train[features].median())
# Predecimos al CAMPEÓN ('winner')
y_train = df_train['winner'].fillna(0)

# Usamos class_weight='balanced' porque hay muy pocos campeones
model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', random_state=42))
model.fit(X_train, y_train)

# Obtenemos la lista de equipos del Mundial 2026 ordenados alfabéticamente
equipos_2026 = df_test['team'].dropna().sort_values().unique()

# --- 3. DEFINIR APP ---
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY]) 
app.title = "Predicción de Campeón 2026"

# --- 4. LAYOUT (Interfaz Gráfica) ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("🏆 Simulador Oráculo: ¿Quién ganará el Mundial 2026?", 
                        className="text-center text-warning mt-4 mb-4"), width=12)
    ]),
    
    dbc.Row([
        # Columna Izquierda: Los Controles (Inputs)
        dbc.Col([
            html.H4("⚽ Selecciona a tu Equipo"),
            html.Hr(),
            
            # Selector de Selección Nacional
            html.Label("Selección Nacional:"),
            dcc.Dropdown(
                id='in-equipo',
                options=[{'label': equipo, 'value': equipo} for equipo in equipos_2026],
                value='Argentina', # Valor por defecto
                style={'color': 'black'} # Para que el texto sea visible en fondo oscuro
            ),
            
            html.Br(),
            html.Br(),
            
            # El Parámetro Esencial
            html.H5("⚡ El Parámetro Esencial"),
            html.P("Simula una racha de victorias o derrotas alterando sus Puntos FIFA antes del Mundial:", 
                className="", style={"fontSize": "14px", 'color': '#fff'}),
            
            dcc.Slider(
                id='in-puntos-extra', 
                min=-300, max=300, step=10, value=0, 
                marks={-300: '-300 pts', 0: 'Actual', 300: '+300 pts'},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
        ], width=5, className="p-4 bg-dark rounded shadow", style={'background-color': '#ccc', 'border': '1px solid #444'}),
        
        # Columna Derecha: El Gráfico (Output)
        dbc.Col([
            dcc.Graph(id='grafico-gauge', style={'height': '100%', 'min-height': '450px'})
        ], width=7, className="d-flex align-items-center justify-content-center")
    ])
], fluid=True)


# --- 5. CALLBACKS (Lógica) ---
@app.callback(
    Output('grafico-gauge', 'figure'),
    [
        Input('in-equipo', 'value'),
        Input('in-puntos-extra', 'value')
    ]
)
def actualizar_simulador(equipo_seleccionado, puntos_extra):
    # 1. Buscamos los datos reales del equipo seleccionado en test.csv
    datos_equipo = df_test[df_test['team'] == equipo_seleccionado].iloc[0]
    
    # 2. Sumamos los puntos extra al parámetro esencial
    puntos_simulados = datos_equipo['fifa_points_pre_tournament'] + puntos_extra

    # 3. Creamos el DataFrame de entrada para el modelo con la data real + la simulación
    df_input = pd.DataFrame({
        'fifa_rank_pre_tournament': [datos_equipo['fifa_rank_pre_tournament']],
        'fifa_points_pre_tournament': [puntos_simulados],
        'squad_total_market_value_eur': [datos_equipo['squad_total_market_value_eur']],
        'goals_scored_last_4y': [datos_equipo['goals_scored_last_4y']],
        'wins_last_4y': [datos_equipo['wins_last_4y']],
        'squad_avg_age': [datos_equipo['squad_avg_age']]
    })

    # 4. Predecir la probabilidad de GANAR EL MUNDIAL (Clase 1)
    probabilidad = model.predict_proba(df_input)[0][1] * 100

    # 5. Crear el gráfico de velocímetro
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probabilidad,
        number={'suffix': "%", 'valueformat': ".1f"},
        title={'text': f"Posibilidad de que {equipo_seleccionado}<br>Gane el Mundial", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#FFD700"}, # Color Oro para el campeón
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': "#8b0000"},     # Rojo (Casi imposible)
                {'range': [20, 60], 'color': "#ff8c00"},    # Naranja (Posible)
                {'range': [60, 100], 'color': "#006400"}    # Verde (Candidato serio)
            ]
        }
    ))
    
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    
    return fig

# --- 6. EJECUTAR ---
if __name__ == '__main__':
    app.run(debug=True, port=8050)