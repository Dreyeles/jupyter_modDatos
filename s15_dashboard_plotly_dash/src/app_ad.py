from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import os

# --- 1. CARGA DE DATOS ---
# Intentamos cargar tu archivo local
file_path = '../datasets/advertising.xls'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print(f"Datos cargados exitosamente: {len(df)} registros.")
else:
    print("No encuentro 'data/advertising.xls'. Usando datos de ejemplo de internet...")

# --- 2. ASIGNACIÓN DE VARIABLES ---
# Seleccionamos las columnas EXACTAS que me mostraste
features = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']
X = df[features]
y = df['Clicked on Ad']

# EDA: Escalado + Regresión Logística
# (El escalado es vital porque 'Income' es 60,000 y 'Age' es 30)
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X, y)

# --- 3. DEFINIR APP ---
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG]) 
app.title = "Ad Click Simulator"

# --- 4. LAYOUT (Interfaz Gráfica) ---
app.layout = dbc.Container([
    html.H1("Simulador de Predicción de Clics", className="text-center my-4", style={'color': '#00df9a'}),
    html.P("Ajusta los parámetros del usuario para calcular su probabilidad de conversión.", className="text-center text-muted"),
    html.Hr(),

    dbc.Row([
        # --- COLUMNA IZQUIERDA: CONTROLES ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Perfil del Usuario", className="fw-bold"),
                dbc.CardBody([
                    
                    # 1. Edad
                    html.Label("Edad (Age)"),
                    dcc.Slider(id='slider-age', 
                               min=int(df['Age'].min()), max=int(df['Age'].max()), step=1, 
                               value=30, 
                               marks={20:'20', 30:'30', 40:'40', 50:'50', 60:'60'}),
                    html.Br(),

                    # 2. Ingresos
                    html.Label("Ingresos de la Zona (Area Income)"),
                    dcc.Slider(id='slider-income', 
                               min=20000, max=80000, step=1000, 
                               value=60000,
                               marks={20000:'$20k', 30000:'$30k', 50000:'$50k', 60000:'$60k', 80000:'$80k'}),
                    html.Br(),

                    # 3. Tiempo en Sitio
                    html.Label("Tiempo Diario en el Sitio (Minutos)"),
                    dcc.Slider(id='slider-time', 
                               min=30, max=90, step=1, 
                               value=65,
                               marks={30:'30m', 60:'60m', 90:'90m'}),
                    html.Br(),

                    # 4. Uso de Internet
                    html.Label("Uso Diario de Internet (Minutos)"),
                    dcc.Slider(id='slider-internet', 
                               min=100, max=275, step=5, 
                               value=200,
                               marks={100:'100m', 200:'200m', 275:'275m'}),
                ])
            ], color="ligth", style={"background-color": "#0b1220", "border": "1px solid #444"})
        ], md=4),

        # --- COLUMNA DERECHA: RESULTADO (VELOCÍMETRO) ---
        dbc.Col([
            dbc.Card(
                dbc.CardBody(
                    dcc.Graph(id='gauge-chart', style={'height': '400px'}),
                    style={"padding": "1rem"}
                ),
                style={
                    "backgroundColor": "#0b1220",   # fondo oscuro personalizado
                    "color": "#00df9a",             # texto verde neón
                    "border": "1px solid #123033",  # borde sutil en tono verde-azulado
                    "borderRadius": "6px"
                }
            )
        ], md=8)
    ])
], fluid=True)

# --- 5. CALLBACKS (Lógica) ---
@app.callback(
    Output('gauge-chart', 'figure'),
    [
        Input('slider-time', 'value'),
        Input('slider-age', 'value'),
        Input('slider-income', 'value'),
        Input('slider-internet', 'value')
    ]
)
def update_gauge(time, age, income, internet):
    # 1. Crear dataframe con los datos del slider
    input_data = pd.DataFrame([[time, age, income, internet]], columns=features)
    
    # 2. Predecir Probabilidad (Clase 1 = Click)
    prob = model.predict_proba(input_data)[0][1] * 100
    
    # 3. Determinar color según riesgo
    color_bar = "red" if prob < 40 else "yellow" if prob < 70 else "#00df9a" # Verde neon

    # 4. Crear gráfico
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob,
        title = {'text': "Probabilidad de Clic"},
        number = {'suffix': "%"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color_bar},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 100], 'color': 'rgba(50, 50, 50, 0.5)'}
            ],
        }
    ))

    # Estilo oscuro para el gráfico
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    
    return fig

# --- 6. EJECUTAR ---
if __name__ == '__main__':
    app.run(debug=True, port=8051)