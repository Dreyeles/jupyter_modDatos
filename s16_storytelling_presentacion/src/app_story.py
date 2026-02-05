from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- 1. CARGA DE DATOS ---
try:
    df = pd.read_csv('../datasets/housing.csv')
except:
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)

df = df.dropna()
df = df[df['median_house_value'] < 500000] 

# --- 2. ENTRENAMIENTO GLOBAL (Para gráficos de Validación) ---
# Entrenamos una vez al inicio para generar la gráfica de "Real vs Predicho"
X_global = df[['median_income', 'housing_median_age', 'total_rooms']]
y_global = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X_global, y_global, test_size=0.2, random_state=42)

model_global = LinearRegression().fit(X_train, y_train)
y_pred_test = model_global.predict(X_test)
score_r2 = r2_score(y_test, y_pred_test)

# --- 3. GRÁFICOS ESTÁTICOS (Contexto y Factores) ---

# TAB 1: Mapa Geográfico
fig_map = px.scatter_map(
    df, lat="latitude", lon="longitude", color="median_house_value",
    size="population", color_continuous_scale=px.colors.cyclical.IceFire,
    size_max=15, zoom=4, map_style="carto-positron", title="Mapa de Valor"
)
fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

# TAB 2: Correlaciones (Matemática) vs Boxplot (Realidad)
corr_matrix = df.corr(numeric_only=True)
corr_target = corr_matrix['median_house_value'].sort_values(ascending=False).drop('median_house_value')
fig_corr = px.bar(x=corr_target.values, y=corr_target.index, orientation='h', title="Correlación Lineal")

fig_box = px.box(df, x="median_house_value", y="ocean_proximity", color="ocean_proximity",
                 title="Impacto de la Ubicación", points=False)
fig_box.update_layout(showlegend=False)

# TAB 4: Validación (La prueba de confianza)
fig_validation = px.scatter(x=y_test, y=y_pred_test, 
                            labels={'x': 'Valor Real', 'y': 'Valor Predicho'},
                            title=f"Precisión del Modelo (R² = {score_r2:.2f})", opacity=0.5)
fig_validation.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="red", dash="dash"))

# --- 4. APP LAYOUT (Narrativa) ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Housing Story"

app.layout = dbc.Container([
    # Encabezado
    dbc.Row([
        dbc.Col(html.H1("California Housing: Historia de una Inversión", className="text-center text-primary my-4"), width=12),
        dbc.Col(html.P("Del caos de los datos a la claridad de la predicción.", className="text-center text-muted"), width=12),
    ]),
    html.Hr(),

    # Sistema de Pestañas (Capítulos de la historia)
    dbc.Tabs([
        # CAPÍTULO 1: CONTEXTO
        dbc.Tab(label="1. El Contexto (Mapa)", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("¿Dónde está el dinero?", className="mt-3"),
                    html.P("El mercado es geográficamente desigual. Las zonas costeras duplican el valor de las zonas interiores."),
                    dbc.Alert("Insight: La ubicación es el factor multiplicador.", color="primary"),
                ], md=4),
                dbc.Col(dcc.Graph(figure=fig_map, style={'height': '550px'}), md=8),
            ], className="mt-4")
        ]),

        # CAPÍTULO 2: FACTORES
        dbc.Tab(label="2. Los Factores (Drivers)", children=[
            dbc.Row([
                dbc.Col([html.H5("Lo que dicen las Matemáticas"), dcc.Graph(figure=fig_corr, style={'height': '400px'})], md=6),
                dbc.Col([html.H5("Lo que dice el Mercado"), dcc.Graph(figure=fig_box, style={'height': '400px'})], md=6),
            ], className="mt-4"),
            dbc.Alert("Conclusión: Combinar Ingreso (Income) + Ubicación (Ocean) es clave.", color="warning", className="mt-3")
        ]),

        # CAPÍTULO 3: SIMULADOR (INTERACTIVO)
        dbc.Tab(label="3. Simulador (IA)", children=[
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Simulador de Tasación"),
                        html.Label("1. Ingreso del Vecindario ($)"),
                        dcc.Slider(id='slider-income', min=1, max=15, step=0.5, value=5, marks={2:'20k', 15:'150k'}),
                        html.Label("2. Antigüedad (Años)", className="mt-3"),
                        dcc.Slider(id='slider-age', min=1, max=52, step=1, value=20, marks={1:'New', 52:'Old'}),
                        html.Label("3. Habitaciones Promedio", className="mt-3"),
                        dcc.Slider(id='slider-rooms', min=1, max=10, step=0.5, value=5, marks={1:'1', 10:'10'}),
                        dbc.Button("Simular", id='btn-predict', color="success", className="w-100 mt-4")
                    ], className="p-4 bg-light rounded border")
                ], md=4),
                dbc.Col(html.Div(id='prediction-output'), md=8),
            ], className="mt-4")
        ]),

        # CAPÍTULO 4: VALIDACIÓN
        dbc.Tab(label="4. Validación (Confianza)", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("¿Por qué confiar?", className="mt-3"),
                    html.P("Sometimos la IA a una prueba ciega. La línea roja representa la perfección."),
                    dbc.Alert(f"El modelo tiene una solidez del {score_r2:.0%} (R²).", color="success"),
                ], md=4),
                dbc.Col(dcc.Graph(figure=fig_validation, style={'height': '500px'}), md=8),
            ], className="mt-4")
        ]),
    ])
], fluid=True)

# --- 5. CALLBACK (CEREBRO DINÁMICO) ---
@app.callback(
    Output('prediction-output', 'children'),
    Input('btn-predict', 'n_clicks'),
    [State('slider-income', 'value'), State('slider-age', 'value'), State('slider-rooms', 'value')]
)
def predecir(n_clicks, income, age, rooms):
    if not n_clicks: return html.H3("👈 Configura la simulación", className="text-center text-muted mt-5")
    
    # Entrenar modelo en vivo para la demo (Contexto dinámico)
    X, y = df[['median_income', 'housing_median_age', 'total_rooms']], df['median_house_value']
    model = LinearRegression().fit(X, y)
    
    # Predecir
    pred = model.predict(pd.DataFrame([[income, age, rooms*400]], columns=X.columns))[0]
    pred = max(pred, 0)

    # Gráfico A: Velocímetro (Gauge)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=pred, title={'text': "Tasación"}, number={'prefix': "$"},
        gauge={'axis': {'range': [0, 500000]}, 'bar': {'color': "#2c3e50"},
               'steps': [{'range': [0, 150000], 'color': "#2ecc71"}, 
                         {'range': [150000, 300000], 'color': "#f1c40f"}, 
                         {'range': [300000, 500000], 'color': "#e74c3c"}]}
    ))
    fig_gauge.update_layout(height=250, margin=dict(t=30, b=20, l=20, r=20))

    # Gráfico B: Histograma de Mercado
    fig_hist = px.histogram(df, x="median_house_value", nbins=40, title="Posición en Mercado", color_discrete_sequence=['#95a5a6'])
    fig_hist.add_vline(x=pred, line_color="#e74c3c", line_dash="dash", line_width=3)
    fig_hist.update_layout(height=250, margin=dict(t=30, b=20), showlegend=False)

    return html.Div([
        html.H2(f"${pred:,.2f}", className="text-center text-primary"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_gauge), md=6), dbc.Col(dcc.Graph(figure=fig_hist), md=6)]),
        dbc.Alert(f"Predicción basada en Ingreso: ${income}0k y Antigüedad: {age} años.", color="info", className="mt-2")
    ])

if __name__ == '__main__':
    app.run(debug=True, port=8060)