from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. PREPARACIÓN DE DATOS ---
df = pd.read_csv('../datasets/advertising.xls')


# Convertimos la variable objetivo a texto para que los gráficos sean legibles
df['Click_Label'] = df['Clicked on Ad'].map({0: 'No Click', 1: 'Sí Click'})

# Entrenamos el modelo (Logistic Regression es ideal para probabilidades)
X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
y = df['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# --- 2. GRÁFICOS DE LA HISTORIA ---

# A) El Perfil Demográfico (Scatter: Edad vs Ingresos)
# Este gráfico cuenta la historia de "Quiénes son"
fig_scatter = px.scatter(
    df, x="Age", y="Area Income", color="Click_Label",
    title="Perfil Demográfico: Edad vs. Ingresos",
    color_discrete_map={'No Click': '#bdc3c7', 'Sí Click': '#e74c3c'}, # Rojo para el target
    opacity=0.7
)
fig_scatter.update_layout(plot_bgcolor="white")

# B) El Comportamiento (Boxplots)
# Este gráfico cuenta la historia de "Qué hacen"
fig_box = px.box(
    df, x="Click_Label", y="Daily Internet Usage", color="Click_Label",
    title="Comportamiento Digital: ¿Cuánto navegan?",
    color_discrete_map={'No Click': '#bdc3c7', 'Sí Click': '#e74c3c'}
)
fig_box.update_layout(plot_bgcolor="white", showlegend=False)

# --- 3. LAYOUT (ESTILO SCROLLYTELLING) ---
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Ad Campaign Report"

app.layout = dbc.Container([
    
    # --- SECCIÓN 1: PORTADA ---
    dbc.Row([
        dbc.Col([
            html.H1("Optimización de Campañas Digitales", className="display-4 text-center mt-5"),
            html.P("Un análisis forense de por qué perdemos dinero en publicidad.", className="lead text-center text-muted"),
            html.Hr(className="my-4"),
        ], width=12)
    ]),

    # --- SECCIÓN 2: EL DIAGNÓSTICO (Texto + Scatter) ---
    dbc.Row([
        # Texto Narrativo (Izquierda)
        dbc.Col([
            html.H3("1. El Diagnóstico", className="text-primary"),
            html.P("Analizamos 1,000 interacciones de usuarios. El gráfico de la derecha revela una verdad incómoda sobre nuestro target."),
            dbc.Alert([
                html.H5("El Cluster Rojo (Clics)", className="alert-heading"),
                html.P("Nuestros usuarios convertidores son mayores de 40 años y tienen menores ingresos."),
                html.Hr(),
                html.P("Los jóvenes con altos ingresos (puntos grises arriba a la izquierda) nos ignoran completamente.", className="mb-0")
            ], color="danger"),
        ], md=4, className="d-flex flex-column justify-content-center"),
        
        # Gráfico (Derecha)
        dbc.Col([
            dcc.Graph(figure=fig_scatter, style={'height': '500px'})
        ], md=8),
    ], className="mb-5"), # Margen inferior para separar secciones

    # --- SECCIÓN 3: EL COMPORTAMIENTO (Texto + Boxplot) ---
    dbc.Row([
        # Gráfico (Izquierda)
        dbc.Col([
            dcc.Graph(figure=fig_box, style={'height': '400px'})
        ], md=7),
        
        # Texto Narrativo (Derecha)
        dbc.Col([
            html.H3("2. La Fatiga Digital", className="text-primary"),
            html.P("¿Por qué los usuarios intensivos de internet no hacen clic?"),
            html.Ul([
                html.Li("Usuarios que NO hacen clic: Navegan > 200 min/día."),
                html.Li("Usuarios que SÍ hacen clic: Navegan < 150 min/día."),
            ]),
            dbc.Alert("Insight: La 'ceguera de banner' es real. A más uso de internet, menor probabilidad de clic.", color="secondary"),
        ], md=5, className="d-flex flex-column justify-content-center"),
    ], className="mb-5"),

    # --- SECCIÓN 4: LA SOLUCIÓN (CALCULADORA) ---
    dbc.Card([
        dbc.CardHeader(html.H3("3. Motor de Predicción (IA)", className="text-center m-0 text-white"), className="bg-dark"),
        dbc.CardBody([
            dbc.Row([
                # Controles
                dbc.Col([
                    html.Label("Tiempo en Sitio (Min)"),
                    dcc.Slider(id='time-slider', min=30, max=90, value=65, marks={30:'30', 90:'90'}),
                    
                    html.Label("Edad (Años)", className="mt-3"),
                    dcc.Slider(id='age-slider', min=19, max=60, value=30, marks={20:'20', 60:'60'}),
                    
                    html.Label("Ingresos Anuales ($)", className="mt-3"),
                    dcc.Slider(id='income-slider', min=20000, max=80000, value=55000, step=5000, marks={20000:'20k', 80000:'80k'}),
                    
                    html.Label("Uso de Internet (Min)", className="mt-3"),
                    dcc.Slider(id='internet-slider', min=100, max=275, value=180, marks={100:'100', 275:'275'}),
                ], md=5, className="p-4"),

                # Resultado (Velocímetro)
                dbc.Col([
                    dcc.Graph(id='gauge-chart')
                ], md=7)
            ])
        ])
    ], className="shadow-lg mb-5"),

], fluid=True)

# --- 4. CALLBACK ---
@app.callback(
    Output('gauge-chart', 'figure'),
    [Input('time-slider', 'value'),
    Input('age-slider', 'value'),
    Input('income-slider', 'value'),
    Input('internet-slider', 'value')]
)
def update_prediction(time, age, income, internet):
    # Predicción
    input_data = pd.DataFrame([[time, age, income, internet]], columns=X.columns)
    prob = model.predict_proba(input_data)[0][1] * 100 # Probabilidad de Clase 1
    
    # Color dinámico
    color = "red" if prob < 50 else "green"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob,
        title = {'text': "Probabilidad de Conversión"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "#ecf0f1"},
                {'range': [50, 100], 'color': "#d5f5e3"}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': prob}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8061)