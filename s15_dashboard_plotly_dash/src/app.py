from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#### DATA

inmob_raw = pd.read_csv('../data/housing.csv')

inmob = pd.read_csv('../data/data_procesada.csv', index_col=0)

### PREPROCESAMIENTO DE DATOS



#### DEFINIR APP

app = Dash(__name__,  external_stylesheets=[dbc.themes.DARKLY])
app.title = "ML Dashboard"
server = app.server

#### CONTROLES

controls = [
    dbc.Row(
        html.H5('Seleccione las caracteristicas del distrito:')
    ),
    html.Br(),
    dbc.Row(
        [
            dbc.Label('Numero de personas'),
            dcc.RangeSlider(
                id='poblacion',
                min=0,
                max=10000,
                step=1000,
                value=[0, 3000]
            ),
        ]
    ),
    dbc.Row(
        [
            dbc.Label('Antiguedad media de los inmuebles (anos)'),
            dcc.RangeSlider(
                id='antiguedad',
                min=10,
                max=50,
                step=10,
                value=[20, 40],
            ),
        ]
    ),
    dbc.Row(
        dbc.Button("Entrenar modelo OLS", color="primary", id="button-train", n_clicks=0), style={"margin-top": "20px"}
    )
]

#### LAYOUT

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H2(
            "Modelamiento de precios inmobiliarios",
            style={"margin-top": 10}
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(html.P("ANtes del preprocesamiento"), md=1, style={"text-align": "center", "margin-top": "20px"}),
                dbc.Col(dcc.Graph(id="avp-graph2", style={"height": "500px"}), md=12),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Card(controls, body=True),md=3),
                dbc.Col(dcc.Graph(id="avp-graph", style={"height": "500px"}), md=4),
                dbc.Col(dcc.Graph(id="coef-graph", style={"height": "600px"}), md=4),
                dbc.Col(html.P("Por Rafael"), md=1, style={"text-align": "center", "margin-top": "20px"}),
            ], style={"margin-top": "30px"}
        ),
    ],
    style={"margin": "auto"},
)

#### CALLBACKS

@app.callback(
    [
        Output(component_id="avp-graph", component_property="figure"),
        Output(component_id="coef-graph", component_property="figure"),
        Output(component_id="avp-graph2", component_property="figure"),
    ],
    [
        Input(component_id="button-train", component_property="n_clicks")
    ],
    [
        State(component_id="poblacion", component_property="value"),
        State(component_id="antiguedad", component_property="value"),
    ],
)
def sample_and_train(input1, state1, state2):

    # Filtrps datos por los controles
    cond1 = (inmob['population'] >= state1[0]) & (inmob['population'] < state1[1])
    cond2 = (inmob['housing_median_age'] >= state2[0]) & (inmob['housing_median_age'] < state2[1])
    conds = cond1 & cond2
    df = inmob.loc[conds, :]

    # Muestreo de entrenamiento y evaluacion
    X = df.drop(columns=['median_house_value', 'latitude', 'longitude'])
    y = df.loc[:, 'median_house_value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Calcular metrica de performance: RMSE
    y_pred = model.predict(X_test)
    test_rmse = metrics.mean_squared_error(y_test, y_pred) ** 0.5

    # OUTPUT: Figura de coeficientes
    coef_fig = px.bar(
        y=X_test.columns,
        x=model.coef_,
        labels={"x": "Peso en la prediccion", "y": "Variables"},
        title="Importancia de las variables en la prediccion del precio",
    )

    # OUTPUT: Figura de valores reales vs predichos
    avp_fig = px.scatter(
        x=y_test,
        y=y_pred,
        labels={"x": "Real", "y": "Predicho"},
        title=f"Precio real vs predicho (RMSE={test_rmse:.2f})",
    )

    avp_fig.add_shape(
        type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max()
    )

    avp_fig2 = avp_fig

    return avp_fig, coef_fig, avp_fig2

if __name__ == "__main__":
    app.run(debug=True)
