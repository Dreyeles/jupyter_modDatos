from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import os

# --- 1. CARGA DE DATOS ---
file_path_train = '../datasets/train.csv'
file_path_test = '../datasets/test.csv'

df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)


# --- 2. ASIGNACIÓN DE VARIABLES ---
# Seleccionamos las columnas EXACTAS que me mostraste
features = ['fifa_rank_pre_tournament', 'fifa_points_pre_tournament', 
            'squad_total_market_value_eur', 'goals_scored_last_4y', 
            'wins_last_4y', 'squad_avg_age']

X = df_train[features].fillna(df_train[features].median())
y = df_train['quarter_finalist']


# 2. División de datos
X_train = X
y_train = y

X_test = df_test[features].fillna(df_train[features].median())
y_test = df_test['quarter_finalist']


# EDA: Escalado + Regresión Logística
# (El escalado es vital porque 'Income' es 60,000 y 'Age' es 30)
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X, y)


# --- 3. DEFINIR APP ---
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG]) 
app.title = "Ad Click Simulator"


# --- 4. LAYOUT (Interfaz Gráfica) ---


# --- 5. CALLBACKS (Lógica) ---



# --- 6. EJECUTAR ---
if __name__ == '__main__':
    app.run(debug=True, port=8050)