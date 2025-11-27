import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# --- 1. Cargar y Preparar Datos ---
try:
    df = pd.read_csv('../datasets/advertising.xls')
    print("Archivo 'advertising.xls' cargado correctamente.")
except FileNotFoundError:
    print("Error: No se encontró 'advertising.xls'. Asegúrate de que esté en la misma carpeta.")
    exit()

# Seleccionamos las 2 variables principales
feature_cols = ['Age', 'Male']
target_col = 'Clicked on Ad'

X = df[feature_cols].values
y = df[target_col].values

# Dividir Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 2. Configurar Límites y Malla ---
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5

xx, yy = np.meshgrid(np.arange(x_min, x_max, 1.0),
                    np.arange(y_min, y_max, 1.0))

# --- DEFINICIÓN DE COLORES PERSONALIZADOS ---
# Colores para los PUNTOS (Clase 0=Rojo, Clase 1=Azul)
cm_points = ListedColormap(['#FF0000', '#0000FF'])

# Colores para el FONDO/ZONAS (Zona 0=Amarillo Claro, Zona 1=Verde Claro)
# Usamos tonos pastel (FFFF99 y 99FF99) para que el fondo no sea agresivo a la vista
cm_background = ListedColormap(['#FFFF99', '#99FF99'])


# --- 3. Configurar Ventana ---
fig, ax = plt.subplots(figsize=(13, 7), facecolor='white')
plt.subplots_adjust(left=0.1, bottom=0.25)

# Graficar los puntos usando el nuevo mapa de colores cm_points
# Train: Puntos sólidos, bordes negros
scatter_train = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_points, 
                        edgecolors='k', s=40, label='Train (Puntos Reales)')
# Test: Puntos más transparentes
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_points, 
        edgecolors='k', alpha=0.4, s=40, label='Test')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel(feature_cols[0], fontsize=11)
ax.set_ylabel(feature_cols[1], fontsize=11)
ax.set_title("Visualizador CART - Zonas de Decisión (Amarillo=No Clic, Verde=Sí Clic)", fontsize=13, fontweight='bold')

# Crear una leyenda personalizada para los colores
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FFFF99', edgecolor='gray', label='Zona Predicción: NO Clic (Amarillo)'),
    Patch(facecolor='#99FF99', edgecolor='gray', label='Zona Predicción: SÍ Clic (Verde)'),
    plt.Line2D([0], [0], marker='o', color='w', label='Usuario Real: NO Clic (Rojo)',
            markerfacecolor='#FF0000', markersize=10, markeredgecolor='k'),
    plt.Line2D([0], [0], marker='o', color='w', label='Usuario Real: SÍ Clic (Azul)',
            markerfacecolor='#0000FF', markersize=10, markeredgecolor='k')
]
ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor='white', framealpha=1)


text_acc = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9))

contour_plot = None

# --- 4. Función de Actualización ---
def update(val):
    global contour_plot
    depth = int(slider_depth.val)
    
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    
    acc_train = clf.score(X_train, y_train)
    acc_test = clf.score(X_test, y_test)
    
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    if contour_plot:
        try: contour_plot.remove()
        except AttributeError: 
            for c in contour_plot.collections: c.remove()
            
    # Dibujar contorno usando el nuevo mapa de colores cm_background
    # Aumentamos un poco el alpha (opacidad) para que se vean bien el verde y amarillo
    contour_plot = ax.contourf(xx, yy, Z, cmap=cm_background, alpha=0.4, levels=[-0.1, 0.5, 1.1])
    
    text_acc.set_text(f"Profundidad: {depth}\nTrain Acc: {acc_train:.3f}\nTest Acc:  {acc_test:.3f}")
    
    if (acc_train - acc_test) > 0.08: # Alerta de Overfitting
        text_acc.set_bbox(dict(boxstyle="round", fc="#ffcccc", ec="red", alpha=0.9))
        text_acc.set_text(text_acc.get_text() + "\n⚠️ ALERTA: OVERFITTING")
    else:
        text_acc.set_bbox(dict(boxstyle="round", fc="white", ec="black", alpha=0.9))

    fig.canvas.draw_idle()

# --- 5. Slider ---
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgray')
slider_depth = Slider(ax_slider, 'Profundidad', 1, 20, valinit=1, valstep=1, color='#555555')

slider_depth.on_changed(update)
update(1)
plt.show()