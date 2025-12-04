import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler

# --- 1. Generar Datos (Círculos o Lunas) ---
# Usamos 'moons' con ruido para que sea desafiante
X, y = make_moons(n_samples=400, noise=0.15, random_state=42)
# Alternativa: X, y = make_circles(n_samples=400, noise=0.1, factor=0.5, random_state=42)

# Escalar es VITAL para SVM
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Grid para el fondo
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Colores
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# --- 2. Configurar Ventana ---
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(left=0.1, bottom=0.35) # Espacio extra para 2 sliders

# Scatter plot fijo
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k', s=30)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_title("Visualizador SVM (RBF Kernel)")

# Texto de parámetros
text_params = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                      bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))

contour_plot = None

# --- 3. Función de Actualización ---
def update(val):
    global contour_plot
    
    # Leer valores (C y Gamma son logarítmicos para mejor control)
    # Los sliders van de -2 a 3, nosotros usamos 10^val
    c_val = 10 ** slider_c.val
    gamma_val = 10 ** slider_gamma.val
    
    # Entrenar modelo
    clf = SVC(kernel='rbf', C=c_val, gamma=gamma_val)
    clf.fit(X, y)
    
    # Predecir superficie
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Borrar anterior
    if contour_plot:
        try: contour_plot.remove() # Matplotlib moderno
        except AttributeError:
            for c in contour_plot.collections: c.remove() # Matplotlib antiguo
            
    # Dibujar nuevo contorno
    # Niveles cercanos a 0 muestran el margen
    contour_plot = ax.contourf(xx, yy, Z, cmap=cm, alpha=0.6, levels=np.linspace(-2, 2, 20))
    
    # Actualizar texto
    text_params.set_text(f"C = {c_val:.2f} (Estrictez)\nGamma = {gamma_val:.2f} (Alcance)")
    
    fig.canvas.draw_idle()

# --- 4. Sliders (Escala Logarítmica Visual) ---
# Slider C
ax_c = plt.axes([0.2, 0.2, 0.6, 0.03])
slider_c = Slider(ax_c, 'C (Log)', -2, 4, valinit=0, valstep=0.1) # 10^-2 a 10^4

# Slider Gamma
ax_gamma = plt.axes([0.2, 0.1, 0.6, 0.03])
slider_gamma = Slider(ax_gamma, 'Gamma (Log)', -2, 3, valinit=-0.5, valstep=0.1) # 10^-2 a 10^3

# Conectar
slider_c.on_changed(update)
slider_gamma.on_changed(update)

# Inicio
update(1)
plt.show()