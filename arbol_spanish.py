import matplotlib.pyplot as plt

# Configuración de la figura con mayor espacio horizontal y alta resolución (dpi=300)
fig, ax = plt.subplots(figsize=(24, 12), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Título principal estilizado
plt.title("Árbol de Decisión para Enfermedad Cardíaca (Profundidad Máxima = 3)", 
          fontsize=16, fontweight='bold', pad=25, color='#1D3557')

# Estilos profesionales para las cajas con colores diferenciados
estilo_no = dict(boxstyle="round,pad=0.4", facecolor="#FFE5D9", edgecolor="#E07A5F", linewidth=1.5)
estilo_si = dict(boxstyle="round,pad=0.4", facecolor="#A8DADC", edgecolor="#457B9D", linewidth=1.5)
estilo_raiz = dict(boxstyle="round,pad=0.5", facecolor="#F4A261", edgecolor="#D62828", linewidth=2)

def dibujar_nodo(x, y, texto, estilo):
    ax.text(x, y, texto, ha='center', va='center', fontsize=7.5, 
            bbox=estilo, family='sans-serif', linespacing=1.3)

def dibujar_flecha(x1, y1, x2, y2, etiqueta=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#457B9D", lw=1.5, shrinkA=5, shrinkB=5))
    if etiqueta:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.8, etiqueta, fontsize=8, ha='center', fontweight='bold', color='#1D3557',
                bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.8))

# --- NIVEL 0 (Nodo Raíz) ---
raiz = "Dif. Caminar Sí <= 0.5\ngini = 0.164\nmuestras = 226,287\nvalores = [205,841, 20,446]\nclase = No"
dibujar_nodo(50, 88, raiz, estilo_raiz)

# Conexiones Nivel 0 -> 1
dibujar_flecha(44, 82, 26, 70, "Verdadero")
dibujar_flecha(56, 82, 74, 70, "Falso")

# --- NIVEL 1 ---
l1_1 = "Ataque Cerebral Sí <= 0.5\ngini = 0.124 | muestras = 192,904\nvalores = [180,046, 12,858] | clase = No"
l1_2 = "Ataque Cerebral Sí <= 0.5\ngini = 0.351 | muestras = 33,383\nvalores = [25,795, 7,588] | clase = No"
dibujar_nodo(26, 67, l1_1, estilo_no)
dibujar_nodo(74, 67, l1_2, estilo_no)

# Conexiones Nivel 1 -> 2
dibujar_flecha(20, 62, 13, 49)
dibujar_flecha(32, 62, 38, 49)
dibujar_flecha(68, 62, 62, 49)
dibujar_flecha(80, 62, 87, 49)

# --- NIVEL 2 ---
l2_1 = "Diabético Sí <= 0.5\ngini = 0.114 | m = 187,871\nval = [176,504, 11,367]"
l2_2 = "Salud Regular <= 0.5\ngini = 0.417 | m = 5,033\nval = [3,542, 1,491]"
l2_3 = "Diabético Sí <= 0.5\ngini = 0.317 | m = 29,380\nval = [23,581, 5,799]"
l2_4 = "Salud Mala <= 0.5\ngini = 0.494 | m = 4,003\nval = [2,214, 1,789]"

dibujar_nodo(13, 46, l2_1, estilo_no)
dibujar_nodo(38, 46, l2_2, estilo_no)
dibujar_nodo(62, 46, l2_3, estilo_no)
dibujar_nodo(87, 46, l2_4, estilo_no)

# Conexiones Nivel 2 -> 3 (Hojas)
xs_nivel2 = [13, 38, 62, 87]
offsets = [-5.5, 5.5]
hojas_coords = []
for x_padre in xs_nivel2:
    for off in offsets:
        x_hijo = x_padre + off
        dibujar_flecha(x_padre, 41, x_hijo, 27)
        hojas_coords.append(x_hijo)

# --- NIVEL 3 (Nodos Hoja / Resultados) ---
hojas_data = [
    ("gini = 0.095\nm = 169,032\n[160586, 8446]\nNo", estilo_no),
    ("gini = 0.262\nm = 18,839\n[15918, 2921]\nNo", estilo_no),
    ("gini = 0.386\nm = 3,988\n[2948, 1040]\nNo", estilo_no),
    ("gini = 0.491\nm = 1,045\n[594, 451]\nNo", estilo_no),
    ("gini = 0.269\nm = 20,723\n[17400, 3323]\nNo", estilo_no),
    ("gini = 0.408\nm = 8,657\n[6181, 2476]\nNo", estilo_no),
    ("gini = 0.480\nm = 2,850\n[1709, 1141]\nNo", estilo_no),
    ("gini = 0.492\nm = 1,153\n[505, 648]\nSí", estilo_si), # Única hoja clase Sí
]

for (x, (texto, estilo)) in zip(sorted(hojas_coords), hojas_data):
    dibujar_nodo(x, 23, texto, estilo)

# Guardar la imagen PNG optimizada y limpia
plt.tight_layout()
plt.savefig("arbol_decision_organizado.png", format="png", dpi=300, bbox_inches='tight')
plt.show()