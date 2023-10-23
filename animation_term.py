# modulos
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import display 
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
plt.style.use('classic')
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Parámetros del problema
# Define tus parámetros aquí
T_inf = 20 # °C 
delta_x = 1/100 # m
k = 0.08 # w/m2°C
h = 100 # w/m2°C
# Crear una matriz A de tamaño (Nx * Ny, Nx * Ny) llena de ceros
Nx = 51  # Número de nodos en el eje x
Ny = 51  # Número de nodos en el eje y
x_len = 51
y_len = 51
p = 0.5 # m profundidad
A = np.zeros(( x_len*y_len , x_len*y_len))
c = np.zeros(x_len*y_len)

# Define las ecuaciones para nodos interiores, nodos en el borde y nodos en el exterior

# puntos interiores sin generación

def equation_interior(i, j, q):
    # Ecuación para nodos interiores
    idx = i + j * Nx
    A[idx, idx] = -4  # Coeficiente para el nodo actual
    A[idx, idx - 1] = 1  # Coeficiente para el nodo a la izquierda
    A[idx, idx + 1] = 1  # Coeficiente para el nodo a la derecha
    A[idx, idx + Nx] = 1  # Coeficiente para el nodo arriba
    A[idx, idx - Nx] = 1  # Coeficiente para el nodo abajo
    c[idx] = 0

# puntos interiores con generación 

def equation_interior_gen(i, j, q):
    # Ecuación para nodos interiores
    idx = i + j * Nx
    A[idx, idx] = -4  # Coeficiente para el nodo actual
    A[idx, idx - 1] = 1  # Coeficiente para el nodo a la izquierda
    A[idx, idx + 1] = 1  # Coeficiente para el nodo a la derecha
    A[idx, idx + Nx] = 1  # Coeficiente para el nodo arriba
    A[idx, idx - Nx] = 1  # Coeficiente para el nodo abajo
    c[idx] = (-q*delta_x**2)/(k*p)


# Esquina central

def equation_interior_wc(i, j, q):
    # Ecuación para nodos interiores
    idx = i + j * Nx
    A[idx, idx] = -2*(3 + h*delta_x/k)  # Coeficiente para el nodo actual
    A[idx, idx - 1] = 2  # Coeficiente para el nodo a la izquierda
    A[idx, idx + 1] = 1  # Coeficiente para el nodo a la derecha
    A[idx, idx + Nx] = 1  # Coeficiente para el nodo arriba
    A[idx, idx - Nx] = 2  # Coeficiente para el nodo abajo
    c[idx] = -2*h*(delta_x/(k*p))*T_inf


# a la derecha zona convectiva

def equation_plane_wc_der(i, j, q):
    # Ecuación para nodos en el borde
    idx = i + j * Nx
    A[idx, idx] = -2*(2 + h*delta_x/k)  # Coeficiente para el nodo actual
    A[idx, idx - 1] = 2  # Coeficiente para el nodo a la izquierda
    A[idx, idx + Nx] = 1  # Coeficiente para el nodo arriba
    A[idx, idx - Nx] = 1  # Coeficiente para el nodo abajo
    c[idx] = -2*h*(delta_x/(k*p))*T_inf


# arriba la zona convectiva

def equation_plane_wc_arr(i, j, q):
    # Ecuación para nodos en el borde
    idx = i + j * Nx
    A[idx, idx] = -2*(2 + h*delta_x/k)  # Coeficiente para el nodo actual
    A[idx, idx - 1] = 1  # Coeficiente para el nodo a la izquierda
    A[idx, idx + 1] = 1  # Coeficiente para el nodo a la derecha
    A[idx, idx - Nx] = 2  # Coeficiente para el nodo abajo
    c[idx] = -2*h*(delta_x/(k*p))*T_inf

# zonas con bordes aislados abajo sin generación

def equation_plane_wc_aba(i, j, q):
    # Ecuación para nodos en el borde
    idx = i + j * Nx
    A[idx, idx] = -4 # Coeficiente para el nodo actual
    A[idx, idx - 1] = 1  # Coeficiente para el nodo a la izquierda
    A[idx, idx + Nx] = 2  # Coeficiente para el nodo arriba
    A[idx, idx + 1] = 1  # Coeficiente para el nodo a la derecha
    c[idx] = 0

# esquinas zonas convectivas convectivas

def equation_esquina_wc_case4(i, j, q):
    # Ecuación para nodos en el borde
    idx = i + j * Nx
    A[idx, idx] = -2*((h*delta_x/k) + 1) # Coeficiente para el nodo actual
    A[idx, idx - 1] = 1  # Coeficiente para el nodo a la izquierda
    A[idx, idx - Nx] = 1  # Coeficiente para el nodo abajo
    c[idx] = -2*(h*delta_x/(k*p))*T_inf


# zonas con bordes aislados izquierda sin generación

def equation_plane_wc_izq(i, j, q):
    # Ecuación para nodos en el borde
    idx = i + j * Nx
    A[idx, idx] = -4  # Coeficiente para el nodo actual
    A[idx, idx + Nx] = 1  # Coeficiente para el nodo arriba
    A[idx, idx + 1] = 2  # Coeficiente para el nodo a la derecha
    A[idx, idx - Nx] = 1  # Coeficiente para el nodo abajo
    c[idx] = 0

# esquinas sin generación y aisladas REVISAR **********************

def equation_esquina_arr(i, j, q):
    # Ecuación para nodos en el borde
    idx = i + j * Nx
    # print(idx)
    A[idx, idx] = -(2 + h*delta_x/k)  # Coeficiente para el nodo actual
    A[idx, idx - Nx] = 1  # Coeficiente para el nodo abajo
    A[idx, idx + 1] = 1  # Coeficiente para el nodo a la derecha
    c[idx] = -h*(T_inf*delta_x/(k*p))

def equation_esquina_aba(i, j, q):
    # Ecuación para nodos en el borde
    idx = i + j * Nx
    A[idx, idx] = -(2 + h*delta_x/k)  # Coeficiente para el nodo actual
    A[idx, idx - 1] = 1  # Coeficiente para el nodo a la izquierda
    A[idx, idx + Nx] = 1  # Coeficiente para el nodo arriba
    c[idx] = -h*(T_inf-delta_x/(k*p))

# bordes con generación asilados 

def equation_borde_aba(i, j, q):
    # Ecuación para nodos en el borde
    idx = i + j * Nx
    A[idx, idx] = -4 # Coeficiente para el nodo actual
    A[idx, idx - 1] = 1  # Coeficiente para el nodo a la izquierda
    A[idx, idx + Nx] = 2  # Coeficiente para el nodo arriba
    A[idx, idx + 1] = 1  # Coeficiente para el nodo a la derecha
    c[idx] = -q*(delta_x**2)/(k*p)

def equation_borde_izq(i, j, q):
    # Ecuación para nodos en el borde
    idx = i + j * Nx
    A[idx, idx] = -4 # Coeficiente para el nodo actual
    A[idx, idx - Nx] = 1  # Coeficiente para el nodo abajo
    A[idx, idx + Nx] = 1  # Coeficiente para el nodo arriba
    A[idx, idx + 1] = 2  # Coeficiente para el nodo a la derecha
    c[idx] = -q*(delta_x**2)/(k*p)

# esquina inferior con generación

def equation_esquina_gen(i, j, q):
    # Ecuación para nodos en el borde
    idx = i + j * Nx
    A[idx, idx] = -2 # Coeficiente para el nodo actual
    A[idx, idx + Nx] = 1  # Coeficiente para el nodo arriba
    A[idx, idx + 1] = 1  # Coeficiente para el nodo a la derecha
    c[idx] = -q*(delta_x**2)/(2*(k*p))

# esquina inferior con generación y aislada

def equation_esquina_gen_ais_izq(i, j, q):
    # Ecuación para nodos en el borde
    idx = i + j * Nx
    A[idx, idx] = -3 # Coeficiente para el nodo actual
    A[idx, idx - Nx] = 1  # Coeficiente para el nodo abajo
    A[idx, idx + Nx] = 1  # Coeficiente para el nodo arriba
    A[idx, idx + 1] = 1  # Coeficiente para el nodo a la derecha
    c[idx] = -q*(delta_x**2)/(2*(k*p))

def equation_esquina_gen_ais_aba(i, j, q):
    # Ecuación para nodos en el borde
    idx = i + j * Nx
    A[idx, idx] = -3 # Coeficiente para el nodo actual
    A[idx, idx - 1] = 1  # Coeficiente para el nodo a la izquierda
    A[idx, idx + Nx] = 1  # Coeficiente para el nodo arriba
    A[idx, idx + 1] = 1  # Coeficiente para el nodo a la derecha
    c[idx] = -q*(delta_x**2)/(2*(k*p))

# esquina interior gen

def equation_esquina_interior_gen(i, j, q):
    # Ecuación para nodos en el borde
    idx = i + j * Nx
    A[idx, idx] = -4 # Coeficiente para el nodo actual
    A[idx, idx - Nx] = 1  # Coeficiente para el nodo abajo
    A[idx, idx - 1] = 1  # Coeficiente para el nodo a la izquierda
    A[idx, idx + Nx] = 1  # Coeficiente para el nodo arriba
    A[idx, idx + 1] = 1  # Coeficiente para el nodo a la derecha
    c[idx] = -q*(delta_x**2)/(4*(k*p))

# bordes con generación no aislados REVISAR *****************

def equation_borde_gen(i, j, q):
    # Ecuación para nodos en el borde
    idx = i + j * Nx
    A[idx, idx] = -4 # Coeficiente para el nodo actual
    A[idx, idx - Nx] = 1  # Coeficiente para el nodo abajo
    A[idx, idx - 1] = 1  # Coeficiente para el nodo a la izquierda
    A[idx, idx + Nx] = 1  # Coeficiente para el nodo arriba
    A[idx, idx + 1] = 1  # Coeficiente para el nodo a la derecha
    c[idx] = -q*(delta_x**2)/(2*(k*p))


# borde interior con generación 
def equation_borde_gen_da(i, j, q):
    # Ecuación para nodos en el borde
    idx = i + j * Nx
    A[idx, idx] = -4 # Coeficiente para el nodo actual
    A[idx, idx - Nx] = 1  # Coeficiente para el nodo abajo
    A[idx, idx - 1] = 1  # Coeficiente para el nodo a la izquierda
    A[idx, idx + Nx] = 1  # Coeficiente para el nodo arriba
    A[idx, idx + 1] = 1  # Coeficiente para el nodo a la derecha
    c[idx] = -q*(delta_x**2)/(2*(k*p))

# definición de limites de malla

# puntos de caja

# Llena la matriz A con las ecuaciones correspondientes para cada tipo de nodo
def termo_solver(q):
    for i in range(x_len):
        for j in range(y_len):
            if i >= 1 and i < 15 and j >= 1 and j < 15:
                equation_interior_gen(i, j, q)
            elif i == 30 and j == 30:
                equation_interior_wc(i, j, q)
            elif (i == 50 and j >= 1 and j < 30) or (i == 30 and j >= 31 and j < 50): ##
                equation_plane_wc_der(i, j, q)
            elif (j == 30 and i >= 31 and i < 50) or (j == 50 and i >= 1 and i < 30): ##
                equation_plane_wc_arr(i, j, q)
            elif (j == 0 and i >= 16 and i < 50):
                equation_plane_wc_aba(i, j, q)
            elif (i == 30 and j == 50) or (i == 50 and j == 30):
                equation_esquina_wc_case4(i, j, q)
            elif (i == 0 and j >= 16 and j < 50):
                equation_plane_wc_izq(i, j, q)
            elif (i == 0 and j == 50):
                equation_esquina_arr(i, j, q)
            elif (i == 50 and j == 0):
                equation_esquina_aba(i, j, q)
            elif (j == 0 and i >= 1 and i < 15):
                equation_borde_aba(i, j, q)
            elif (i == 0 and j >= 1 and j < 15):
                equation_borde_izq(i, j, q)
            elif (i == 0 and j == 0):
                equation_esquina_gen(i, j, q)
            elif (i == 0 and j == 15):
                equation_esquina_gen_ais_izq(i, j, q)
            elif (j == 0 and i == 15):
                equation_esquina_gen_ais_aba(i, j, q)
            elif (j == 15 and i == 15):
                equation_esquina_interior_gen(i, j, q)
            elif (i == 15 and j >= 1 and j < 15):
                equation_borde_gen(i, j, q)
            elif (j == 15 and i >= 1 and i < 15):
                equation_borde_gen_da(i, j, q)
            elif (i >= 16 and i < 50 and j >= 1 and j < 30):
                # Nodo interior
                equation_interior(i, j, q)
            elif (i >= 1 and i < 16 and j >= 16 and j < 30):
                # Nodo interior
                equation_interior(i, j, q)
            elif (i >= 1 and i < 30 and j >= 30 and j < 50):
                # Nodo interior
                equation_interior(i, j, q)
    # Resuelve el sistema de ecuaciones A * b = c

    filas_a_eliminar = []
    for i, fila in enumerate(A):
        if np.all(fila == 0):
            filas_a_eliminar.append(i)

    columnas_a_eliminar = []
    for j in range(A.shape[1]):
        if np.all(A[:, j] == 0):
            columnas_a_eliminar.append(j)

    A_n = np.delete(A, filas_a_eliminar, axis=0)
    A_n = np.delete(A_n, columnas_a_eliminar, axis=1)
    c_n = np.delete(c, filas_a_eliminar)

    # Actualiza la matriz de temperaturas T con los valores en b
    b = np.linalg.solve(A_n, c_n)
    # malla
    Lx = 0.5  # Longitud en el eje x
    La = 0.3
    Lb = 0.3
    # Ly = 0.5  # Longitud en el eje y
    # Crear una malla
    x_a = np.linspace(0, Lx, Nx)
    y_a = np.linspace(0, La, 31)

    x_b = np.linspace(0, Lb, 31)
    y_b = np.linspace(0.31, 0.5, 20)

    X_, Y_ = np.meshgrid(x_a, y_a)
    X2, Y2 = np.meshgrid(x_b, y_b)

    coordinates_a = np.vstack((X_.flatten(), Y_.flatten())).T
    coordinates_b = np.vstack((X2.flatten(), Y2.flatten())).T

    data = pd.DataFrame({
        'X': np.concatenate((coordinates_a[:,0],coordinates_b[:,0]))
    })

    data['Y'] = np.concatenate((coordinates_a[:,1],coordinates_b[:,1]))
    data['Temp'] = b


    # Visualización de la temperatura
    # X = data['X']
    # Y = data['Y']
    # T = data['Temp']
    return data

dq = 100  # w
goal = 700  # °C
init = 10**3
info = termo_solver(init)  # Asume que termo_solver y init están definidos en tu código

# Función que genera o actualiza los datos en cada iteración del bucle
def generate_data(dq):
    updated_info = termo_solver(init + dq)
    x_valor_deseado = 0.3  
    y_valor_deseado = 0.2
    x, y, temperatura = updated_info['X'], updated_info['Y'], updated_info['Temp']
    filtro = (updated_info['X'] == x_valor_deseado) & (updated_info['Y'] == y_valor_deseado)
    indice = updated_info.index[filtro].tolist()
    esquina = updated_info.loc[indice, 'Temp'].to_list()
    return x, y, temperatura, esquina

# Función de actualización para la animación
def update(frame):
    ax.clear()
    global dq
    dq += 100
    x, y, temperatura, esquina = generate_data(dq)
    tcf = ax.tricontourf(x, y, temperatura, levels=level, linewidths=0.5, cmap='plasma')
    contorno_lines = ax.tricontour(x, y, temperatura, levels=level, colors='k', linewidths=1)
    ax.set_title(f'Modelo de tranferencia No.{frame} con {init+dq} w')
    x_min, x_max = 0.30, 0.5  # Ejemplo de rango de coordenada X
    y_min, y_max = 0.30, 0.5  # Ejemplo de rango de coordenada Y
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=True, color='white')
    cuadrado1 = Rectangle((0.25, 0.1), 0.05, 0.1, fill=False, color='white')
    cuadrado2 = Rectangle((0.1, 0.25), 0.1, 0.05, fill=False, color='white')
    x_valor_deseado = 0.3  
    y_valor_deseado = 0.2
    value = np.round(esquina,2)
    
    ax.annotate(
    f'{value}°C',
    xy=(x_valor_deseado, y_valor_deseado), xycoords='data',
    xytext=(-60, 30), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="0.8"),
    arrowprops=dict(arrowstyle="->",
                    shrinkA=0, shrinkB=10,
                    connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    plt.gca().add_patch(cuadrado1)
    plt.gca().add_patch(cuadrado2)
    plt.gca().add_patch(rect)

# Crear una figura y un eje
fig, ax = plt.subplots()
# Inicializar los datos
x, y, temperatura, esquina = generate_data(dq)
level = 20
tcf = ax.tricontourf(x, y, temperatura, levels=level, linewidths=0.5, cmap='plasma')
contorno_lines = ax.tricontour(x, y, temperatura, levels=level, colors='k', linewidths=1)
cuadrado1 = Rectangle((0.25, 0.1), 0.05, 0.1, fill=False, color='white')
cuadrado2 = Rectangle((0.1, 0.25), 0.1, 0.05, fill=False, color='white')
x_min, x_max = 0.30, 0.5  # Ejemplo de rango de coordenada X
y_min, y_max = 0.30, 0.5  # Ejemplo de rango de coordenada Y
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
value = np.round(esquina,2)
ax.annotate(
    f'{esquina}°C',
    xy=(0.3, 0.2), xycoords='data',
    xytext=(-60, 30), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="0.8"),
    arrowprops=dict(arrowstyle="->",
                    shrinkA=0, shrinkB=10,
                    connectionstyle="angle,angleA=0,angleB=90,rad=10"))
rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=True, color='white')
plt.gca().add_patch(rect)
plt.gca().add_patch(cuadrado1)
plt.gca().add_patch(cuadrado2)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
# Crear la animación
ani = FuncAnimation(fig, update, frames=range(100), repeat=False) 
# ani.save('increasingStraightLine.mp4') 
# Mostrar la animación
plt.show()