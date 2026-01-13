# -*- coding: utf-8 -*-
"""
Projeto de um trecho de uma LT

@author: Guilherme Soares
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
%matplotlib

# Fecha todas as janelas de gráficos abertas
plt.close('all') 

# Equação para encontrar x0 (coordenada x do ponto mais baixo da catenária)
def equation(x0, *data):
    xte, xtd, yte, ytd, T0, ms = data
    return yte - ytd - T0/ms*(math.cosh(ms/T0*(xte-x0))-math.cosh(ms/T0*(xtd-x0)))

# Carrega dados de um arquivo CSV
# M vai ter primeiro a posição e depois a altura usecols=(3,2)
M = np.loadtxt("route.csv", delimiter=',', skiprows=1, usecols=(3, 2)) 

# Converte a unidade da primeira coluna para metros
M[:,0] = M[:,0] * 1000

# plt.figure(3)
# plt.plot(M[:,0], M[:,1])

# Definição das restrições de altura
rmin = 6
R = np.array([[200, 8], [400, 8], [600, 9]])

# Inicializa a altura mínima y_min com base em rmin
y_min = M[:,1] + rmin

# Ajusta y_min com base nas restrições específicas em R
for i in range(len(R[:,0])):
    indice = np.where(M[:,0] >= R[i,0])[0][0]
    y_min[indice-2:indice+2] = M[indice-2:indice+2, 1] + R[i, 1]

# Cria a figura para relevo, torres e catenárias
plt.figure(1)
plt.plot(M[:,0], M[:,1], 'b', M[:,0], y_min, 'r')

# Definição das coordenadas x aproximadas das torres
# Entre 250 a 400 metros de distância entre as torres
xt1 = 0
xt2 = 290
xt3 = 680 # TODO: ADICIONADO DE EXEMPLO
H = 20

# Encontra os índices correspondentes em M
i_t1 = np.where(M[:,0] >= xt1)[0][0]
i_t2 = np.where(M[:,0] >= xt2)[0][0]
i_t3 = np.where(M[:,0] >= xt3)[0][0] # TODO: ADICIONADO DE EXEMPLO

# Corrige as coordenadas x das torres
xt1 = M[i_t1, 0]
xt2 = M[i_t2, 0]
xt3 = M[i_t3, 0] # TODO: ADICIONADO DE EXEMPLO

# Calcula as coordenadas y das torres com a altura H adicionada
yt1 = M[i_t1, 1] + H
yt2 = M[i_t2, 1] + H
yt3 = M[i_t3, 1] + H # TODO: ADICIONADO DE EXEMPLO

# Define as posições das torres T1 e T2
T1 = np.array([[xt1, M[i_t1, 1]], [xt1, yt1]])
T2 = np.array([[xt2, M[i_t2, 1]], [xt2, yt2]])
T3 = np.array([[xt3, M[i_t3, 1]], [xt3, yt3]])

# Plota as torres na figura
plt.figure(1)
plt.plot(T1[:,0], T1[:,1], 'k')
plt.plot(T2[:,0], T2[:,1], 'k')
plt.plot(T3[:,0], T3[:,1], 'k') # TODO: ADICIONADO DE EXEMPLO

# Definição dos parâmetros do cabo
Tnom = 140.6e3  # Tração nominal
Tmax = 0.25 * Tnom  # Tração máxima permitida
ms = 1473 * 9.81 / 1000  # Massa específica do cabo (N/m)


# Exemplo (não vai ter na prova)
# =========================== VÃO 1 ==============================
T0 = 0.96 * Tmax  # Tração inicial
# Ajustar a posição da torre á direita (ajuste grosso)
# Ajustar o T0 (ajuste fino)

# Calcula x0 (coordenada x do ponto mais baixo da catenária)
param = (xt1, xt2, yt1, yt2, T0, ms)
x0 = fsolve(equation, xt1, args=param)[0]

# Calcula y0 (coordenada y do ponto mais baixo da catenária)
y0 = yt1 - T0/ms*(math.cosh((ms/T0)*(xt1-x0))-1)

# Define os pontos x e y da catenária no intervalo de xt1 a xt2
catenaria_x = np.arange(xt1-x0, xt2-x0, 1)
catenaria_y = np.array([T0/ms*(math.cosh(ms/T0*x)-1) for x in catenaria_x])

# Ajusta os pontos da catenária para o sistema de coordenadas real
catenaria_x_real = catenaria_x + x0
catenaria_y_real = catenaria_y + y0

# Plota a catenária na figura
plt.figure(1)
plt.plot(catenaria_x_real, catenaria_y_real, 'm')

# Gráfico da tração no cabo
T = T0 + catenaria_y * ms
Vtmax = Tmax * np.ones(len(catenaria_x_real))

# Cria a figura para o gráfico da tração
plt.figure(2)
plt.xlabel("x[m]")
plt.ylabel("Tração [N]")
plt.plot(catenaria_x_real, T, 'b', catenaria_x_real, Vtmax, 'r')
# =========================== VÃO 1 ==============================

# =========================== VÃO 2 ============================== # TODO: ADICIONADO DE EXEMPLO
T0 = 0.965 * Tmax  # Tração inicial
# Ajustar a posição da torre á direita (ajuste grosso)
# Ajustar o T0 (ajuste fino)

# Calcula x0 (coordenada x do ponto mais baixo da catenária)
param = (xt2, xt3, yt2, yt3, T0, ms)
x0 = fsolve(equation, xt2, args=param)[0]

# Calcula y0 (coordenada y do ponto mais baixo da catenária)
y0 = yt2 - T0/ms*(math.cosh((ms/T0)*(xt2-x0))-1)

# Define os pontos x e y da catenária no intervalo de xt1 a xt2
catenaria_x = np.arange(xt2-x0, xt3-x0, 1)
catenaria_y = np.array([T0/ms*(math.cosh(ms/T0*x)-1) for x in catenaria_x])

# Ajusta os pontos da catenária para o sistema de coordenadas real
catenaria_x_real = catenaria_x + x0
catenaria_y_real = catenaria_y + y0

# Plota a catenária na figura
plt.figure(1)
plt.plot(catenaria_x_real, catenaria_y_real, 'm')

# Gráfico da tração no cabo
T = T0 + catenaria_y * ms
Vtmax = Tmax * np.ones(len(catenaria_x_real))

# Cria a figura para o gráfico da tração
plt.figure(2)
plt.xlabel("x[m]")
plt.ylabel("Tração [N]")
plt.plot(catenaria_x_real, T, 'b', catenaria_x_real, Vtmax, 'r')
# =========================== VÃO 1 ==============================