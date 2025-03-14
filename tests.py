from math_and_consts import *
import math as m
from icecream import ic
import datetime
import time
import scipy
import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
import concurrent.futures
import threading
import multiprocessing
import atmospy
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pyshtools

'''v = m.sqrt(6.674 * 10**(-11) * 5.972 * 10**(24) / 6551000)
print(v)'''


# Исходные данные
mu = 398600  # Гравитационный параметр Земли, км^3/с^2
R_earth = 6371  # Радиус Земли, км
h = 180  # Высота круговой орбиты, км
delta_v = 0.5  # Прирост скорости, км/с

# Вычисления базовых параметров
r0 = R_earth + h  # Радиус перигея, км
v_circular = np.sqrt(mu / r0)  # Круговая скорость, км/с
v_initial = v_circular + delta_v  # Начальная скорость, км/с
energy = v_initial**2 / 2 - mu / r0  # Полная удельная энергия, км^2/с^2

# Определение типа орбиты
v_parabolic = np.sqrt(2 * mu / r0)  # Параболическая скорость

eccentricity = m.sqrt(1 + (v_circular**2 * (R_earth + h)**2 * energy) / mu**2)
a = -mu / energy  # Большая полуось орбиты, км
p = a * (1 - eccentricity**2)

# Период обращения (для эллиптической орбиты)
T = 2 * np.pi * np.sqrt(a**3 / mu)  # Период обращения, с

# Уравнение Кеплера методом Ньютона
def solve_kepler(M, e, tol=1e-9):
    E = M  # Начальное приближение
    while True:
        delta_E = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= delta_E
        if abs(delta_E) < tol:
            break
    return E

# Функции для расчета параметров орбиты
def true_anomaly(E, e):
    return 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(E / 2))


def radius(E, a, e):
    return p / (1 + np.cos(E))

def velocities(E, a, e, mu):
    r = p / (1 + m.cos(true_anomaly(E, e)))
    vt = np.sqrt(mu / p) * (1 + np.cos(true_anomaly(E, e))) # Трансверсальная скорость
    vr = np.sqrt(mu / p) * e * np.sin(true_anomaly(E, e))  # Радиальная скорость
    v = np.sqrt(vr ** 2 + vt ** 2)
    return vt, vr, v

# Временной массив
time = np.linspace(0, T, 1000)
mean_anomaly = 2 * np.pi * time / T

# Расчеты для всех моментов времени
ecc_anomalies = np.array([solve_kepler(M, eccentricity) for M in mean_anomaly])
true_anomalies = np.array([true_anomaly(E, eccentricity) for E in ecc_anomalies])
radii = np.array([radius(E, p, eccentricity) for E in ecc_anomalies])
velocities_data = np.array([velocities(E, a, eccentricity, mu) for E in ecc_anomalies])

# Распаковка скоростей
vt_data = velocities_data[:, 0]
vr_data = velocities_data[:, 1]
v_data = velocities_data[:, 2]

# Построение графиков
plt.figure(figsize=(12, 10))

# График радиуса
plt.subplot(3, 2, 1)
plt.plot(time, radii)
plt.title('Радиус-вектор от времени')
plt.xlabel('Время (с)')
plt.ylabel('Радиус (км)')
plt.grid(True)

# График истинной аномалии
plt.subplot(3, 2, 2)
plt.plot(time, true_anomalies)
plt.title('Истинная аномалия от времени')
plt.xlabel('Время (с)')
plt.ylabel('Истинная аномалия (рад)')
plt.grid(True)

# График модуля скорости
plt.subplot(3, 2, 3)
plt.plot(time, v_data)
plt.title('Модуль скорости от времени')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (км/с)')
plt.grid(True)

# График трансверсальной скорости
plt.subplot(3, 2, 4)
plt.plot(time, vt_data)
plt.title('Трансверсальная скорость от времени')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (км/с)')
plt.grid(True)

# График радиальной скорости
plt.subplot(3, 2, 5)
plt.plot(time, vr_data)
plt.title('Радиальная скорость от времени')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (км/с)')
plt.grid(True)

# График эксцентрической аномалии
plt.subplot(3, 2, 6)
plt.plot(time, ecc_anomalies)
plt.title('Эксцентрическая аномалия от времени')
plt.xlabel('Время (с)')
plt.ylabel('Аномалия (рад)')
plt.grid(True)

plt.tight_layout()
plt.show()
