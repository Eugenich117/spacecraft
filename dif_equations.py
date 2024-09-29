import numpy as np
import scipy
from icecream import ic
import math as m
from math_and_consts import *


class Difur:
    def __init__(self, data):
        self.data = data
        self.Rp = data["Rp"]#большая полуось
        self.e = data["e"]#эксцентриситет
        self.ArgLat = data["ArgLat"]#аргумент широты
        self.ArgPerigee = data["ArgPerigee"]#аргумент перигея
        self.step = data["step"]
        self.integr = data["integr"]
        self.interval = data["interval"]
        self.work = data["work"]
        self.turn = data["turn"]
        self.stabelize = data["stabelize"]


    def stabilization_esccentr(self, e, atta):#постоянство эксцентриситета
         return m.atan(-((m.sin(atta) * (1 + e * m.cos(atta)))) / (e * (m.cos(atta)) ** 2 + 2 * m.cos(atta) + e))

    def stabilization_perig(self, e, atta):#постоянство расстояния до перигея
        return m.atan((m.sin(atta) * (1 + e * m.cos(atta))) / ((2 * (1 - m.cos(atta)) + e * m.sin(atta) ** 2)))

    def stabilization_apog(self, e, atta):# постоянство расстояния до апогея
        return m.atan((m.sin(atta) * (1 + e * m.cos(atta))) / ((2 * (1 + m.cos(atta)) - e * m.sin(atta) ** 2)))

    def stabilization_phocpar(self, e, atta):# постоянство фокального параметра ( пи или 0)
        return m.pi * cToDeg #можно дать выбор пользователю, но тогда будет слишком много кнопок

    def stabilization_a(self, e, atta):# постоянство большой полуоси
        return m.atan(-((e * (m.sin(atta))) / (1 + e * m.cos(atta))))

    def stabilization_apsid(self, e, atta):# постоянство положения линии аписд
        return m.atan((m.cos(atta) / m.sin(atta)) * ((1 + e * m.cos(atta)) / (2 + e * m.cos(atta))))

    def stabilization_max_speed_esccentr(self, e, atta):# максимальная скорость изменения эксцентриситета
        return m.atan((e * (m.cos(atta)) ** 2 + 2 * m.cos(atta) + e) / ((m.sin(atta) * (1 + e * m.cos(atta)))))

    def stabilization_max_speed_perig(self, e, atta):#максимальная скорость изменения расстояния до перигея
        return m.atan(((2 * (1 - m.cos(atta)) + e *  m.sin(atta) ** 2)) / (m.sin(atta) * (1 + e * m.cos(atta))))

    def stabilization_max_speed_apog(self, e, atta):# Максимальная скорость изменения расстояния до апогея
        return m.atan(((2 * (1 + m.cos(atta)) - e * m.sin(atta) ** 2)) / (m.sin(atta) * (1 + e * m.cos(atta))))

    def stabilization_max_speed_phocpar(self, e, atta):# Максимальная скорость изменения фокального параметра
        return m.pi/2 * cToDeg

    def stabilization_max_speed_a(self, e, atta):# максимальная скорость изменения большой полуоси
        return m.atan((1 + e * m.cos(atta)) / (e * m.sin(atta)))

    def stabilization_max_speed_apsid(self, e, atta):# максимальная скорость вращения линии аписд
        return m.atan(-(m.sin(atta) / m.cos(atta)) * ((2 + e * m.cos(atta)) / (1 + e * m.cos(atta))))

    def de_func(self, initial):  # эксцентриситет
        F = initial['F']
        lam = initial['lam']
        e = initial['e']
        p = initial['p']
        atta = initial['atta']
        derivative = float(F * m.sqrt(p / cMu) * (m.sin(atta) * m.cos(lam) + ((e * (m.cos(atta)) ** 2 + 2 * m.cos(atta)
                                            + e) / (1 + e * m.cos(atta))) * m.sin(lam)))
        return derivative, 'e'

    def dp_func(self, initial):  # форкальный параметр
        F = initial['F']
        lam = initial['lam']
        e = initial['e']
        p = initial['p']
        atta = initial['atta']
        derivative = float(((2 * F * p) / (1 + e * m.cos(atta))) * m.sqrt(p / cMu) * m.sin(lam))
        return derivative, 'p'


    def domega_func(self, initial):  # аргумент перигея
        F = initial['F']
        lam = initial['lam']
        e = initial['e']
        p = initial['p']
        atta = initial['atta']
        derivative = float((F / e) * m.sqrt(p / cMu) * (-m.cos(atta) * m.cos(lam) +
            (((2 + e * m.cos(atta)) * m.sin(atta)) / (1 + e * m.cos(atta))) * m.sin(lam)))
        return derivative, 'omega'

    def datta_func(self, initial):  # истинная аномалия
        F = initial['F']
        lam = initial['lam']
        e = initial['e']
        p = initial['p']
        atta = initial['atta']
        r = initial['r']
        derivative = float((m.sqrt(cMu * p) / (r ** 2)) + ((F * m.cos(atta)) / e) * m.sqrt(p / cMu) * m.cos(lam)
                           - ((F * m.sin(atta)) / e) * (1 + r / p) * m.sqrt(p / cMu) * m.sin(lam))
        return derivative, 'atta'

    def runge_kutta_4(self, equations, initial, dt, dx):
        '''equations - это список названий функций с уравнениями для системы
        initial это переменные с начальными условиями
        dx - это список переменных, которые будут использованы для интегрирования уравнения'''
        k1 = {key: 0 for key in initial.keys()}
        k2 = {key: 0 for key in initial.keys()}
        k3 = {key: 0 for key in initial.keys()}
        k4 = {key: 0 for key in initial.keys()}

        derivatives_1 = {key: initial[key] for key in initial}
        derivatives_2 = {key: initial[key] for key in initial}
        derivatives_3 = {key: initial[key] for key in initial}
        derivatives_4 = {key: initial[key] for key in initial}

        new_values = [0] * len(equations)

        for i, eq in enumerate(equations):
            derivative, key = eq(initial)
            k1[key] += derivative
            derivatives_1[key] = initial[key] + derivative * dt / 2
            derivatives_1[dx[i]] += dt / 2
            # derivatives_1 = {key: value / 2 for key, value in derivatives_1.items()}

        for i, eq in enumerate(equations):
            derivative, key = eq(derivatives_1)
            k2[key] += derivative
            derivatives_2[key] = initial[key] + derivative * dt / 2
            derivatives_2[dx[i]] += dt / 2
            # derivatives_2 = {key: value / 2 for key, value in derivatives_2.items()}

        for i, eq in enumerate(equations):
            derivative, key = eq(derivatives_2)
            k3[key] += derivative
            derivatives_3[key] = initial[key] + derivative * dt
            derivatives_3[dx[i]] += dt

        for i, eq in enumerate(equations):
            derivative, key = eq(derivatives_3)
            k4[key] += derivative
            derivatives_4[key] = initial[key] + derivative * dt
            new_values[i] = initial[key] + (1 / 6) * dt * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key])
        return new_values

    def euler(self, equations, initial, dt, dx):
        # в equations пишем названия функций с уравнениями, а в initial пишем все переменные, которые нам нужны
        new_value_list = [0] * len(equations)

        for i, eq in enumerate(equations):
            derivative, key = eq(initial, dt)
            new_value_list[i] = initial[key] + derivative * dt  # Обновляем значение переменной по индексу

        return new_value_list

    def euler_cauchy(self, equations, initial, dt, dx):
        new_value_list = [0] * len(equations)
        k1 = {key: 0 for key in initial.keys()}
        k2 = {key: 0 for key in initial.keys()}
        derivatives_1 = {key: initial[key] for key in initial}
        '''equations - это список названий функций с уравнениями для системы
        initial это переменные с начальными условиями
        dx - это список переменных, которые будут использованы для интегрирования уравнения'''
        for eq in equations:
            derivative, key = eq(initial, dt)
            k1[key] += derivative  # полшага
            derivatives_1[key] = initial[key] + derivative * dt
        for i, eq in enumerate(equations):
            derivatives_1[dx[i]] += dt
            derivative_2, key = eq(derivatives_1, dt)
            k2[key] += derivative_2
            new_value_list[i] = initial[key] + ((dt / 2) * (k1[key] + k2[key]))

        return new_value_list

        '''  дорман-принц не работает в текущем исполнении'''
    '''def dorman_prince(self, F, lam, r, e, p, omega, atta, dt):
        # Коэффициенты для метода Дормана-Принса
        c20, c21 = 1 / 5, 1 / 5
        c30, c31, c32 = 3 / 40, 9 / 40, 3 / 10
        c40, c41, c42, c43 = 44 / 45, -56 / 15, 32 / 9, 19372 / 6561
        c50, c51, c52, c53, c54 = 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656
        c60, c61, c62, c63, c64, c65 = 35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84

        # Вычисление коэффициентов k для каждой переменной
        k1_e = self.de_func(F, lam, e, p, atta, dt)
        k1_p = self.dp_func(F, lam, e, p, atta, dt)
        k1_omega = self.domega_func(F, lam, e, p, atta, dt)
        k1_atta = self.datta_func(F, lam, e, p, atta, r, dt)

        e1 = e + c20 * k1_e * dt
        p1 = p + c20 * k1_p * dt
        omega1 = omega + c20 * k1_omega * dt
        atta1 = atta + c20 * k1_atta * dt

        k2_e = self.de_func(F, lam, e1, p1, atta1, dt)
        k2_p = self.dp_func(F, lam, e1, p1, atta1, dt)
        k2_omega = self.domega_func(F, lam, e1, p1, atta1, dt)
        k2_atta = self.datta_func(F, lam, e1, p1, atta1, r, dt)

        e2 = e + (c30 * k1_e + c31 * k2_e) * dt
        p2 = p + (c30 * k1_p + c31 * k2_p) * dt
        omega2 = omega + (c30 * k1_omega + c31 * k2_omega) * dt
        atta2 = atta + (c30 * k1_atta + c31 * k2_atta) * dt

        k3_e = self.de_func(F, lam, e2, p2, atta2, dt)
        k3_p = self.dp_func(F, lam, e2, p2, atta2, dt)
        k3_omega = self.domega_func(F, lam, e2, p2, atta2, dt)
        k3_atta = self.datta_func(F, lam, e2, p2, atta2, r, dt)

        e3 = e + (c40 * k1_e + c41 * k2_e + c42 * k3_e) * dt
        p3 = p + (c40 * k1_p + c41 * k2_p + c42 * k3_p) * dt
        omega3 = omega + (c40 * k1_omega + c41 * k2_omega + c42 * k3_omega) * dt
        atta3 = atta + (c40 * k1_atta + c41 * k2_atta + c42 * k3_atta) * dt

        k4_e = self.de_func(F, lam, e3, p3, atta3, dt)
        k4_p = self.dp_func(F, lam, e3, p3, atta3, dt)
        k4_omega = self.domega_func(F, lam, e3, p3, atta3, dt)
        k4_atta = self.datta_func(F, lam, e3, p3, atta3, r, dt)

        e4 = e + (c50 * k1_e + c51 * k2_e + c52 * k3_e + c53 * k4_e) * dt
        p4 = p + (c50 * k1_p + c51 * k2_p + c52 * k3_p + c53 * k4_p) * dt
        omega4 = omega + (c50 * k1_omega + c51 * k2_omega + c52 * k3_omega + c53 * k4_omega) * dt
        atta4 = atta + (c50 * k1_atta + c51 * k2_atta + c52 * k3_atta + c53 * k4_atta) * dt

        k5_e = self.de_func(F, lam, e4, p4, atta4, dt)
        k5_p = self.dp_func(F, lam, e4, p4, atta4, dt)
        k5_omega = self.domega_func(F, lam, e4, p4, atta4, dt)
        k5_atta = self.datta_func(F, lam, e4, p4, atta4, r, dt)

        e5 = e + (c60 * k1_e + c61 * k2_e + c62 * k3_e + c63 * k4_e + c64 * k5_e) * dt
        p5 = p + (c60 * k1_p + c61 * k2_p + c62 * k3_p + c63 * k4_p + c64 * k5_p) * dt
        omega5 = omega + (c60 * k1_omega + c61 * k2_omega + c62 * k3_omega + c63 * k4_omega + c64 * k5_omega) * dt
        atta5 = atta + (c60 * k1_atta + c61 * k2_atta + c62 * k3_atta + c63 * k4_atta + c64 * k5_atta) * dt

        # Расчет новых значений переменных
        e += (c60 * k1_e + c61 * k2_e + c62 * k3_e + c63 * k4_e + c64 * k5_e + c65 * k5_e) * dt
        p += (c60 * k1_p + c61 * k2_p + c62 * k3_p + c63 * k4_p + c64 * k5_p + c65 * k5_p) * dt
        omega += (
                             c60 * k1_omega + c61 * k2_omega + c62 * k3_omega + c63 * k4_omega + c64 * k5_omega + c65 * k5_omega) * dt
        atta += (c60 * k1_atta + c61 * k2_atta + c62 * k3_atta + c63 * k4_atta + c64 * k5_atta + c65 * k5_atta) * dt

        return e, p, omega, atta'''



