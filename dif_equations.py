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


    def gravi_perturbations(self, initial):
        r = initial['r']
        i = initial['i']
        u = initial['u']
        S = -1.5 * J2 * cMu * cRe ** 2 / r ** 4 * (1 - 3 * m.sin(i) ** 2 * m.sin(u) ** 2)
        T = -3 * J2 * cMu * cRe ** 2 / r ** 4 * m.sin(i) ** 2 * m.sin(u) * m.cos(u)
        W = -3 * J2 * cMu * cRe ** 2 / r ** 4 * m.sin(i) * m.cos(i) * m.sin(u)

        return S, T, W

    '''def atmospheric_acceleration(self, initial):

        Ca = initial["Ca"]
        S_surf = initial["S"]  # Площадь, м²
        mass = 100  # кг

        # Орбитальные элементы
        a_km = initial["a"]
        e = initial["e"]
        inc = initial["i"]  # наклон (inclination), рад
        RAAN = initial["ascnode"]  # восходящий узел (Right Ascension of Ascending Node), рад
        argp = initial["omega"]  # аргумент перицентра (argument of pericenter), рад
        nu = initial["atta"]  # истинная аномалия (true anomaly), рад

        # Радиус и высота
        a = a_km * 1000  # большая полуось, м
        r = a * (1 - e ** 2) / (1 + e * np.cos(nu))  # текущий радиус, м
        h_km = r / 1000 - 6371  # высота над поверхностью, км

        if h_km > 1500:
            return 0, 0, 0

        # Гравитационный параметр
        mu = cMu * 1e9  # м³/с²

        # Орбитальная скорость по энергии
        v = np.sqrt(mu * (2 / r - 1 / a))

        # Атмосферная плотность (экспоненциальная модель)
        def approx_density(h_km):
            if h_km < 25:
                return 1.225 * np.exp(-h_km / 7.64)
            elif h_km < 100:
                return 3.899e-15 * np.exp(-(h_km - 90) / 6.5)
            elif h_km < 1500:
                return 5e-11 * np.exp(-(h_km - 100) / 100)
            else:
                return 0

        rho = approx_density(h_km)

        # Орбитальный момент
        h = np.sqrt(mu * a * (1 - e ** 2))

        # Вектор скорости в перицентральной системе координат (PQW)
        vx_p = (mu / h) * (-np.sin(nu))
        vy_p = (mu / h) * (e + np.cos(nu))
        vz_p = 0
        v_vec_pqw = np.array([vx_p, vy_p, vz_p])

        # Поворот из PQW в ECI (через RAAN, inc, argp)
        def rotation_matrix(RAAN, inc, argp):
            cosO, sinO = np.cos(RAAN), np.sin(RAAN)
            cosi, sini = np.cos(inc), np.sin(inc)
            cosw, sinw = np.cos(argp), np.sin(argp)

            R = np.array([
                [cosO * cosw - sinO * sinw * cosi, -cosO * sinw - sinO * cosw * cosi, sinO * sini],
                [sinO * cosw + cosO * sinw * cosi, -sinO * sinw + cosO * cosw * cosi, -cosO * sini],
                [sinw * sini, cosw * sini, cosi]
            ])
            return R

        R = rotation_matrix(RAAN, inc, argp)
        v_vec_eci = R @ v_vec_pqw

        # Нормированный вектор скорости
        v_hat = v_vec_eci / np.linalg.norm(v_vec_eci)

        # Модуль аэродинамического ускорения
        a_drag = 0.5 * Ca * (S_surf / mass) * rho * v ** 2

        # Вектор аэродинамического ускорения (против движения)
        a_vec = -a_drag * v_hat

        # Вектор положения в PQW
        rx_p = r * np.cos(nu)
        ry_p = r * np.sin(nu)
        rz_p = 0
        r_vec_pqw = np.array([rx_p, ry_p, rz_p])
        r_vec_eci = R @ r_vec_pqw

        # Базис СКО: S, T, W
        S_dir = r_vec_eci / np.linalg.norm(r_vec_eci)
        W_dir = np.cross(r_vec_eci, v_vec_eci)
        W_dir /= np.linalg.norm(W_dir)
        T_dir = np.cross(W_dir, S_dir)

        # Разложение ускорения по S, T, W
        a_S = np.dot(a_vec, S_dir)
        a_T = np.dot(a_vec, T_dir)
        a_W = np.dot(a_vec, W_dir)

        return a_S, a_T, a_W'''

    def none_perturbations(self, initial):
        S, T, W = 0, 0, 0
        return S, T, W

    def atmospheric_acceleration(self, initial):
        """Вычисляет компоненты аэродинамического ускорения (S, T, W)"""
        Cx = initial["Cx"]
        S_surf = initial["S"]  # Площадь поперечного сечения, м²
        mass = initial['mass']  # Масса аппарата, кг

        # Орбитальные параметры
        r_km = initial["r"]  # Радиус-вектор, км
        a_km = initial["a"]  # Большая полуось, км
        e = initial["e"]  # Эксцентриситет
        u = initial["u"]  # Аргумент широты, рад

        h_km = r_km - 6371
        if h_km > 1500:
            return 0.0, 0.0, 0.0

        # Перевод в метры
        r = r_km * 1000
        a = a_km * 1000

        # Константы атмосферы
        H = 8.5e3  # Масштабная высота атмосферы, м
        ro0 = 1.225  # Плотность у поверхности, кг/м³
        h = r - 6371e3  # Высота над поверхностью, м

        # Плотность атмосферы
        try:
            exponent = -h / H
            if exponent < -700:  # предел float64, чтобы избежать underflow
                ro = 0.0
            else:
                ro = ro0 * m.exp(exponent)
        except OverflowError:
            ro = 0.0

        # Гравитационный параметр
        mu = cMu * 1e9  # м³/с²

        # Орбитальные параметры
        p = a * (1 - e ** 2)
        A = 1 + e * m.cos(u)
        B = m.sqrt(1 + e ** 2 + 2 * e * m.cos(u))
        sigma = Cx * S_surf / (2 * mass)

        # Аэродинамическое ускорение по нормали к траектории (суммарное)
        Fa = sigma * mu / p * B ** 2 * ro

        # Разложение на компоненты
        T = -Fa * A / B  # Тангенциальная компонента
        S = -Fa * e * m.sin(u) / B  # Радиальная компонента
        W = 0.0  # Нормальная компонента (не учитывается)

        return S, T, W

    '''   def atmospheric_acceleration(self, initial):
        """Вычисляет компоненты аэродинамического ускорения (S, T, W)"""
        Cx = initial["Cx"]
        S_surf = initial["S"]  # Площадь поперечного сечения (м²)
        mass = 100  # кг

        # Орбитальные параметры
        r_km = initial["r"]  # в км
        a_km = initial["a"]  # большая полуось (в км)
        e = initial['e']
        h_km = r_km - 6371

        if h_km > 1500:
            return 0, 0, 0

        # Перевод в метры
        r = r_km * 1000
        a = a_km * 1000

        # Константы атмосферы
        H = 8.5e3  # м
        ro0 = 1.225  # кг/м³
        h = r - 6371e3  # м

        # Атмосферная плотность
        ro = ro0 * m.exp(-h / H)

        # Скорость на эллиптической орбите
        mu = cMu * 1e9  # Приводим к м³/с²
        try:
            #v = m.sqrt(mu * (2 / r - 1 / a))  # м/с
            v = m.sqrt((1 - e) / (a * (1 + e)))
        except Exception as e:
            print(f"r = {r}, a = {a}, e = {e}")

        # Аэродинамическое ускорение
        a_drag = 0.5 * Cx * (S_surf / mass) * ro * v ** 2

        # Воздействие по трансверсальной оси (вдоль вектора скорости)
        return 0, -a_drag, 0'''


    def da_func(self, initial):  # большая полуось
        F = initial['F']
        lam = initial['lam']
        beta = m.pi / 2
        e = initial['e']
        p = initial['p']
        atta = initial['atta']
        r = initial['r']
        i = initial['i']
        u = initial['u']

        if initial['perturbation'] == 'none':
            S, T, W = 0, 0, 0
        if initial['perturbation'] == 'gravy':
            S, T, W = self.gravi_perturbations(initial)  # гравитационные возмущения
        if initial['perturbation'] == 'atmosphere':
            S, T, W = self.atmospheric_acceleration(initial) #ускорения от атмосферных возмущений

        try:
            derivative = float((2 * p / (1 - e) ** 2) * m.sqrt(p / cMu) * (e * m.sin(atta) * S + (1 + e * m.cos(atta)) * T))
            return derivative, 'a'
        except (ValueError, ZeroDivisionError) as e:
            print(f"Error in radius: {e}")

    def de_func(self, initial):  # эксцентриситет
        F = initial['F']
        lam = initial['lam']
        beta = m.pi / 2
        p = initial['p']
        e = initial['e']
        atta = initial['atta']
        r = initial['r']
        i = initial['i']
        u = initial['u']

        if initial['perturbation'] == 'none':
            S, T, W = 0, 0, 0
        if initial['perturbation'] == 'gravy':
            S, T, W = self.gravi_perturbations(initial)  # гравитационные возмущения
        if initial['perturbation'] == 'atmosphere':
            S, T, W = self.atmospheric_acceleration(initial)  # ускорения от атмосферных возмущений

        try:
            derivative = float(m.sqrt(p / cMu) * (S * m.sin(atta) + T * ((1 + r / p) * m.cos(atta) + e * (r / p))))
            #derivative = float(m.sqrt(p / cMu) * (S * m.sin(atta) + ((e * m.cos(atta) ** 2 + 2 * m.cos(atta) + e) / (1 + e * m.cos(atta))) * T))
            return derivative, 'e'
        except (ValueError, ZeroDivisionError) as e:
            print(f"Error in eccen: {e}")

    def di_func(self, initial):  # наклонение
        F = initial['F']
        beta = m.pi / 2
        p = initial['p']
        e = initial['e']
        atta = initial['atta']
        u = initial['u']
        r = initial['r']
        i = initial['i']

        if initial['perturbation'] == 'none':
            S, T, W = 0, 0, 0
        if initial['perturbation'] == 'gravy':
            S, T, W = self.gravi_perturbations(initial)  # гравитационные возмущения
        if initial['perturbation'] == 'atmosphere':
            S, T, W = self.atmospheric_acceleration(initial)  # ускорения от атмосферных возмущений

        try:
            derivative = float(m.sqrt(p / cMu) * (m.cos(u) / (1 + e * m.cos(atta))) * W)
            return derivative, 'i'
        except (ValueError, ZeroDivisionError) as e:
            print(f"Error in i: {e}")

    def domega_func(self, initial):  # аргумент перицентра
        F = initial['F']
        lam = initial['lam']
        beta = m.pi / 2
        a = initial['a']
        p = initial['p']
        e = initial['e']
        atta = initial['atta']
        u = initial['u']
        i = initial['i']
        r = initial['r']
        omega = initial['omega']

        if initial['perturbation'] == 'none':
            S, T, W = 0, 0, 0
        if initial['perturbation'] == 'gravy':
            S, T, W = self.gravi_perturbations(initial)  # гравитационные возмущения
        if initial['perturbation'] == 'atmosphere':
            S, T, W = self.atmospheric_acceleration(initial)  # ускорения от атмосферных возмущений

        try:
            if e == 0:  # круговая орбита
                derivative = 0
                return derivative, 'omega'
            else:  # эллиптическая орбита
                derivative = float((1 / e) * m.sqrt(p / cMu) * (-S * m.cos(atta) + (m.sin(atta) * (2 + e * m.cos(atta)) /
                    (1 + e * m.cos(atta))) * T - ((e * m.sin(u) / m.tan(i)) / (1 + e * m.cos(atta))) * W))
            return derivative, 'omega'
        except (ValueError, ZeroDivisionError) as e:
            print(f"Error in omega: {e}")

    def dascnode_func(self, initial):  # долгота восходящего узла
        F = initial['F']
        beta = m.pi / 2
        p = initial['p']
        e = initial['e']
        atta = initial['atta']
        u = initial['u']
        i = initial['i']
        r = initial['r']

        if initial['perturbation'] == 'none':
            S, T, W = 0, 0, 0
        if initial['perturbation'] == 'gravy':
            S, T, W = self.gravi_perturbations(initial)  # гравитационные возмущения
        if initial['perturbation'] == 'atmosphere':
            S, T, W = self.atmospheric_acceleration(initial)  # ускорения от атмосферных возмущений

        try:
            derivative = float(m.sqrt(p / cMu) * (m.sin(u) / ((1 + e * m.cos(atta)) * m.sin(i)) * W))
            #derivative = 2 * m.pi / (365.2422 * 86400)
            return derivative, 'ascnode'
        except (ValueError, ZeroDivisionError) as e:
            print(f"Error in radius: {e}")


    def du_func(self, initial):  # аргумент широты
        F = initial['F']
        beta = m.pi / 2
        p = initial['p']
        e = initial['e']
        atta = initial['atta']
        u = initial['u']
        i = initial['i']
        r = initial['r']

        if initial['perturbation'] == 'none':
            S, T, W = 0, 0, 0
        if initial['perturbation'] == 'gravy':
            S, T, W = self.gravi_perturbations(initial)  # гравитационные возмущения
        if initial['perturbation'] == 'atmosphere':
            S, T, W = self.atmospheric_acceleration(initial)  # ускорения от атмосферных возмущений

        try:
            if e == 0:
                derivative = 0
            else:
                term1 = (1 + e * m.cos(atta)) ** 2
                term2 = W * ((p ** 2) / (cMu * (1 + e * m.cos(atta)))) * (1 / m.tan(i)) * m.sin(u)
                derivative = float((m.sqrt(cMu * p) / p ** 2) * (term1 - term2))
            return derivative, 'u'
        except (ValueError, ZeroDivisionError) as e:
            print(f"Error in u: {e}")


    def datta_func(self, initial):  # истинная аномалия
        F = initial['F']
        lam = initial['lam']
        e = initial['e']
        p = initial['p']
        atta = initial['atta']
        r = initial['r']
        beta = m.pi / 2
        u = initial['u']
        i = initial['i']
        a = initial['a']

        if initial['perturbation'] == 'none':
            S, T, W = 0, 0, 0
        if initial['perturbation'] == 'gravy':
            S, T, W = self.gravi_perturbations(initial)  # гравитационные возмущения
        if initial['perturbation'] == 'atmosphere':
            S, T, W = self.atmospheric_acceleration(initial)  # ускорения от атмосферных возмущений

        try:
            if e == 0:  # круговая орбита
                derivative = m.sqrt(cMu / a ** 3) + (2 / m.sqrt(cMu * a)) * T
            else:
                derivative = float((m.sqrt(cMu * p) / (r ** 2)) + (1 / e) * m.sqrt(p / cMu) * (S * m.cos(atta) + T * m.sin(atta)) * (1 + r / p))
            return derivative, 'atta'
        except (ValueError, ZeroDivisionError) as e:
            print(f"Error in atta: {e}")


    def atta_func(self, initial):
        u = initial['u']
        atta = initial['atta']
        e = initial['e']
        omega = initial['omega']
        a = initial['a']

        if u > 2 * m.pi:
            u %= (2 * m.pi)
        if u < 0:
            u += 2 * m.pi

        atta = u - omega
        if atta >= m.pi:
            atta -= (2 * m.pi)
        if atta <= - m.pi:
            atta += 2 * m.pi
        p = a * (1 - e ** 2)

        r = p / (1 + e * m.cos(atta))
        print(atta, u, r, p)
        return atta, u, r, p


    '''  смешивает ключи и неверно интерпретирует уравнения
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
        '''
    def runge_kutta_1(self, equations, initial, dt, dx):
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
            #derivatives_1['atta'], derivatives_1['u'], derivatives_1['r'], derivatives_1['p'] = self.atta_func(initial)
            # derivatives_1 = {key: value / 2 for key, value in derivatives_1.items()}

        for i, eq in enumerate(equations):
            derivative, key = eq(derivatives_1)
            k2[key] += derivative
            derivatives_2[key] = initial[key] + derivative * dt / 2
            derivatives_2[dx[i]] += dt / 2
            #derivatives_2['atta'], derivatives_2['u'], derivatives_2['r'], derivatives_2['p'] = self.atta_func(derivatives_1)
            # derivatives_2 = {key: value / 2 for key, value in derivatives_2.items()}

        for i, eq in enumerate(equations):
            derivative, key = eq(derivatives_2)
            k3[key] += derivative
            derivatives_3[key] = initial[key] + derivative * dt
            derivatives_3[dx[i]] += dt
            #derivatives_3['atta'], derivatives_3['u'], derivatives_3['r'], derivatives_3['p'] = self.atta_func(derivatives_2)

        for i, eq in enumerate(equations):
            derivative, key = eq(derivatives_3)
            k4[key] += derivative
            derivatives_4[key] = initial[key] + derivative * dt
            new_values[i] = initial[key] + (1 / 6) * dt * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key])
        return new_values

    def runge_kutta_4(self, equations, initial, dt, dx):
        k1 = {key: 0 for key in dx}
        k2 = {key: 0 for key in dx}
        k3 = {key: 0 for key in dx}
        k4 = {key: 0 for key in dx}

        # k1
        for eq in equations:
            derivative, key = eq(initial)
            if key in dx:
                k1[key] = derivative

        # k2
        state_k2 = initial.copy()
        for key in dx:
            state_k2[key] += dt / 2 * k1[key]

        for eq in equations:
            derivative, key = eq(state_k2)
            if key in dx:
                k2[key] = derivative

        # k3
        state_k3 = initial.copy()
        for key in dx:
            state_k3[key] += dt / 2 * k2[key]

        for eq in equations:
            derivative, key = eq(state_k3)
            if key in dx:
                k3[key] = derivative

        # k4
        state_k4 = initial.copy()
        for key in dx:
            state_k4[key] += dt * k3[key]

        for eq in equations:
            derivative, key = eq(state_k4)
            if key in dx:
                k4[key] = derivative

        # итоговое обновление только для переменных из dx
        new_values = []
        for key in dx:
            new_val = initial[key] + (dt / 6) * (
                    k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key]
            )
            new_values.append(new_val)

        return new_values


    def euler(self, equations, initial, dt, dx):
        # в equations пишем названия функций с уравнениями, а в initial пишем все переменные, которые нам нужны
        new_value_list = [0] * len(equations)

        for i, eq in enumerate(equations):
            derivative, key = eq(initial)
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
            derivative, key = eq(initial)
            k1[key] += derivative  # полшага
            derivatives_1[key] = initial[key] + derivative * dt
        for i, eq in enumerate(equations):
            derivatives_1[dx[i]] += dt
            derivative_2, key = eq(derivatives_1)
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



