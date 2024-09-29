from math_and_consts import *
import math as m
from icecream import ic
#from math import fabs
import datetime
import time
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
import concurrent.futures
import threading
import multiprocessing
cMu = 0.398603E6  # km^3/s^2
start_time = time.time()
def compute_trajectory(i, acceleration, napor, TETTA, X, Y, V_MOD, T, PX, nx):
    print(f"{multiprocessing.current_process()}, value = {i}")
    TETTA[i].append(i)
    X[i].append(i)
    Y[i].append(i)
    V_MOD[i].append(i)
    T[i].append(i)
    napor[i].append(i)
    nx[i].append(i)
    PX[i].append(i)
    print(f"Process {i} finished, data: TETTA={TETTA[i]}, X={X[i]}, Y={Y[i]}")  # Вывод для проверки данных

if __name__ == '__main__':
    '''prc = []  # Список для хранения процессов

    # Создаем и запускаем процессы вручную
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,))
        prc.append(p)
        p.start()

    # Ожидаем завершения всех процессов
    for p in prc:
        p.join()'''

    # Теперь создаем и запускаем задачи в пуле процессов

    manager = multiprocessing.Manager()
    acceleration = manager.list([[] for _ in range(5)])
    napor = manager.list([manager.list() for _ in range(5)])
    TETTA = manager.list([manager.list() for _ in range(5)])
    X = manager.list([manager.list() for _ in range(5)])
    Y = manager.list([manager.list() for _ in range(5)])
    T = manager.list([manager.list() for _ in range(5)])
    PX = manager.list([manager.list() for _ in range(5)])
    nx = manager.list([manager.list() for _ in range(5)])
    V_MOD = manager.list([manager.list() for _ in range(5)])

    for i in range(5):
        X[i].append(50)
    print(X)
    with multiprocessing.Pool(5) as p:
        #p.starmap(compute_trajectory, [(i, acceleration, napor, TETTA, X, Y, V_MOD, T, PX, nx) for i in range(5)])
        for i in range(5):
            p.apply(compute_trajectory, (i, acceleration, napor, TETTA, X, Y, V_MOD, T, PX, nx))
        '''results = p.map(compute_trajectory, range(5))
        p.close()
        p.join()'''
        '''multiprocessing.cpu_count() - функция, которая считает количество ядер процессора '''
        '''pr = multiprocessing.Process(target=compute_trajectory, args=(i,))
        prc.append(pr)
        pr.start()'''
        '''for i in range(5):
        p.apply_async(compute_trajectory, args=(i,)) #функция не возвращает прямого результата, требует отдельной функции для обработки результатов'''
    ''' вариант с потоками, он ускоряет процесс, но не сильно, за счет эффективного управления ресурсами потока, 
    это хорошо, но не то, что хотели
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Запускаем выполнение функции compute_trajectory для каждого i
        futures = [executor.submit(compute_trajectory, i) for i in range(5)]'''



    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)


'''if __name__ == '__main__':
    prc = []
    for i in range(5):
        prc.append(i+1) 
    print(prc)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        p.map(worker, list(range(5)))'''

