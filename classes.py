from math_and_consts import *
from icecream import ic
import datetime
import time
from astropy.time import Time
import scipy
from astropy.coordinates import EarthLocation

class TSpacecraft:
    def __init__(self, data):
        self.data = data
        self.FEpoch = int(datetime.datetime.now().timestamp())
        self.FInitOrbit = TOrbitClass(data)
        self.FCurentOrbit = TOrbitClass(data)
        self.FTime = datetime.datetime.now().time()

    def kepler_equation(self, e, M):
        Eold = M
        Enew = 1
        Stop = False
        while not Stop:
            Enew = M + e * m.sin(Eold)
            Stop = abs(Eold - Enew) <= 1e-9
            Eold = Enew
        return Enew

    def change_time(self, my_time):
        M0 = self.FInitOrbit.mean_anomaly()

        Mnew = M0 + self.FInitOrbit.mean_motion() * my_time
        Enew = self.kepler_equation(self.FInitOrbit.e, Mnew)

        Tg = m.sqrt((1 + self.FInitOrbit.e) / (1 - self.FInitOrbit.e)) * m.tan((0.5 * Enew))
        TrAnom = 2 * m.atan(Tg)
        if TrAnom > 2 * m.pi:
            TrAnom %= (2 * m.pi)
        if TrAnom < 0:
            TrAnom += 2 * m.pi
        Unew = self.FInitOrbit.ArgPerigee + TrAnom
        if Unew > 2 * m.pi:
            Unew %= (2 * m.pi)
        if Unew < 0:
            Unew += 2 * m.pi
        self.data["ArgLat"] = Unew
        self.FCurentOrbit = TOrbitClass(self.data)
        self.FCurentOrbit.assign_graf(my_time)
        result_class_dict_graf, result_dict_graf = self.FCurentOrbit.class_to_cart_graf(my_time)
        return result_class_dict_graf, result_dict_graf

    def update(self, data_correction, my_time):
        self.data = data_correction
        self.FCurentOrbit = TOrbitClass(data_correction)
        self.FCurentOrbit.assign_graf(my_time)
        result_class_dict_graf, result_dict_graf = self.FCurentOrbit.class_to_cart_graf(my_time)
        return result_class_dict_graf, result_dict_graf
    def set_cur_time(self):
        return datetime.datetime.now().time()

    def CallSidTime(self):
        return TOrbitClass.SiderealTime

    Time = property(set_cur_time)
    InitOrbit = property(lambda self: self.FInitOrbit)
    CurentOrbit = property(lambda self: self.FCurentOrbit.SiderealTime())

ic.disable()
class TOrbitClass:
    def __init__(self, data):
        self.data = data
        self.Epoch = int(datetime.datetime.now().timestamp())
        self.Pos = T3DVector(0, 0, 0)
        self.Vel = T3DVector(0, 0, 0)
        self.Geo = TGeo()
        self.Epoch_graf = int(datetime.datetime.now().timestamp())
        self.curtime = datetime.datetime.now()
        self.Rp = data["Rp"]
        self.e = data["e"]
        self.ArgLat = data["ArgLat"]
        self.Incl = data["Incl"]
        self.AscNode = data["AscNode"]
        self.ArgPerigee = data["ArgPerigee"]
        self.step = data["step"]
        self.time = datetime.datetime.now().time()

    def assign(self):
        self.Rp = self.data["Rp"]
        self.e = self.data["e"]
        self.ArgLat = self.data["ArgLat"]
        self.Incl = self.data["Incl"]
        self.AscNode = self.data["AscNode"]
        self.ArgPerigee = self.data["ArgPerigee"]
        self.step = self.data["step"]

    def assign_graf(self, my_time):
        self.Epoch_graf = int((datetime.datetime.now() + datetime.timedelta(seconds=my_time)).timestamp())
        self.curtime = datetime.datetime.now() + datetime.timedelta(seconds=my_time)
        self.Rp = self.data["Rp"]
        self.e = self.data["e"]
        self.ArgLat = self.data["ArgLat"]
        self.Incl = self.data["Incl"]
        self.AscNode = self.data["AscNode"]
        self.ArgPerigee = self.data["ArgPerigee"]
        self.step = self.data["step"]
        self.classToCart_graf = self.class_to_cart_graf(my_time)
        #self.ToGeo_graf = self.to_geo_graf

    def class_to_cart(self):
        ic.disable()
        result_class_dict = {}

        r = self.radius()
        Cu = m.cos(self.ArgLat)
        Su = m.sin(self.ArgLat)
        Co = m.cos(self.AscNode)
        So = m.sin(self.AscNode)
        Ci = m.cos(self.Incl)
        Si = m.sin(self.Incl)
        ic(Cu, Su, Co, So, Ci, Si, r)

        result_class_dict['Pos'] = {
            'X': r * (Cu * Co - Su * So * Ci),
            'Y': r * (Cu * So + Su * Co * Ci),
            'Z': r * Su * Si
        }

        Vr_r = self.vradial() / r  # тут почему-то 0 в радиал
        rw = self.ang_rate() * r
        ic(Vr_r, rw)
        result_class_dict['Vel'] = {
            'X': Vr_r * result_class_dict['Pos']['X'] - rw * (Su * Co + Cu * So * Ci),
            'Y': Vr_r * result_class_dict['Pos']['Y'] - rw * (Su * So - Cu * Co * Ci),
            'Z': Vr_r * result_class_dict['Pos']['Z'] - rw * (-Cu * Si)
        }

        return result_class_dict

    def to_geo(self, result_class_dict):
        ic.disable()
        S = self.SiderealTime()
        ic(S)
        pos_dict = {
            'X': m.cos(S) * result_class_dict['Pos']['X'] + m.sin(S) * result_class_dict['Pos']['Y'],
            'Y': -m.sin(S) * result_class_dict['Pos']['X'] + m.cos(S) * result_class_dict['Pos']['Y'],
            'Z': result_class_dict['Pos']['Z']
        }

        R = m.sqrt(result_class_dict['Pos']['X'] ** 2 + result_class_dict['Pos']['Y'] ** 2 + result_class_dict['Pos']['Z'] ** 2)

        latitude = m.asin((pos_dict['Z'] / R))
        ic(R, latitude)

        result_dict = {
            'Pos': pos_dict,
            'R': R,
            'Latitude': latitude
        }

        R_xy = m.sqrt(result_class_dict['Pos']['X'] ** 2 + result_class_dict['Pos']['Y'] ** 2)

        if R_xy <= 0:
            longitude = S
        else:
            Cl = pos_dict['X'] / R_xy
            Sl = pos_dict['Y'] / R_xy
            Lon = m.acos(Cl)
            longitude = Lon * m.copysign(1, Sl)

        result_dict['Longitude'] = longitude

        return result_dict

    def class_to_cart_graf(self, my_time):
        ic.disable()
        result_class_dict_graf = {}

        r = self.radius()
        Cu = m.cos(self.ArgLat)
        Su = m.sin(self.ArgLat)
        Co = m.cos(self.AscNode)
        So = m.sin(self.AscNode)
        Ci = m.cos(self.Incl)
        Si = m.sin(self.Incl)
        ic(Cu, Su, Co, So, Ci, Si, r)

        result_class_dict_graf['Pos'] = {
            'X': r * (Cu * Co - Su * So * Ci),
            'Y': r * (Cu * So + Su * Co * Ci),
            'Z': r * Su * Si
        }

        Vr_r = self.vradial() / r
        rw = self.ang_rate() * r
        ic(Vr_r, rw)
        result_class_dict_graf['Vel'] = {
            'X': Vr_r * result_class_dict_graf['Pos']['X'] - rw * (Su * Co + Cu * So * Ci),
            'Y': Vr_r * result_class_dict_graf['Pos']['Y'] - rw * (Su * So - Cu * Co * Ci),
            'Z': Vr_r * result_class_dict_graf['Pos']['Z'] - rw * (-Cu * Si)
        }

        result_dict_graf = self.to_geo_graf(my_time, result_class_dict_graf)
        return result_class_dict_graf, result_dict_graf

    def to_geo_graf(self, my_time, result_class_dict_graf):
        ic.enable()
        S = self.SiderealTime_graf(my_time)
        ic(S)
        pos_dict = {
            'X': m.cos(S) * result_class_dict_graf['Pos']['X'] + m.sin(S) * result_class_dict_graf['Pos']['Y'],
            'Y': -m.sin(S) * result_class_dict_graf['Pos']['X'] + m.cos(S) * result_class_dict_graf['Pos']['Y'],
            'Z': result_class_dict_graf['Pos']['Z']
        }
        R = m.sqrt(result_class_dict_graf['Pos']['X'] ** 2 + result_class_dict_graf['Pos']['Y'] ** 2
            + result_class_dict_graf['Pos']['Z'] ** 2)

        latitude = m.asin((pos_dict['Z'] / R))

        result_dict_graf = {
            'Pos': pos_dict,
            'R': R,
            'Latitude': latitude
        }

        R_xy = m.sqrt(result_class_dict_graf['Pos']['X'] ** 2 + result_class_dict_graf['Pos']['Y'] ** 2)

        if R_xy <= 0:
            longitude = S
        else:
            Cl = pos_dict['X'] / R_xy
            Sl = pos_dict['Y'] / R_xy
            Lon = m.acos(Cl)
            longitude = Lon * m.copysign(1, Sl)

        result_dict_graf['Longitude'] = longitude

        return result_dict_graf

    '''def DateTimeToJulianDate(self):
    #работает мягко говоря так себе, но формула крутая
        ic.disable()
        AValue = datetime.datetime.now()
        # Используем функции year, month, day, hour, minute, second объекта datetime
        LYear, LDay = AValue.year - 1900, AValue.day
        LMonth = AValue.month - 3
        if LMonth < 0:
            LMonth += 12
            LYear -= 1
        LHour, LMinute, LSecond = AValue.hour, AValue.minute, AValue.second
        ic(LYear, LMonth, LDay, LHour, LMinute, LSecond)
        mjd = 15078 + 365 * LYear + int(LYear / 4) + int(0.5 + 30.6 * LMonth) + \
              LDay + LHour / 24 + LMinute / 1440 + LSecond / 86400 - 0.125  # c юлианской датой из калькулятора в инете разница в -0.1242360925898538
        rd = int(mjd + 0.125) - 15078
        nd = int(rd)
        nz = int(rd / 1461.01)
        na = nd - 1461.01 * nz
        nb = int(na / 365.25)
        myear = 4 * nz + nb + 1900
        if na == 1451:
            mmonth = 2
            mday = 29
        else:
            nz = na - 365 * nb
            ma = int((nz - 0.5) / 30.6)
            mmonth = ma + 3
            mday = nz - int(30.6 * mmonth - 91.3)
        if mmonth > 12:
            mmonth -= 12
            myear += 1
        sp = 24 * (mjd + 0.125 - int(mjd + 0.125))
        mhour = int(sp)
        sp = 60 * (sp - mhour)
        mmin = int(sp)
        msec = 60 * (sp - mmin)
        ic(myear, mmonth, mday, mhour, mmin, msec)
        julian_date = (float("{:.1f}".format(myear)) * 1000000 + mmonth * 10000 + mday * 100 + mhour + (mmin / 60)
                       + (msec / 3600))

        ic(julian_date)
        return julian_date'''

    def SiderealTime(self):
        my_time = int(datetime.datetime.now().timestamp())
        time_obj = Time(my_time, format='unix')
        julian_date = time_obj.jd
        # Создаем объект Time с использованием времени в формате Unix
        # Определяем местоположение наблюдателя (пример: Гринвич)
        location = EarthLocation.of_site('greenwich')
        # Вычисляем среднее звездное время в Гринвиче
        gst = time_obj.sidereal_time('mean', longitude=location.lon)
        gst_in_radians = gst.rad
        '''d = int(julian_date - JD2K15)
        ic(julian_date)
        M = self.Epoch - int(self.Epoch)  # (self.Epoch - self.DateTimeToJulianDate() / cTc)
        ic(M)
        t = d / cJC
        S = 1.7533685592 + 0.0172027918051 * d + 6.2831853072 * M + 6.7707139e-6 * t ** 2 - 4.50876e-10 * t ** 3

        n = int(S / cTwoPi)
        SiderealTime_const = S - n * cTwoPi'''
        return gst_in_radians  #SiderealTime_const
    #яернов и чернявский орбиты спутников ДЗЗ
    '''def DateTimeToJulianDate_graf(self, my_time):
        ic.disable()
        AValue = datetime.datetime.now() + datetime.timedelta(seconds=my_time)
        ic(AValue)
        # Используем функции year, month, day, hour, minute, second объекта datetime
        LYear, LDay = AValue.year - 1900, AValue.day
        LMonth = AValue.month - 3
        if LMonth < 0:
            LMonth += 12
            LYear -= 1
        LHour, LMinute, LSecond = AValue.hour, AValue.minute, AValue.second
        ic(LYear, LMonth, LDay, LHour, LMinute, LSecond)
        mjd = 15078 + 365 * LYear + int(LYear / 4) + int(0.5 + 30.6 * LMonth) + \
              LDay + LHour / 24 + LMinute / 1440 + LSecond / 86400 - 0.125  # c юлианской датой из калькулятора в инете разница в -0.1242360925898538
        rd = int(mjd + 0.125) - 15078
        nd = int(rd)
        nz = int(rd / 1461.01)
        na = nd - 1461.01 * nz
        nb = int(na / 365.25)
        myear = 4 * nz + nb + 1900
        if na == 1451:
            mmonth = 2
            mday = 29
        else:
            nz = na - 365 * nb
            ma = int((nz - 0.5) / 30.6)
            mmonth = ma + 3
            mday = nz - int(30.6 * mmonth - 91.3)
        if mmonth > 12:
            mmonth -= 12
            myear += 1
        sp = 24 * (mjd + 0.125 - int(mjd + 0.125))
        mhour = int(sp)
        sp = 60 * (sp - mhour)
        mmin = int(sp)
        msec = 60 * (sp - mmin)
        ic(myear, mmonth, mday, mhour, mmin, msec)
        julian_date = float("{:.1f}".format(myear)) * 1000000 + mmonth * 10000 + mday * 100 + mhour + (mmin / 60) + (
                msec / 3600)

        ic(julian_date)
        return julian_date'''

    def SiderealTime_graf(self, my_time):
        ic.disable()
        my_time = int((datetime.datetime.now() + datetime.timedelta(seconds=my_time)).timestamp())
        time_obj = Time(my_time, format='unix')
        julian_date = time_obj.jd
        # Создаем объект Time с использованием времени в формате Unix
        # Определяем местоположение наблюдателя (пример: Гринвич)
        location = EarthLocation.of_site('greenwich')

        # Вычисляем среднее звездное время в Гринвиче
        gst = time_obj.sidereal_time('mean', longitude=location.lon)
        gst_in_radians = gst.rad
        '''d = int(julian_date - JD2K15)

        M = my_time - int(my_time)
        t = d / cJC
        S = 1.7533685592 + 0.0172027918051 * d + 6.2831853072 * M + (6.7707139e-6 * t - 4.50876e-10 * t ** 2) * t
        n = int(S / cTwoPi)
        SiderealTime_const = S - n * cTwoPi'''
        return gst_in_radians  # SiderealTime_const

    ic.disable()
    def ang_rate(self): #угловая скорость
        ic.disable()
        try:
            ang_rate_const = m.sqrt(cMu * self.parameter()) / (self.radius() ** 2)
            ic(ang_rate_const)
            return ang_rate_const
        except ValueError as ve:
            print(f"Error in ang_rate: {ve}")
            return None

    def radius(self):
        ic.disable()
        try:
            radius_const = float(self.parameter() / (1 + self.e * m.cos(self.true_anomaly())))
            ic(radius_const)
            return radius_const
        except (ValueError, ZeroDivisionError) as e:
            print(f"Error in radius: {e}")
            return None

    def parameter(self):
        ic.disable()
        try:
            parameter_const = float(self.Rp * (1 + self.e))  # float(self.semi_major_axis() * (1 - self.e**2))
            ic(parameter_const)
            return parameter_const
        except ValueError as ve:
            print(f"Error in parameter: {ve}")
            return None

    def true_anomaly(self):
        ic.disable()
        try:
            true_anomaly_const = float(self.ArgLat - self.ArgPerigee)
            if true_anomaly_const > 2 * m.pi:
                true_anomaly_const %= (2 * m.pi)
            if true_anomaly_const < 0:
                true_anomaly_const += 2 * m.pi
            ic(true_anomaly_const)
            return true_anomaly_const
        except ValueError as ve:
            print(f"Error in true_anomaly: {ve}")
            return None

    def vradial(self):
        ic.disable()
        try:
            vradial_const = float(m.sqrt(cMu / self.parameter()) * self.e * m.sin(self.true_anomaly()))
            ic(vradial_const)
            return vradial_const
        except (ValueError, ZeroDivisionError) as e:
            print(f"Error in vradial: {e}")
            return None

    def mean_motion(self):
        ic.disable()
        try:
            mean_motion_const = float((2 * m.pi) / self.period())
            ic(mean_motion_const)
            return mean_motion_const
        except (ValueError, ZeroDivisionError) as e:
            print(f"Error in mean_motion: {e}")
            return None

    def period(self):
        ic.disable()
        try:
            period_const = float(2 * m.pi * self.semi_major_axis() * m.sqrt(self.semi_major_axis() / cMu))
            ic(period_const)
            return period_const
        except (ValueError, ZeroDivisionError) as e:
            print(f"Error in period: {e}")
            return None

    def semi_major_axis(self):
        ic.disable()
        try:
            semi_major_axis_const = float(self.parameter() / (1 - self.e ** 2))   # (cMu/(self.mean_motion()**2)) **(1/3)
            ic(semi_major_axis_const)
            return semi_major_axis_const
        except ZeroDivisionError as ze:
            print(f"Error in semi_major_axis: {ze}")
            return None

    def ecc_anomaly(self):
        ic.disable()
        try:
            ecc_anomaly_const = float(2 * m.atan((m.sqrt((1 - self.e) / (1 + self.e))) * m.tan((self.true_anomaly())/ 2)))
            ic(ecc_anomaly_const)
            return ecc_anomaly_const
        except ValueError as ve:
            print(f"Error in ecc_anomaly: {ve}")
            return None

    def mean_anomaly(self):
        ic.disable()
        try:
            mean_anomaly_const = float(self.ecc_anomaly() - self.e * m.sin(self.ecc_anomaly()))
            ic(mean_anomaly_const)
            return mean_anomaly_const
        except ValueError as ve:
            print(f"Error in mean_anomaly: {ve}")
            return None

    def perigee_pass_time(self):
        ic.disable()
        try:
            perigee_pass_time_const = float(
                float(datetime.datetime.now().time()) - (self.mean_anomaly() / self.mean_motion()))
            ic(perigee_pass_time_const)
            return perigee_pass_time_const
        except (ValueError, ZeroDivisionError) as e:
            print(f"Error in perigee_pass_time: {e}")
            return None

    def kepler_equation(self, e, M):
        ic.enable()
        Eps = 1e-9  # точность
        Eold = M
        Enew = 1
        Stop = False
        while not Stop:
            Enew = M + e * m.sin(Eold * cToRad)
            Stop = abs(Eold - Enew) <= Eps
            Eold = Enew
        ic(Enew)
        return Enew

    '''def change_time(self, my_time):
        M0 = self.FInitOrbit.mean_anomaly()

        Mnew = M0 + self.FInitOrbit.mean_motion() * my_time
        Enew = self.kepler_equation(self.FInitOrbit.e, Mnew)

        Tg = m.sqrt((1 + self.FInitOrbit.e) / (1 - self.FInitOrbit.e)) * m.tan(0.5 * Enew * cToRad)
        TrAnom = 2 * m.atan(Tg * cToRad)

        Unew = self.FInitOrbit.ArgPerigee + TrAnom
        result_class_dict_graf, result_dict_graf = self.FCurentOrbit.assign_graf(my_time)
        return result_class_dict_graf, result_dict_graf'''

