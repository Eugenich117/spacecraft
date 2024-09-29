import math as m

#модуль математики

# Constants
cRe = 6371.210  # EarthMeanRadius in km
cW = 0.7292115E-4  # Angular velocity of Earth rotation in 1/s
cMu = 0.398603E6  # km^3/s^2
JD2K15 = 2451545.0
cToDeg = 180 / m.pi
cToRad = m.pi / 180
cTwoPi = 2 * m.pi
cAlmost0 = 1e-8
cEps = (23 + (27 / 60) + (8 / 3600)) * cToRad
cC20 = 1.082627E-3
cTs = 86164.03
cTc = 86400
cAls = 499.00478353
cVl = 299792.458
cJC = 36525

'''если че в процессе написания основных функций я отказался от использовани этих классов, потому что хранение данных в словарях 
 удобнее по сравнению с вот этим вот, тем более они предлагают сокращение кода в 1 строку, что вообще никак не ускоряет, 
 а наоборот замедляет разработку возней с созданием экземпляра класса'''
class MathVec:
    def __init__(self, v):
        self.V = v

    def VecMod(self, V):
        self.mod = m.sqrt(V.X*V.X + V.Y*V.Y + V.Z*V.Z)

    def ScalAxB(self, A, B):
        self.scal = A.X * B.X + A.Y * B.Y + A.Z * B.Z

    def SumVec(self, A, B):
        self.X = A.X + B.X
        self.Y = A.Y + B.Y
        self.Z = A.Z + B.Z

    def RasVec(self, A, B):
        self.X = A.X - B.Y
        self.Y = A.Y - B.Z
        self.Z = A.Z - B.Z

    def vec_norma(self, Pos):
        return m.sqrt(Pos.X ** 2 + Pos.Y ** 2 + Pos.Z ** 2)


class T3DVector:
    def __init__(self, x, y, z):
        self.X = x
        self.Y = y
        self.Z = z


class TGeo:
    def __init__(self, pos=None, r=None, longitude=None, latitude=None):
        self.Pos = pos
        self.R = r
        self.Longitude = longitude
        self.Latitude = latitude