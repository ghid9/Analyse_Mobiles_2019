# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:18:36 2020

@author: Ghid, with contributions from Mailys Chanial
"""
import os
from datetime import timedelta
import numpy as np
import pandas as pd


def utci_raw(Ta, va, D_Tmrt, ehPa):
    """Computation of UTCI
    UTCI is computed with the fast calculation method ,  6th degree polynomial expression
    Ta: Air Temperature
    va: wind speed at 10 meter height
    Tmrt: mean radiant temperature
    D_Tmrt: Tmrt-Ta
    ehPa: water vapour pressure
     """
    Pa = ehPa / 10  # Pression de vapeur hPa => Pa
    utci_polynomial = Ta + 6.07562052E-01 + (-2.27712343E-02) * Ta + 8.06470249E-04 * Ta * Ta + (
        -1.54271372E-04) * Ta * Ta * Ta + (-3.24651735E-06) * Ta * Ta * Ta * Ta + (
                          7.32602852E-08) * Ta * Ta * Ta * Ta * Ta + 1.35959073E-09 * Ta * Ta * Ta * Ta * Ta * Ta + (
                          -2.25836520E+00) * va + 8.80326035E-02 * Ta * va + 2.16844454E-03 * Ta * Ta * va + (
                          -1.53347087E-05) * Ta * Ta * Ta * va + (-5.72983704E-07) * Ta * Ta * Ta * Ta * va + (
                          -2.55090145E-09) * Ta * Ta * Ta * Ta * Ta * va + (-7.51269505E-01) * va * va + (
                          -4.08350271E-03) * Ta * va * va + (-5.21670675E-05) * Ta * Ta * va * va + (
                          1.94544667E-06) * Ta * Ta * Ta * va * va + 1.14099531E-08 * Ta * Ta * Ta * Ta * va * va + (
                          1.58137256E-01) * va * va * va + (-6.57263143E-05) * Ta * va * va * va + (
                          2.22697524E-07) * Ta * Ta * va * va * va + (-4.16117031E-08) * Ta * Ta * Ta * va * va * va + (
                          -1.27762753E-02) * va * va * va * va + 9.66891875E-06 * Ta * va * va * va * va + (
                          2.52785852E-09) * Ta * Ta * va * va * va * va + 4.56306672E-04 * va * va * va * va * va + (
                          -1.74202546E-07) * Ta * va * va * va * va * va + (
                          -5.91491269E-06) * va * va * va * va * va * va + (
                          3.98374029E-01) * D_Tmrt + 1.83945314E-04 * Ta * D_Tmrt + (
                          -1.73754510E-04) * Ta * Ta * D_Tmrt + (
                          -7.60781159E-07) * Ta * Ta * Ta * D_Tmrt + 3.77830287E-08 * Ta * Ta * Ta * Ta * D_Tmrt + (
                          5.43079673E-10) * Ta * Ta * Ta * Ta * Ta * D_Tmrt + (-2.00518269E-02) * va * D_Tmrt + (
                          8.92859837E-04) * Ta * va * D_Tmrt + 3.45433048E-06 * Ta * Ta * va * D_Tmrt + (
                          -3.77925774E-07) * Ta * Ta * Ta * va * D_Tmrt + (
                          -1.69699377E-09) * Ta * Ta * Ta * Ta * va * D_Tmrt + (
                          1.69992415E-04) * va * va * D_Tmrt + (-4.99204314E-05) * Ta * va * va * D_Tmrt + (
                          2.47417178E-07) * Ta * Ta * va * va * D_Tmrt + 1.07596466E-08 * Ta * Ta * Ta * va * va * D_Tmrt + (
                          8.49242932E-05) * va * va * va * D_Tmrt + 1.35191328E-06 * Ta * va * va * va * D_Tmrt + (
                          -6.21531254E-09) * Ta * Ta * va * va * va * D_Tmrt + (
                          -4.99410301E-06) * va * va * va * va * D_Tmrt + (
                          -1.89489258E-08) * Ta * va * va * va * va * D_Tmrt + (
                          8.15300114E-08) * va * va * va * va * va * D_Tmrt + 7.55043090E-04 * D_Tmrt * D_Tmrt + (
                          -5.65095215E-05) * Ta * D_Tmrt * D_Tmrt + (-4.52166564E-07) * Ta * Ta * D_Tmrt * D_Tmrt + (
                          2.46688878E-08) * Ta * Ta * Ta * D_Tmrt * D_Tmrt + (
                          2.42674348E-10) * Ta * Ta * Ta * Ta * D_Tmrt * D_Tmrt + 1.54547250E-04 * va * D_Tmrt * D_Tmrt + (
                          5.24110970E-06) * Ta * va * D_Tmrt * D_Tmrt + (
                          -8.75874982E-08) * Ta * Ta * va * D_Tmrt * D_Tmrt + (
                          -1.50743064E-09) * Ta * Ta * Ta * va * D_Tmrt * D_Tmrt + (
                          -1.56236307E-05) * va * va * D_Tmrt * D_Tmrt + (
                          -1.33895614E-07) * Ta * va * va * D_Tmrt * D_Tmrt + (
                          2.49709824E-09) * Ta * Ta * va * va * D_Tmrt * D_Tmrt + (
                          6.51711721E-07) * va * va * va * D_Tmrt * D_Tmrt + (
                          1.94960053E-09) * Ta * va * va * va * D_Tmrt * D_Tmrt + (
                          -1.00361113E-08) * va * va * va * va * D_Tmrt * D_Tmrt + (
                          -1.21206673E-05) * D_Tmrt * D_Tmrt * D_Tmrt + (
                          -2.18203660E-07) * Ta * D_Tmrt * D_Tmrt * D_Tmrt + (
                          7.51269482E-09) * Ta * Ta * D_Tmrt * D_Tmrt * D_Tmrt + (
                          9.79063848E-11) * Ta * Ta * Ta * D_Tmrt * D_Tmrt * D_Tmrt + (
                          1.25006734E-06) * va * D_Tmrt * D_Tmrt * D_Tmrt + (
                          -1.81584736E-09) * Ta * va * D_Tmrt * D_Tmrt * D_Tmrt + (
                          -3.52197671E-10) * Ta * Ta * va * D_Tmrt * D_Tmrt * D_Tmrt + (
                          -3.36514630E-08) * va * va * D_Tmrt * D_Tmrt * D_Tmrt + (
                          1.35908359E-10) * Ta * va * va * D_Tmrt * D_Tmrt * D_Tmrt + (
                          4.17032620E-10) * va * va * va * D_Tmrt * D_Tmrt * D_Tmrt + (
                          -1.30369025E-09) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + (
                          4.13908461E-10) * Ta * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + (
                          9.22652254E-12) * Ta * Ta * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + (
                          -5.08220384E-09) * va * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + (
                          -2.24730961E-11) * Ta * va * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + (
                          1.17139133E-10) * va * va * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + (
                          6.62154879E-10) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + (
                          4.03863260E-13) * Ta * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + (
                          1.95087203E-12) * va * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + (
                          -4.73602469E-12) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + 5.12733497E+00 * Pa + (
                          -3.12788561E-01) * Ta * Pa + (
                          -1.96701861E-02) * Ta * Ta * Pa + 9.99690870E-04 * Ta * Ta * Ta * Pa + (
                          9.51738512E-06) * Ta * Ta * Ta * Ta * Pa + (-4.66426341E-07) * Ta * Ta * Ta * Ta * Ta * Pa + (
                          5.48050612E-01) * va * Pa + (-3.30552823E-03) * Ta * va * Pa + (
                          -1.64119440E-03) * Ta * Ta * va * Pa + (
                          -5.16670694E-06) * Ta * Ta * Ta * va * Pa + 9.52692432E-07 * Ta * Ta * Ta * Ta * va * Pa + (
                          -4.29223622E-02) * va * va * Pa + 5.00845667E-03 * Ta * va * va * Pa + (
                          1.00601257E-06) * Ta * Ta * va * va * Pa + (-1.81748644E-06) * Ta * Ta * Ta * va * va * Pa + (
                          -1.25813502E-03) * va * va * va * Pa + (-1.79330391E-04) * Ta * va * va * va * Pa + (
                          2.34994441E-06) * Ta * Ta * va * va * va * Pa + 1.29735808E-04 * va * va * va * va * Pa + (
                          1.29064870E-06) * Ta * va * va * va * va * Pa + (
                          -2.28558686E-06) * va * va * va * va * va * Pa + (
                          -3.69476348E-02) * D_Tmrt * Pa + 1.62325322E-03 * Ta * D_Tmrt * Pa + (
                          -3.14279680E-05) * Ta * Ta * D_Tmrt * Pa + 2.59835559E-06 * Ta * Ta * Ta * D_Tmrt * Pa + (
                          -4.77136523E-08) * Ta * Ta * Ta * Ta * D_Tmrt * Pa + 8.64203390E-03 * va * D_Tmrt * Pa + (
                          -6.87405181E-04) * Ta * va * D_Tmrt * Pa + (-9.13863872E-06) * Ta * Ta * va * D_Tmrt * Pa + (
                          5.15916806E-07) * Ta * Ta * Ta * va * D_Tmrt * Pa + (
                          -3.59217476E-05) * va * va * D_Tmrt * Pa + (
                          3.28696511E-05) * Ta * va * va * D_Tmrt * Pa + (
                          -7.10542454E-07) * Ta * Ta * va * va * D_Tmrt * Pa + (
                          -1.24382300E-05) * va * va * va * D_Tmrt * Pa + (
                          -7.38584400E-09) * Ta * va * va * va * D_Tmrt * Pa + (
                          2.20609296E-07) * va * va * va * va * D_Tmrt * Pa + (
                          -7.32469180E-04) * D_Tmrt * D_Tmrt * Pa + (
                          -1.87381964E-05) * Ta * D_Tmrt * D_Tmrt * Pa + 4.80925239E-06 * Ta * Ta * D_Tmrt * D_Tmrt * Pa + (
                          -8.75492040E-08) * Ta * Ta * Ta * D_Tmrt * D_Tmrt * Pa + 2.77862930E-05 * va * D_Tmrt * D_Tmrt * Pa + (
                          -5.06004592E-06) * Ta * va * D_Tmrt * D_Tmrt * Pa + (
                          1.14325367E-07) * Ta * Ta * va * D_Tmrt * D_Tmrt * Pa + (
                          2.53016723E-06) * va * va * D_Tmrt * D_Tmrt * Pa + (
                          -1.72857035E-08) * Ta * va * va * D_Tmrt * D_Tmrt * Pa + (
                          -3.95079398E-08) * va * va * va * D_Tmrt * D_Tmrt * Pa + (
                          -3.59413173E-07) * D_Tmrt * D_Tmrt * D_Tmrt * Pa + (
                          7.04388046E-07) * Ta * D_Tmrt * D_Tmrt * D_Tmrt * Pa + (
                          -1.89309167E-08) * Ta * Ta * D_Tmrt * D_Tmrt * D_Tmrt * Pa + (
                          -4.79768731E-07) * va * D_Tmrt * D_Tmrt * D_Tmrt * Pa + (
                          7.96079978E-09) * Ta * va * D_Tmrt * D_Tmrt * D_Tmrt * Pa + (
                          1.62897058E-09) * va * va * D_Tmrt * D_Tmrt * D_Tmrt * Pa + (
                          3.94367674E-08) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * Pa + (
                          -1.18566247E-09) * Ta * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * Pa + (
                          3.34678041E-10) * va * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * Pa + (
                          -1.15606447E-10) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * Pa + (
                          -2.80626406E+00) * Pa * Pa + (
                          5.48712484E-01) * Ta * Pa * Pa + (-3.99428410E-03) * Ta * Ta * Pa * Pa + (
                          -9.54009191E-04) * Ta * Ta * Ta * Pa * Pa + 1.93090978E-05 * Ta * Ta * Ta * Ta * Pa * Pa + (
                          -3.08806365E-01) * va * Pa * Pa + 1.16952364E-02 * Ta * va * Pa * Pa + (
                          4.95271903E-04) * Ta * Ta * va * Pa * Pa + (-1.90710882E-05) * Ta * Ta * Ta * va * Pa * Pa + (
                          2.10787756E-03) * va * va * Pa * Pa + (-6.98445738E-04) * Ta * va * va * Pa * Pa + (
                          2.30109073E-05) * Ta * Ta * va * va * Pa * Pa + 4.17856590E-04 * va * va * va * Pa * Pa + (
                          -1.27043871E-05) * Ta * va * va * va * Pa * Pa + (
                          -3.04620472E-06) * va * va * va * va * Pa * Pa + (
                          5.14507424E-02) * D_Tmrt * Pa * Pa + (-4.32510997E-03) * Ta * D_Tmrt * Pa * Pa + (
                          8.99281156E-05) * Ta * Ta * D_Tmrt * Pa * Pa + (
                          -7.14663943E-07) * Ta * Ta * Ta * D_Tmrt * Pa * Pa + (
                          -2.66016305E-04) * va * D_Tmrt * Pa * Pa + 2.63789586E-04 * Ta * va * D_Tmrt * Pa * Pa + (
                          -7.01199003E-06) * Ta * Ta * va * D_Tmrt * Pa * Pa + (
                          -1.06823306E-04) * va * va * D_Tmrt * Pa * Pa + (
                          3.61341136E-06) * Ta * va * va * D_Tmrt * Pa * Pa + (
                          2.29748967E-07) * va * va * va * D_Tmrt * Pa * Pa + 3.04788893E-04 * D_Tmrt * D_Tmrt * Pa * Pa + (
                          -6.42070836E-05) * Ta * D_Tmrt * D_Tmrt * Pa * Pa + (
                          1.16257971E-06) * Ta * Ta * D_Tmrt * D_Tmrt * Pa * Pa + (
                          7.68023384E-06) * va * D_Tmrt * D_Tmrt * Pa * Pa + (
                          -5.47446896E-07) * Ta * va * D_Tmrt * D_Tmrt * Pa * Pa + (
                          -3.59937910E-08) * va * va * D_Tmrt * D_Tmrt * Pa * Pa + (
                          -4.36497725E-06) * D_Tmrt * D_Tmrt * D_Tmrt * Pa * Pa + (
                          1.68737969E-07) * Ta * D_Tmrt * D_Tmrt * D_Tmrt * Pa * Pa + (
                          2.67489271E-08) * va * D_Tmrt * D_Tmrt * D_Tmrt * Pa * Pa + (
                          3.23926897E-09) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * Pa * Pa + (
                          -3.53874123E-02) * Pa * Pa * Pa + (
                          -2.21201190E-01) * Ta * Pa * Pa * Pa + 1.55126038E-02 * Ta * Ta * Pa * Pa * Pa + (
                          -2.63917279E-04) * Ta * Ta * Ta * Pa * Pa * Pa + 4.53433455E-02 * va * Pa * Pa * Pa + (
                          -4.32943862E-03) * Ta * va * Pa * Pa * Pa + 1.45389826E-04 * Ta * Ta * va * Pa * Pa * Pa + (
                          2.17508610E-04) * va * va * Pa * Pa * Pa + (-6.66724702E-05) * Ta * va * va * Pa * Pa * Pa + (
                          3.33217140E-05) * va * va * va * Pa * Pa * Pa + (-2.26921615E-03) * D_Tmrt * Pa * Pa * Pa + (
                          3.80261982E-04) * Ta * D_Tmrt * Pa * Pa * Pa + (
                          -5.45314314E-09) * Ta * Ta * D_Tmrt * Pa * Pa * Pa + (
                          -7.96355448E-04) * va * D_Tmrt * Pa * Pa * Pa + 2.53458034E-05 * Ta * va * D_Tmrt * Pa * Pa * Pa + (
                          -6.31223658E-06) * va * va * D_Tmrt * Pa * Pa * Pa + (
                          3.02122035E-04) * D_Tmrt * D_Tmrt * Pa * Pa * Pa + (
                          -4.77403547E-06) * Ta * D_Tmrt * D_Tmrt * Pa * Pa * Pa + (
                          1.73825715E-06) * va * D_Tmrt * D_Tmrt * Pa * Pa * Pa + (
                          -4.09087898E-07) * D_Tmrt * D_Tmrt * D_Tmrt * Pa * Pa * Pa + 6.14155345E-01 * Pa * Pa * Pa * Pa + (
                          -6.16755931E-02) * Ta * Pa * Pa * Pa * Pa + 1.33374846E-03 * Ta * Ta * Pa * Pa * Pa * Pa + (
                          3.55375387E-03) * va * Pa * Pa * Pa * Pa + (-5.13027851E-04) * Ta * va * Pa * Pa * Pa * Pa + (
                          1.02449757E-04) * va * va * Pa * Pa * Pa * Pa + (
                          -1.48526421E-03) * D_Tmrt * Pa * Pa * Pa * Pa + (
                          -4.11469183E-05) * Ta * D_Tmrt * Pa * Pa * Pa * Pa + (
                          -6.80434415E-06) * va * D_Tmrt * Pa * Pa * Pa * Pa + (
                          -9.77675906E-06) * D_Tmrt * D_Tmrt * Pa * Pa * Pa * Pa + 8.82773108E-02 * Pa * Pa * Pa * Pa * Pa + (
                          -3.01859306E-03) * Ta * Pa * Pa * Pa * Pa * Pa + 1.04452989E-03 * va * Pa * Pa * Pa * Pa * Pa + (
                          2.47090539E-04) * D_Tmrt * Pa * Pa * Pa * Pa * Pa + 1.48348065E-03 * Pa * Pa * Pa * Pa * Pa * Pa
    return utci_polynomial


def dateparse(x):
    """dateparse function
    applies the crrect format to imported data
    """
    return pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


# Mean radiant temperature calculation
def Tmrt_ISO(Tg, va, epsilon, Diameter, Ta):
    """ISO 7726
    We do not apply a correction to air temperature while computing Tmrt
Va: wind speed at 1,5 meter height
Tg:globe temperature
epsilon: emissivity (  0.95 for the globe )
Diameter: black globe diameter
Ta: air temperature
"""
    A = (Tg + 273.15) ** 4
    B = (1.1 * 10 ** 8) * va ** 0.6
    C = epsilon * Diameter ** 0.4
    D = (Tg - Ta)
    Tmrt = ((A + (B / C) * D) ** 0.25) - 273.15
    return Tmrt


# Black Globe Data

emissivity_globe = 0.95
Diametre_globe = 150 * (10 ** -3)


def pression_vap_sat(Ta):
    """
    Equilibrium water vapour pressure
    """
    g = [-2.8365744E3, -6.028076559E3, 1.954263612E1, -2.737830188E-2, 1.6261698E-5, 7.0229056E-10, -1.8680009E-13,
         2.7150305]
    tk = Ta + 273.15  # air temp in K
    es = g[7] * np.log(tk)

    for k in range(len(g) - 1):
        es = es + g[k] * tk ** (k - 2)

    es = np.exp(es) * 0.01  # *0.01: convert Pa to hPa
    return es


def utci(Tg, Ta, va, RH, h):
    """
    Utci: fast calculation computation
    Tg:  globe temperature
    Ta: air temperature
    va: wind speed
    RH: relative humidity
    h: height at which wind speed measurement was maid
    """
    # Adjusted wind speed
    h_factor = np.log10(10 / 0.01) / np.log10(h / 0.01)
    va_10m = va * h_factor
    # Mean radiant temperature with black globe
    TMRT = Tmrt_ISO(Tg, va, emissivity_globe, Diametre_globe, Ta)
    D_Tmrt = TMRT - Ta
    # Water Vapour Pressure
    es = pression_vap_sat(Ta)
    RH = RH / 100
    ehPa = RH * es
    # Compute UTCI
    utci_calc = utci_raw(Ta, va_10m, D_Tmrt, ehPa)
    utci_final = round(utci_calc, 1)

    return utci_final


# %% OASIS DICTIONARY

"""DATA"""
""""dictionnary to access data files """
dict_path_oasis = {'emeriau': r"Emeriau",
                   'JeanDolent': r"JeanDolent", 'JeanneDarc': r'JeanneDArc', 'keller': r'Keller',
                   'MHilszEE': r'MaryseHilszEE', 'MHilszEM': r'MaryseHilszEM', 'QuatreFils': r'QuatreFils',
                   'OGreard': r'OctaveGreard', 'Palviset': r'PierreAlviset', 'Tandou': r'Tandou'}
# OASIS SCHOOLS
Lecoles = list(dict_path_oasis.keys())

# %% REFERENCE STATIONS

"THe public space station is used as the reference station for calculation"

liste = os.listdir(dict_path_oasis['Palviset'])
data_fixe = pd.read_csv(dict_path_oasis['Palviset'] + "\\PalvisetEP.txt",
                        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                        sep=";", date_parser=dateparse, skiprows=4,
                        names=['DateTime', 'TensAlim', 'Tair4m', 'RH', 'T_air', 'NetRad', 'Tg', 'va', 'DirV'])

# DATETIME FOR UPSAMPLING
# first make sure the date format is datetime
if type(data_fixe.DateTime[0]):
    data_fixe['Datetime'] = pd.to_datetime(data_fixe.DateTime, yearfirst=True)
else:
    pass
# set datetime index for upsampling
data_fixe = data_fixe.set_index('Datetime')
# Resample data
data_fixe_resample = data_fixe.resample('15S')
# fill NaN
data_fixe_upsampled = data_fixe_resample.interpolate()

# %% CREATE SYNCHORNISATION CORRECTION

# Create delta Tair et delta RH
data_fixe_upsampled = data_fixe_upsampled.assign(D_Tair=data_fixe_upsampled['T_air'] - data_fixe_upsampled.T_air[0],
                                                 D_RH=data_fixe_upsampled['RH'] - data_fixe_upsampled.Tg[0])
#  Short Term correction : Air Temperature and Relative Humidity
Correction = data_fixe_upsampled[['D_Tair', 'D_RH']]

# %% MOBILE MEASUREMENTS
# choose measurements site
print('Veuillez entrer Le nom de la cour')
ecole = input()
print("le nom de la cours est : " + str(ecole))

# GENERATE TABLE CONTAINING VALUES AVERAGED ON LAST 5 MINUTES OF THE MEASUREMENT

# UTCI TABLE FORMAT FOR MOBILE MEASUREMENTS
UTCI_TABLE = {"Ecole": [],
      "Point": [],
      "Taircor": [],
      "Tg": [],
      "RH": [],
      "RHcor": [],
      "Tair": [],
      "Va": [],
      "vapSat": [],
      "Tmrt": [],
      "UTCI": [],
      }

UTCI_Df= pd.DataFrame(UTCI_TABLE)

# MEASUREMENTS LOOP

liste = os.listdir(dict_path_oasis['Palviset'])
i = 0
while i < len(liste) - 1:
    data_m = pd.read_csv(dict_path_oasis['Palviset'] + '//' + liste[i], sep=";", usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                         header=None, skiprows=1, date_parser=dateparse)
    data_point = data_m.dropna() #remove no data
    data_point.columns = ["DateTime", "Tnw", "Tg", "T_air", "P", "RH", "Vent", "Tm,r ", "WBGT", "WCI"]

    # Synchronise start of measurements for mobile and fixed spots
    data_point['DateTime'] = pd.to_datetime( data_point['DateTime'])
    data_point['NDateTime'] = data_point['DateTime'] + timedelta(seconds=-data_point['DateTime'][0].second)
    data_sync = data_point.drop(['Tnw', 'WBGT', 'WCI'], axis=1)
    Data_sync= data_point.set_index('NDateTime')  # set datetime as index

    #Place the Fixed and  mobile DataFrames side by side
    # Sync_Table: table for synchronizing measurements
    TabCor = pd.concat([Correction, Data_sync], axis=1)
    TabCor.dropna()
    # Ceate a synchronised mobile measurements dataframe
    Mobile_sync= TabCor.assign(Tcor_air=TabCor.T_air + TabCor.D_Tair, RHcor=TabCor.RH + TAC.RH)
    #Compute the averaged Values
    Tairsync = Mobile_sync.Tcor_air.tail(21).mean()
    Tg = Mobile_sync.Tg.tail(21).mean()
    RH = Mobile_sync.RH.tail(21).mean()
    RHsync = MC.RH.tail(21).mean()
    Tair = MC.T_air.tail(21).mean()
    wind_speed = MC.Vent.tail(21).mean()
    Tmrt= Tmrt_ISO(Tg, wind_speed, emissivity_globe, Diametre_globe, Tair)
    vap_pressure= pression_vap_sat(Tairsync)
    UUtci = utci(TgMoy, TcorAirMoy, vitv, RHcorMoy)
    PU.append([ecole, i, TcorAirMoy, TgMoy, RHcorMoy, RHMoy, T_airMoy, vitv, UVapsat, UTmrt, UUtci])

    print("point" + " " + str(i + 1) + " " + "OK")
    i = i + 1

"""Tableau UTCI pour les points de chaque balade dans chaque école"""
# Add to dataframe les lignes relatives à chaque point

DPU = pd.DataFrame(PU, columns=TU)
DU = DU.append(DPU)

"""CONSIGNER LES VALEURS"""
outfile = open(dict_path['Palviset'] + '/Palviset_UTCI.csv', 'w')
DU.to_csv(outfile, sep=';', index=False)
outfile.close()
print("valeurs Consignées")

"""Graphe pour visualiser la station fixe, les mesures mobiles et les mesures mobiles corrigées"""

# MULTIPLE PLOTS SAME FIGURE
# plt.plot(SwDown18, SwUp18, "o", label='Station Fixe')
# plt.plot(SwDown19, SwUp19, "o",label='Après travaux ')
# plt.legend(loc='best')
# plt.title("Evolution de l'albédo= Rayon réfléchi/Rayon incident")
# plt.show()
