#### Functions to work out size and temperature dependency

import numpy as np

# B0 for uptake
def size_growth(B_g, Ma):
    B0_g = B_g * Ma**0.75
    return B0_g

# B0 for maintenance respiration
def size_resp(B_r, Ma):
    B0_r = B_r * Ma**0.75
    return B0_r

# Arrhenius/Sharpe-Schoolfield for maintenance growth
def temp_growth(k, T, Tref, T_pk,N, B_g, Ma, Ea_U, Ea_D):
    Sharpe = (size_growth(B_g, Ma) * np.exp((-Ea_U/k) * ((1/T)-(1/Tref)))/(1 + (Ea_U/(Ea_D - Ea_U)) * np.exp(Ea_D/k * (1/T_pk - 1/T))))
    return Sharpe

# Arrhenius/Sharpe-Schoolfield for maintenance respiration
def temp_resp(k, T, Tref, T_pk,N, B_r, Ma,  Ea_R, Ea_D):
    Sharpe = (size_resp(B_r, Ma) * np.exp((-Ea_R/k) * ((1/T)-(1/Tref)))/(1 + (Ea_R/(Ea_D - Ea_R)) * np.exp(Ea_D/k * (1/T_pk - 1/T))))
    return Sharpe
