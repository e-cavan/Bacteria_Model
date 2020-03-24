import scipy as sc
import numpy as np
from scipy.integrate import odeint
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
import sys
import importlib
import math
from sklearn.utils import shuffle
from random import randint 
import itertools

###### Functions ########

# B0 for uptake
def size_growth(B_g, Ma):
    B0_g = B_g * Ma**0.75
    return B0_g

# B0 for maintenance respiration
def size_resp(B_rm, Ma):
    B0_rm = B_rm * Ma**0.75
    return B0_rm

# Arrhenius/Sharpe-Schoolfield for uptake
def temp_growth(k, T, Tref, T_pk,N, B_g, Ma, Ea, Ea_D):
    Sharpe = (size_growth(B_g, Ma) * np.exp((-Ea/k) * ((1/T)-(1/Tref))))#/(1 + (Ea/(Ea_D - Ea)) * np.exp(Ea_D/k * (1/T_pk - 1/T))))
    return Sharpe

# Arrhenius/Sharpe-Schoolfield for maintenance respiration
def temp_resp(k, T, Tref, T_pk,N, B_rm, Ma, Ea, Ea_D):
    Sharpe = (size_resp(B_rm, Ma) * np.exp((-Ea/k) * ((1/T)-(1/Tref))))#/(1 + (Ea/(Ea_D - Ea)) * np.exp(Ea_D/k * (1/T_pk - 1/T))))
    return Sharpe

# Parameters
def params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D):
    #Uptake
    u1 = np.zeros([M,M]) # M M
    u_temp = temp_growth(k, T, Tref, T_pk, N, B_g, Ma, Ea, Ea_D) # uptake rates and maintenance are temp-dependant
    #u_temp = np.array([2.128953, 2.128953])
    np.fill_diagonal(u1,u_temp) # fill in temp dependant uptake on diagonals
    #np.fill_diagonal(u1,1) # fill in temp dependant uptake on diagonals
    u2 = np.zeros([M,M])
    if N == M:
        U = u1
    else:
        np.fill_diagonal(u2,u_temp[M:M*2]) # fill in temp dependant uptake on diagonals
        U = np.concatenate((u1, u2), axis=0)  

    # Maintenance respiration
    ar_rm = temp_resp(k, T, Tref,T_pk, N, B_rm, Ma, Ea, Ea_D) # find how varies with temperature (ar = arrhenius)
    #Rm = sc.full([N], (0.5))
    Rm = ar_rm
    
    # Growth respiration
    Rg = sc.full([M], (0))
    
    # Excretion
    l = np.zeros([M,M])
    for i in range(M-1): 
        l[i,i+1] =  0.4
    l[M-1,0] = 0.4

    # External resource input
    p = np.concatenate((np.array([1]), np.repeat(1,M-1))) #np.ones(M)

    return U, Rm, Rg, l, p

def metabolic_model(pops,t):
    x = pops

    xc =  x[0:N] # consumer
    xr =  x[N:N+M] # resources

    ## Consumers
    # calculate 'middle'/ growth - Rg - leakeage term
    SL = (1 - Rg - l_sum) * xr
    #uptake rate and maintenance
    C = (np.sum(SL * U, axis=1)) - Rm
    #dCdt
    dCdt = xc * C
    
    ## Resources
    dSdt = p - np.multiply((xc @ U).transpose(), xr) + np.einsum('i,k,ik,kj->j', xc, xr, U, l)

    return np.array(np.concatenate((dCdt, dSdt)))

######### Main Code ###########

######## Set up parameters ###########

N = 10 # Number of species
M = 10 # Number of nutrients
K = 2 # number of species (* 100)
k = 0.0000862
Tref = 273.15 # 0 degrees C
pk = 20 # Peak above Tref ('low' = 12 and 'high' = 20)
T_pk = Tref + pk
B_g = 0.5 # B0 for growth
B_rm = 0.1 # B0 for respiration (maintence)
#B_g = np.repeat(np.array([0.3, 0.4, 0.5, 0.6, 0.7]),Fg_num)
#B_rm = (0.5 * B_g) - 0.1
Ma = 1 # Mass
T = 273.15+20
Ea = np.concatenate([np.repeat(0.65,3), np.repeat(0.5,5), np.repeat(1.0,2)])
Ea_D = np.repeat(3.5,N) # Deactivation energy
t_fin = 100
t = sc.linspace(0,t_fin-1,t_fin)


##### Intergrate system forward #####
result_array = np.empty((0,N+M)) 
x0 = np.concatenate((sc.full([N], (0.1)),sc.full([M], (1.0))))

# Set up model
U = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D)[0]
Rm = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D)[1] 
Rg = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D)[2] # l + Rg must be less than 1
l = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D)[3]
p = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D)[4]
l_sum = np.sum(l, axis=1)

# Run model
pops = odeint(metabolic_model, y0=x0, t=t)
print(pops[t_fin-1,:])

#### Plot output ####
t_plot = sc.linspace(0,len(result_array),len(result_array))
plt.plot(t, pops[:,0:N], 'g-', label = 'Consumers', linewidth=0.7)
plt.plot(t, pops[:,N:N+M], 'b-', label = 'Resources', linewidth=0.7)
plt.grid
plt.ylabel('Population density')
plt.xlabel('Time')
plt.title('Bacteria-Nutrients population dynamics')
plt.legend([Line2D([0], [0], color='green', lw=2), Line2D([0], [0], color='blue', lw=2)], ['Bacteria', 'Nutrients'])
#plt.savefig('Figure_ein_exist.png')
plt.show()


### Solving using sympy 

from sympy import *
import scipy as sc
u1, u2, S, Rg, Rm1, Rm2, C1, C2,rho, t = var("u1, u2, S, Rg, Rm1, Rm2, C1, C2, rho,t",real = True)

dC1_dt = C1 * ((u1 * S * (1-Rg)) - Rm1)
dC2_dt = C2 * ((u2 * S * (1-Rg)) - Rm2)
dS_dt = rho - ((u1 * S * C1) + (u2 * S * C2))
dC1_dt, dC2_dt, dS_dt
C1_eqlb = Eq(dC1_dt, 0)
C2_eqlb = Eq(dC2_dt, 0)
S_eqlb = Eq(dS_dt, 0)
C1_eqlb,C2_eqlb, S_eqlb
C1_eqlb_sol = solve(C1_eqlb, S)
C2_eqlb_sol = solve(C2_eqlb, S)
S_eqlb_sol = solve(S_eqlb, C1)
print(C1_eqlb_sol);print(C2_eqlb_sol); print(S_eqlb_sol); print(solve((C1_eqlb,C2_eqlb, S_eqlb), C1,C2,S))