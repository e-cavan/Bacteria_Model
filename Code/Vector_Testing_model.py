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
    B0_g = B_g * Ma**-0.25
    return B0_g

# B0 for maintenance respiration
def size_resp(B_rm, Ma):
    B0_rm = B_rm * Ma**-0.25
    return B0_rm

# Arrhenius/Sharpe-Schoolfield for uptake
def temp_growth(k, T, Tref, T_pk,N, B_g, Ma, Ea_U, Ea_D):
    Sharpe = (size_growth(B_g, Ma) * np.exp((-Ea_U/k) * ((1/T)-(1/Tref))))#/(1 + (Ea/(Ea_D - Ea)) * np.exp(Ea_D/k * (1/T_pk - 1/T))))
    return Sharpe

# Arrhenius/Sharpe-Schoolfield for maintenance respiration
def temp_resp(k, T, Tref, T_pk,N, B_rm, Ma, Ea_R, Ea_D):
    Sharpe = (size_resp(B_rm, Ma) * np.exp((-Ea_R/k) * ((1/T)-(1/Tref))))#/(1 + (Ea/(Ea_D - Ea)) * np.exp(Ea_D/k * (1/T_pk - 1/T))))
    return Sharpe

# Parameters
def params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea_U, Ea_R, Ea_D):
    #Uptake
    u1 = np.zeros([M,M]) # M M
    #u_temp = temp_growth(k, T, Tref, T_pk, N, B_g, Ma, Ea_U, Ea_D) # uptake rates and maintenance are temp-dependant
    u_temp = np.array([10, 20, 30])
    np.fill_diagonal(u1,u_temp) # fill in temp dependant uptake on diagonals
    #np.fill_diagonal(u1,1) # fill in temp dependant uptake on diagonals
    u2 = np.zeros([M,M])
    u3 = np.zeros([M,M])
    if N == M:
        U = u1
    elif M == N/2:
        np.fill_diagonal(u2,u_temp[M:M*2]) # fill in temp dependant uptake on diagonals
        U = np.concatenate((u1, u2), axis=0) 
    else:
        np.fill_diagonal(u2,u_temp[M:M*2]) # fill in temp dependant uptake on diagonals
        np.fill_diagonal(u3,u_temp[M+1:M*3])
        U = np.concatenate((u1, u2, u3), axis=0)  
    
    # Maintenance respiration
    ar_rm = temp_resp(k, T, Tref,T_pk, N, B_rm, Ma, Ea_R, Ea_D) # find how varies with temperature (ar = arrhenius)
    #Rm = sc.full([N], (0.5))
    Rm = ar_rm
    #Rm = np.array([1, 2])
    
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

def metabolic_model(pops,t, U, Rm, Rg, l, p, l_sum, Ea_U,Ea_, Ea_D, N, M, T, Tref, B_rm, B_g, Ma, k):
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
    dSdt = p - (np.multiply((xc @ U).transpose(), xr) - np.einsum('i,k,ik,kj->j', xc, xr, U, l))

    return np.array(np.concatenate((dCdt, dSdt)))

######### Main Code ###########

######## Set up parameters ###########

N = 3 # Number of species
M = 1 # Number of nutrients
K = 2 # number of species (* 100)
k = 0.0000862
Tref = 273.15 # 0 degrees C
pk = 20 # Peak above Tref ('low' = 12 and 'high' = 20)
T_pk = Tref + pk
B_g = 0.5 #np.concatenate([np.repeat(0.7,5), np.repeat(0.4,5)]) # B0 for growth
B_rm = 0.1 #np.concatenate([np.repeat(0.3,5), np.repeat(0.1,5)]) #0.1 # B0 for respiration (maintence)
#B_g = np.repeat(np.array([0.3, 0.4, 0.5, 0.6, 0.7]),Fg_num)
#B_rm = (0.5 * B_g) - 0.1
Ma = 1 # Mass
T = 273.15+20
#Ea_U = np.array([0.6, 0.8, 1.0])
#Ea_R = np.array([0.7,  0.9, 1.1]) # must be different ratio to change from Ea_U
Ea_U = np.concatenate([np.repeat(0.6,N/2), np.repeat(1.0,N/2)])
Ea_R = np.concatenate([np.repeat(0.6,N/2), np.repeat(1.0,N/2)]) # must be different ratio to change from Ea_U
Ea_D = np.repeat(3.5,N) # Deactivation energy
t_fin = 1000
t = sc.linspace(0,t_fin-1,t_fin)


##### Intergrate system forward #####
result_array = np.empty((0,N+M)) 
x0 = np.concatenate((sc.full([N], (0.1)),sc.full([M], (1.0))))
#x0 = np.array([(1.5),(0.4),(0.3)])

# plot change in ratio U/Rm vs time to equil
ratio = np.delete(sc.linspace(0.01,2,1000), np.array(sc.linspace(490, 510, 20)))
time_eq = np.array([])

for i in range(len(ratio)): 
    # Set up model
    U = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea_U, Ea_R, Ea_D)[0]
    #Rm = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea_U, Ea_R,Ea_D)[1]
    #  
    if M == N/2:
        Rm = np.array([1,2*ratio[i]])
    else:
        Rm = np.array([1,2*ratio[i], 3*ratio[i]])

    Rg = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea_U, Ea_R, Ea_D)[2] # l + Rg must be less than 1
    l = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea_U, Ea_R, Ea_D)[3]
    p = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea_U, Ea_R, Ea_D)[4]
    l_sum = np.sum(l, axis=1)

    # Run model
    pars = (U, Rm, Rg, l, p, l_sum, Ea_U, Ea_R, Ea_D, N, M, T, Tref, B_rm, B_g, Ma, k)
    pops = odeint(metabolic_model, y0=x0, t=t, args = (pars))
    pops = np.round(pops, 3)
    #print(pops[t_fin-1,:])
    #print(len(np.unique(pops, axis=0)))

#### outputs for time vs u/rm ratio plot
    #time_eq = np.append(time_eq, len(np.unique(pops, axis=0))) # how long to equil
    time_eq = np.append(time_eq, np.where(pops[:,:] == 0)[0][0]) # when a B goes to 0
plt.plot(ratio, time_eq, 'ro')
plt.grid
plt.xlabel('Ratio (U1/Rm1 : U2/Rm2)')
plt.ylabel('Timesteps to extinction of one bacteria')
#plt.scatter([ratio], [time_eq], color='blue')
plt.show()

### Plot output ####
# t_plot = sc.linspace(0,len(result_array),len(result_array))
# plt.plot(t, pops[:,0:N], 'g-', label = 'Consumers', linewidth=0.7)
# plt.plot(t, pops[:,N:N+M], 'b-', label = 'Resources', linewidth=0.7)
# plt.grid
# plt.ylabel('Population density')
# plt.xlabel('Time')
# plt.title('Bacteria-Substrate population dynamics')
# plt.legend([Line2D([0], [0], color='green', lw=2), Line2D([0], [0], color='blue', lw=2)], ['Bacteria', 'Substrate'])
# plt.savefig('Figure_ein_exist.png')
# plt.show()


# ---------------------------------------------------

### Findiong equil solutions using sympy 

# from sympy import *
# import scipy as sc
# u1, u2, S, Rg, Rm1, Rm2,a, C1, C2,rho, t = var("u1, u2, S, Rg, Rm1, Rm2,a, C1, C2, rho,t",real = True)

# dC1_dt = C1 * ((u1 * S * (1-Rg-a)) - Rm1)
# dC2_dt = C2 * ((u2 * S * (1-Rg-a)) - Rm2)
# dS_dt = rho - ((u1 * S * C1) - (u1 * S * C1 * a)) - ((u2 * S * C2) - (u2 * S * C2 * a))
# dC1_dt, dC2_dt, dS_dt
# C1_eqlb = Eq(dC1_dt, 0)
# C2_eqlb = Eq(dC2_dt, 0)
# S_eqlb = Eq(dS_dt, 0)
# C1_eqlb,C2_eqlb, S_eqlb
# C1_eqlb_sol = solve(C1_eqlb, S)
# C2_eqlb_sol = solve(C2_eqlb, S)
# S_eqlb_sol = solve(S_eqlb, C1)

### Phase plane analysis (2 species, 1 resource)

# import matplotlib.cm
# import pylab as p
# import matplotlib.cm
# from scipy import integrate


# u = 2.84
# a = 0.4
# Rm = 0.57
# row=1


# def dX_dt(X, t = 0):
#     dC = X[0] * ((u * X[2] * (1-a) - Rm))
#     dC1 = X[1] * ((u * X[2] * (1-a) - Rm))
#     dS = row - (X[0] * X[2] * u) + (X[0] * X[2] * u * a) - (X[0] * X[2] * u) + (X[0] * X[2] * u * a)
#     return np.array([dC, dC1, dS])
              
# values  = sc.linspace(0.3, 0.9, 5)                          # position of X0 between X_f0 and X_f1
# vcolors = p.cm.autumn_r(sc.linspace(0.3, 1., len(values)))  # colors for each trajectory

# X_f0 = np.array([     0. ,  0. , 0.])
# X_f1 = np.array([ row/(2*Rm), row/(2*Rm), -Rm/(u*(a-1))])

# t = sc.linspace(0, 15,  1000)

# f2 = p.figure()
# for v, col in zip(values, vcolors):
#     X0 = v * X_f1                               # starting point
#     X = integrate.odeint(dX_dt, X0, t)         # we don't need infodict here
#     p.plot( X[:,0], X[:,1], lw=3.5*v, color=col, label='X0=(%.1f, %.1f)' % ( X0[0], X0[1]) )

# ymax = p.ylim(ymin=0)[1]                        # get axis limits
# xmax = p.xlim(xmin=0)[1]
# nb_points   = 40

# x = sc.linspace(0, xmax, nb_points)
# y = sc.linspace(0, ymax, nb_points)
# z = sc.linspace(0, 1, nb_points)

# X1 , Y1  = np.meshgrid(x, y)                       # create a grid
# q = dX_dt([X1, Y1, z])  
# DX1, DY1 = q[0,:,:] , q[1,:,:]                      # compute growth rate on the gridt
# M = (np.hypot(DX1, DY1))                           # Norm of the growth rate 
# M[ M == 0] = 1.                                 # Avoid zero division errors 
# DX1 /= M                                        # Normalize each arrows
# DY1 /= M

# #-------------------------------------------------------
# # Drow direction fields, using matplotlib 's quiver function
# p.title('Trajectories and direction fields')
# Q = p.quiver(X1, Y1, DX1, DY1, M, pivot='mid', cmap=p.cm.jet)
# p.xlabel('Number of Bacteria 1')
# p.ylabel('Number of Bacteria 2')
# p.legend()
# p.grid()
# p.xlim(0, xmax)
# p.ylim(0, ymax)
# p.show()