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
    np.fill_diagonal(u1,u_temp) # fill in temp dependant uptake on diagonals
    #np.fill_diagonal(u1,1) # fill in temp dependant uptake on diagonals
    u2 = np.zeros([M,M])
    np.fill_diagonal(u2,u_temp[M:M*2]) # fill in temp dependant uptake on diagonals
    if N == M:
        U = u1
    else:
        U = np.concatenate((u1, u2), axis=0)  
   

    # Maintenance Respiration
    ar_rm = temp_resp(k, T, Tref,T_pk, N, B_rm, Ma, Ea, Ea_D) # find how varies with temperature (ar = arrhenius)
    #Rm = sc.full([N], (0.5))
    #Rm = np.array([0.4257905,0.4257905])
    Rm = ar_rm
    
    # Growth Respiration
    Rg = sc.full([M], (0))
    
    # Excretion
    l = np.zeros([M,M])
    for i in range(M-1): 
        l[i,i+1] =  0.4
    l[M-1,0] = 0.4
    p = np.concatenate((np.array([1]), np.repeat(1,M-1)))
    return U, Rm, Rg, l, p

# Model
def metabolic_model(pops,t):
    x = pops
    #print(x)

    Nt = np.array([])
    Mt = np.array([])
    
    for i in range(int(N)):
        #print(i)
        m = x[i] * Rm[i] # maintenance lost -> C * Rm
        et = 0
        gt = 0
        rgt = 0
        for j in range(int(M)):
            g = U[i,j] * x[i] * x[N+j] # growth -> U * C * S
            gt = gt + g
            rg = U[i,j] * x[i] * x[N+j] * Rg[j] # loss to growth -> U * C * S * Rg
            rgt = rgt + rg
            for k in range(int(M)):
                e = U[i,j] * x[i] * x[N+j] * l[j,k] # excretion -> U * C * S * l
                et = et + e     
        xt = gt - rgt - m - et # -> UCS - UCSRg - UCSl - CRm -> C(US(1-Rg-l) - Rm)
        Nt = np.append(Nt, xt)

    for a in range(int(M)):
        #print(a)
        xR = p[a] # external resrouces
        gRt = 0
        eRt = 0
        for b in range(int(N)):
            #print(b)
            gR = U[b,a] * x[b] * x[N+a] # loss of resources to growth -> U * C * S
            gRt = gRt + gR
            
            for c in range(int(M)):
                eR = U[b,c] * x[b] * x[N+c] * l[c,a] # gain of resources from excretion -> U * C * S * l
                eRt = eRt + eR
        rt = xR + eRt - gRt
        Mt = np.append(Mt, rt) 
    return np.array(np.concatenate((Nt, Mt)))

######### Main Code ###########

######## Set up parameters ###########

N = 10 # Number of species
M = 5 # Number of nutrients
K = 2 # number of species (* 100)
k = 0.0000862
Tref = 273.15 # 0 degrees C
pk = 20 # Peak above Tref ('low' = 12 and 'high' = 20)
T_pk = Tref + pk
B_g = 0.5 # B0 for growth
B_rm = 0.1 # B0 for respiration (maintence)
Ma = 1 # Mass
Ea_D = np.repeat(3.5,N) # Deactivation energy
t_fin = 100
t = sc.linspace(0,t_fin-1,t_fin)
x0 = np.concatenate((sc.full([N], (0.1)),sc.full([M], (1.0))))
T = 273.15+20
Ea = np.concatenate([np.repeat(0.9,3), np.repeat(0.7,5), np.repeat(0.8,2)])
U = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D)[0] # make less than 1
Rm = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D)[1] # make less than 1
Rg = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D)[2] # l + Rg must be less than 1
l = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D)[3]
p = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D)[4]
l_sum = np.sum(l, axis=1)

##### Intergrate system forward #####
pops = odeint(metabolic_model, y0=x0, t=t)
print(pops[t_fin-1,:])



#### Plot output ####
result_array = np.empty((0,N+M))     ### doesnt change each run, pops/result array not correct
t_plot = sc.linspace(0,len(result_array),len(result_array))
plt.plot(t, pops[:,0:N], 'g-', label = 'Consumers', linewidth=0.7)
plt.plot(t, pops[:,N:N+M], 'b-', label = 'Resources', linewidth=0.7)
plt.grid
plt.ylabel('Population density')
plt.xlabel('Time')
plt.title('Bacteria-Nutrients population dynamics')
plt.legend([Line2D([0], [0], color='green', lw=2), Line2D([0], [0], color='blue', lw=2)], ['Bacteria', 'Nutrients'])
plt.savefig('Figure_ein_exist.png')
plt.show()







# --------------------------------------------
#### Old code


# import scipy as sc
# import numpy as np
# from scipy.integrate import odeint
# import matplotlib.pylab as pie
# import time
# start_time = time.time()
# import matplotlib.pylab as plt
# from matplotlib.lines import Line2D


# N = 2
# M = 1
# t_fin = 100
# t = sc.linspace(0,t_fin-1,t_fin)
# x0 = np.concatenate((sc.full([N], (0.1)),sc.full([M], (1.0))))


# gt = 0
# rgt = 0
# et = 0
# gRt = 0
# eRt = 0


# def size_growth(B_g, Ma):
#     B0_g = B_g * Ma**0.75
#     return B0_g

# def size_resp(B_rm, Ma):
#     B0_rm = B_rm * Ma**0.75
#     return B0_rm

# def temp_growth(k, T, Tref, T_pk,N, B_g, Ma, Ea, Ea_D):
#     Sharpe = (size_growth(B_g, Ma) * np.exp((-Ea/k) * ((1/T)-(1/Tref))))#/(1 + (Ea/(Ea_D - Ea)) * np.exp(Ea_D/k * (1/T_pk - 1/T))))
#     return Sharpe

# def temp_resp(k, T, Tref, T_pk,N, B_rm, Ma, Ea, Ea_D):
#     Sharpe = (size_resp(B_rm, Ma) * np.exp((-Ea/k) * ((1/T)-(1/Tref))))#/(1 + (Ea/(Ea_D - Ea)) * np.exp(Ea_D/k * (1/T_pk - 1/T))))
#     return Sharpe

# N = 2 # Number of species
# M = 1 # Number of nutrients
# K = 2 # number of species (* 100)
# k = 0.0000862
# Tref = 273.15 # 0 degrees C
# pk = 20 # Peak above Tref ('low' = 12 and 'high' = 20)
# T_pk = Tref + pk
# B_g = 0.5 # B0 for growth
# B_rm = 0.1#np.array([0.5,0.8]) # B0 for respiration (maintence)
# Ma = 1 # Mass
# Ea_D = np.repeat(3.5,N) # Deactivation energy
# Rg = sc.full([M], (0.0))
# l = np.zeros([M,M])
# for i in range(M-1):
#     l[i,i+1] = 0.4
# l[M-1,0] = 0.4
# p = np.array([1,1]) #np.ones(M)
# u1 = np.zeros([M,M])
# #np.fill_diagonal(u1,1)
# T=293.15
# Ea=np.array([1.0, 1.0])
# u_temp = temp_growth(k, T, Tref, T_pk, N, B_g, Ma, Ea, Ea_D) # uptake rates and maintenance are temp-dependant
# u1 = u_temp.reshape(2,1) 
# #np.fill_diagonal(u1,u_temp) # fill in temp dependant uptake on diagonals
# #u2 = np.tile(u1, (N,1)) # replicate matrix out dependant to a large size
# #u1 = u2[0:N, 0:M]
# #Rm = sc.full([N], (0.5))
# ar_rm = temp_resp(k, T, Tref,T_pk, N, B_rm, Ma, Ea, Ea_D) # find how varies with temperature (ar = arrhenius)
# Rm = ar_rm
   

# pars = (N, M, u1, Rm, Rg, l, p)
# def metabolic_model(pops,N, M, u1, Rm, Rg, l, p, t):
#     k = 0.0000862
#     Tref = 273.15 # 0 degrees C
#     B_g = 0.5 # B0 for growth
#     B_rm = 0.1 #np.array([0.5,0.8]) # B0 for respiration (maintence)
#     Ma = 1 # Mass
#     Ea_D = np.repeat(3.5,N) # Deactivation energy
#     T = 293.15
#     Ea = np.array([1.0, 1.0]) 
#     N = 2
#     M = 1
#     T_pk = 0
    
#     def size_growth(B_g, Ma):
#         B0_g = B_g * Ma**0.75
#         return B0_g

#     def size_resp(B_rm, Ma):
#         B0_rm = B_rm * Ma**0.75
#         return B0_rm

#     def temp_growth(k, T, Tref, T_pk,N, B_g, Ma, Ea, Ea_D):
#         Sharpe = (size_growth(B_g, Ma) * np.exp((-Ea/k) * ((1/T)-(1/Tref))))#/(1 + (Ea/(Ea_D - Ea)) * np.exp(Ea_D/k * (1/T_pk - 1/T))))
#         return Sharpe

#     def temp_resp(k, T, Tref, T_pk,N, B_rm, Ma, Ea, Ea_D):
#         Sharpe = (size_resp(B_rm, Ma) * np.exp((-Ea/k) * ((1/T)-(1/Tref))))#/(1 + (Ea/(Ea_D - Ea)) * np.exp(Ea_D/k * (1/T_pk - 1/T))))
#         return Sharpe

#     u1 = np.zeros([M,M])
#     #np.fill_diagonal(u1,1)
#     u_temp = temp_growth(k, T, Tref, T_pk, N, B_g, Ma, Ea, Ea_D) # uptake rates and maintenance are temp-dependant
#     u1 = u_temp.reshape(2,1)
#     #np.fill_diagonal(u1,u_temp) # fill in temp dependant uptake on diagonals
#     #u2 = np.tile(u1, (N,1)) # replicate matrix out dependant to a large size
#     #u1 = u2[0:N, 0:M]
#     #Rm = sc.full([N], (0.5))
#     ar_rm = temp_resp(k, T, Tref,T_pk, N, B_rm, Ma, Ea, Ea_D) # find how varies with temperature (ar = arrhenius)
#     Rm = ar_rm
#     Rg = sc.full([M], (0.0))
#     l = np.zeros([M,M])
#     for i in range(M-1):
#         l[i,i+1] = 0.4
#     l[M-1,0] = 0.4
#     p = np.array([1,1]) #np.ones(M)
#     t_fin = 100
#     t = sc.linspace(0,t_fin-1,t_fin)
#     x0 = np.concatenate((sc.full([N], (0.1)),sc.full([M], (1.0))))

#     Nt = np.array([])
#     Mt = np.array([])

#     x = pops
#     #print(x)
#     for i in range(int(N)):
#         #print(i)
#         m = x[i] * Rm[i] # maintenance lost -> C * Rm
#         et = 0
#         gt = 0
#         rgt = 0
#         for j in range(int(M)):
#             g = u1[i,j] * x[i] * x[N+j] # growth -> U * C * S
#             gt = gt + g
#             rg = u1[i,j] * x[i] * x[N+j] * Rg[j] # loss to growth -> U * C * S * Rg
#             rgt = rgt + rg
#             for k in range(int(M)):
#                 e = u1[i,j] * x[i] * x[N+j] * l[j,k] # excretion -> U * C * S * l
#                 et = et + e     
#         xt = gt - rgt - m - et # -> UCS - UCSRg - UCSl - CRm -> C(US(1-Rg-l) - Rm)
#         Nt = np.append(Nt, xt)

#     for a in range(int(M)):
#         #print(a)
#         xR = p[a] # external resrouces
#         gRt = 0
#         eRt = 0
#         for b in range(int(N)):
#             #print(b)
#             gR = u1[b,a] * x[b] * x[N+a] # loss of resources to growth -> U * C * S
#             gRt = gRt + gR
            
#             for c in range(int(M)):
#                 eR = u1[b,c] * x[b] * x[N+c] * l[c,a] # gain of resources from excretion -> U * C * S * l
#                 eRt = eRt + eR
#         rt = xR + eRt - gRt
#         Mt = np.append(Mt, rt) 
#     return np.array(np.concatenate((Nt, Mt)))

# pops = odeint(metabolic_model, y0=x0, t=t, args=pars)

# #print("--- %s seconds ---" % (time.time() - start_time))
# print(pops[t_fin-1,:])

# t_plot = sc.linspace(0,len(pops),len(pops))
# #plt.plot(t_plot, result_array[:,0:N], 'g-', label = 'Consumers', linewidth=0.7)
# #plt.plot(t_plot, result_array[:,N:N+M], 'b-', label = 'Resources', linewidth=0.7)
# plt.plot(t, pops[:,0:N], 'g-', label = 'Consumers', linewidth=0.7)
# plt.plot(t, pops[:,N:N+M], 'b-', label = 'Resources', linewidth=0.7)
# plt.grid
# plt.ylabel('Population density')
# plt.xlabel('Time')
# plt.title('Bacteria-Nutrients population dynamics')
# plt.legend([Line2D([0], [0], color='green', lw=2), Line2D([0], [0], color='blue', lw=2)], ['Bacteria', 'Nutrients'])
# #plt.legend([Line2D([0], [0], color='green', lw=2)], ['Consumers'])
# plt.savefig('Figure_ein_exist.png')
# plt.show()

# pie.plot(t, pops[:,0:100], 'g-', label = 'Resources')
# pie.plot(t, pops[:,101:150], 'b-', label = 'Consumers')
# pie.grid
# pie.xlabel('Time')
# pie.ylabel('Population density')
# pie.title('Consumer-Resource population dynamics')
# pie.show()
