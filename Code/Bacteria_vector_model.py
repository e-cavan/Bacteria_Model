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

###### Functions ########

# B0 for uptake
def size_growth(B_g, Ma):
    B0_g = B_g * Ma**0.75
    return B0_g

# B0 for maintenance respiration
def size_resp(B_rm, Ma):
    B0_rm = B_rm * Ma**0.75
    return B0_rm

# Arrhenius/Sharpe-Schoolfield for maintenance growth
def temp_growth(k, T, Tref, T_pk,N, B_g, Ma, Ea, Ea_D):
    Sharpe = (size_growth(B_g, Ma) * np.exp((-Ea/k) * ((1/T)-(1/Tref)))/(1 + (Ea/(Ea_D - Ea)) * np.exp(Ea_D/k * (1/T_pk - 1/T))))
    return Sharpe

# Arrhenius/Sharpe-Schoolfield for maintenance respiration
def temp_resp(k, T, Tref, T_pk,N, B_rm, Ma, Ea, Ea_D):
    Sharpe = (size_resp(B_rm, Ma) * np.exp((-Ea/k) * ((1/T)-(1/Tref)))/(1 + (Ea/(Ea_D - Ea)) * np.exp(Ea_D/k * (1/T_pk - 1/T))))
    return Sharpe

# Parameters
def params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D):
    #Uptake
    u1 = np.zeros([M,M])
    u_temp = temp_growth(k, T, Tref, T_pk, N, B_g, Ma, Ea, Ea_D) # uptake rates and maintenance are temp-dependant
    np.fill_diagonal(u1,u_temp[0:M]) # fill in temp dependant uptake on diagonals
    u2 = np.zeros([M,M])
    np.fill_diagonal(u2,u_temp[M:M*2]) # fill in temp dependant uptake on diagonals
    if N == M:
        U = u1
    else:
        np.fill_diagonal(u2,u_temp[M:M*2]) # fill in temp dependant uptake on diagonals
        U = np.concatenate((u1, u2), axis=0)  
    
    # Maintenance respiration
    ar_rm = temp_resp(k, T, Tref,T_pk, N, B_rm, Ma, Ea, Ea_D) # find how varies with temperature (ar = arrhenius)
    #Rm = sc.full([N], (0.3))
    Rm = ar_rm
    
    # Growth respiration
    Rg = sc.full([M], (0))
    
    # Excretion
    l = np.zeros([M,M])
    for i in range(M-1): 
        l[i,i+1] =  0.4
    l[M-1,0] = 0.4

    # External resource input
    p = np.repeat(1, M) #np.ones(M) #np.concatenate((np.array([0.4]), np.repeat(0.0, M-1)))  #
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
K = 10 # number of species (* 100)
Fg = 5 # number of functional groups
Fg_num = N/Fg # number of species in each functional group
k = 0.0000862
Tref = 273.15 # 0 degrees C
pk = 20 # Peak above Tref ('low' = 12 and 'high' = 20)
T_pk = Tref + pk
B_g = 0.5 # B0 for growth
B_rm = 0.1 # B0 for respiration (maintence)
#B_g = np.repeat(np.array([0.3, 0.4, 0.5, 0.6, 0.7]),Fg_num)
#B_rm = (0.5 * B_g) - 0.1
Ma = 1 # Mass
Ea_D = np.repeat(3.5,N) # Deactivation energy
#temp = np.array([5, 10,15,20,25])
t_n = 5 # number of temperatures (0-10 inclusive)
t_n_output = (t_n * N) + N # number of rows in output array
rep = math.ceil(N/M)
ass = 3 # assembly number
t_fin = 200
t = sc.linspace(0,t_fin-1,t_fin)
x0 = np.concatenate((sc.full([N], (0.1)),sc.full([M], (1.0))))

##### Intergrate system forward #####

# for data output files
result_array = np.empty((0,N+M))    
row = np.repeat(sc.linspace(1,ass,ass), N) 
B_n = sc.linspace(1,N,N)
N_n = np.tile(sc.linspace(1,M,M), rep)
output_tmp_lrg = np.empty((0,10))

for i in range(4, t_n):
#for i in range(0, len(temp)):
#for i in range(1,3):

    T = 273.15 + i # temperature 
    Temp = np.repeat(T - 273.15, N)
    Ea_tmp2 = np.repeat(np.array([0.6, 0.7, 0.8, 0.9,1.0]),Fg_num) # Activation energy (same for g and rm)
    #Ea_tmp2 = np.repeat(np.array([0.7, 0.7, 0.7, 0.7, 0.7]),Fg_num) # Activation energy (same for g and rm)
    Ea = shuffle(Ea_tmp2, random_state=0)        

    # Varying B0
    #B_g = np.exp(2.5 + (-5 * Ea))
    #B_rm = 0.5 *(np.exp(2.5 + (-5 * Ea))) #(0.5 * (B_g)) #

    output_tmp = np.empty((0,9))
    print('Temperature =',i)

    for j in range(ass):
        t_fin=200
        t = sc.linspace(0,t_fin-1,t_fin)
        Ea_1 = Ea # To fill output dataframe
        #B_g_1 = B_g # To fill output dataframe when B0 varies
        #B_rm_1 = B_rm # To fill output dataframe when B0 varies
        B_g_1 = np.repeat(B_g, N) # To fill output dataframe when B0 constant
        B_rm_1 = np.repeat(B_rm, N) # To fill output dataframe when B0 constant

        # Set up model
        U = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D)[0] # make less than 1
        Rm = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D)[1] # make less than 1
        Rg = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D)[2] # l + Rg must be less than 1
        l = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D)[3]
        p = params(N, M, T, k, Tref, T_pk, B_g, B_rm,Ma, Ea, Ea_D)[4]
        l_sum = np.sum(l, axis=1)

        # Run model
        pops = odeint(metabolic_model, y0=x0, t=t)

        # Steady state test
        ss_test = np.round(abs((pops[t_fin-1,0:N]) - (pops[t_fin-50,0:N])),3) 
        #print((ss_test))
        while True:
            if  np.any(ss_test > 0):
                t = sc.linspace(0,99,100)
                pops2 = odeint(metabolic_model, y0=pops[t_fin-1,:], t=t)
                pops = np.append(pops, pops2, axis=0)
                t_fin = t_fin + 100
                ss_test = np.round(abs((pops[t_fin-1,0:N]) - (pops[t_fin-50,0:N])),3)
            elif np.all(ss_test == 0):
                break
            else:
                pops=pops
        #print(t_fin)

        # Output for analysis in R
        T_fin_B = pops[t_fin-1,0:N]
        T_fin_N = np.tile(pops[t_fin-1,N:N+M],rep)
        T_fin = sc.full([N], t_fin)
        output_tmp = np.append(output_tmp, np.column_stack((T_fin_B, T_fin_N, Ea_1, Temp, B_n, N_n, T_fin, B_g_1, B_rm_1)),  axis=0)
        
        # Assembly
        rem_find = pops[t_fin-1,0:N] # final individuals numbers
        ext = np.where(rem_find<0.01) # Eas to keep
        rem_find = np.where(rem_find<0.01,0.1,rem_find) # replace extinct with 0.1
        x0 = np.concatenate((rem_find, pops[t_fin-1,N:N+M])) # new x0 with nutrients
        # Varying temp sensitivity 
        Ea_tmp = np.repeat(np.array([0.6, 0.7, 0.8, 0.9, 1.0]),N) # Create large temporary Ea for new species
        np.random.shuffle(Ea_tmp) # randomise vector
        Ea_tmp = Ea_tmp[0:len(ext[0])] # Cut vector so one Ea for each individual
        Ea[ext] = Ea_tmp # Replace removed Eas with new Eas
        # Varying temperature sensitivity
        #B_g_tmp_2 = np.exp(2.5 + (-5 * Ea_tmp))
        #B_rm_tmp_2 = 0.5 *(np.exp(2.5 + (-5 * Ea_tmp))) #(0.5 * B_g_tmp_2)     
        #B_g_tmp_2 = B_g_tmp_2[0:len(ext[0])] # Cut vector so one Ea for each individual
        #B_rm_tmp_2 = B_rm_tmp_2[0:len(ext[0])] # Cut vector so one Ea for each individual
        #B_g[ext] = B_g_tmp_2 # Replace removed Eas with new Eas
        #B_rm[ext] = B_rm_tmp_2 # Replace removed Eas with new Eas
        result_array = np.append(result_array, pops, axis=0)

    x0 = np.concatenate((sc.full([N], (0.1)),sc.full([M], (1.0))))
    output = np.column_stack((row, output_tmp))
    output_tmp_lrg = np.append(output_tmp_lrg, output,  axis=0)


#### Save outputs ####
#np.savetxt('Smith_params_new_temp_size_brm_ss.csv', output_tmp_lrg, delimiter=',')
#np.savetxt('sens_analysis_B_g-0.7.csv', output_tmp_lrg, delimiter=',')    
#np.savetxt('output_low-peak_traits_size_comp.csv', output_tmp_lrg, delimiter=',')
#np.savetxt('raw_test_ss.csv', result_array, delimiter=',')
#result_array = result_array[t_fin:len(result_array),:]
#   result_array = np.append(result_array, np.column_stack((pops[t_fin,0:N], np.repeat(pops[t_fin,N:N+M], rep))), axis=0)
# result_array = result_array[N:t_n_output,:]
# output_array = np.empty((N,3))
# for i in range(t_n):
#     T = 273.15 + i # temperature 
#     output = np.column_stack((np.array((np.repeat(T,N ))),Ea, Ea_D)) 
#     output_array = np.append(output_array, output, axis=0)
# output_array = output_array[N:t_n_output,:]

# final = np.column_stack((output_array, result_array))

# np.savetxt('final_nov_2019.csv', final, delimiter=',')





#### Plot output ####
t_plot = sc.linspace(0,len(result_array),len(result_array))
plt.plot(t_plot, result_array[:,0:N], 'g-', label = 'Consumers', linewidth=0.7)
plt.plot(t_plot, result_array[:,N:N+M], 'b-', label = 'Resources', linewidth=0.7)
#plt.plot(t, pops[:,0:N], 'g-', label = 'Consumers', linewidth=0.7)
#plt.plot(t, pops[:,N:N+M], 'b-', label = 'Resources', linewidth=0.7)
plt.grid
plt.ylabel('Population density')
plt.xlabel('Time')
plt.title('Bacteria-Nutrients population dynamics')
plt.legend([Line2D([0], [0], color='green', lw=2), Line2D([0], [0], color='blue', lw=2)], ['Bacteria', 'Nutrients'])
#plt.legend([Line2D([0], [0], color='green', lw=2)], ['Consumers'])
plt.savefig('Figure_ein_exist_2.png')
#plt.ylim([0, 10])
plt.show()



