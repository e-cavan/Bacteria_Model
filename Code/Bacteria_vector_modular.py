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
import size_temp_funcs as st
import parameters as par
import model_func as mod
import random


######### Main Code ###########


######## Set up parameters ###########

N = 10 # Number of consumers
M = 5 # Number of resources

# Temperature params
k = 0.0000862 # Boltzman constant
Tref = 273.15 # Reference temperature Kelvin, 0 degrees C
pk = 20 # Peak above Tref, degrees C
T_pk = Tref + pk # Peak above Tref, Kelvin
Ma = 1 # Mass
Ea_D = np.repeat(3.5,N) # Deactivation energy - only used if use Sharpe-Schoolfield temp-dependance
t_n = 21 # Number of temperatures to run the model at, model starts at 20

# Assembly
ass = 2 # Assembly number, i.e. how many times the system can assemble

t_fin = 100 # Number of time steps
t = sc.linspace(0,t_fin-1,t_fin) # Time steps
x0 = np.concatenate((sc.full([N], (0.1)),sc.full([M], (0.1)))) # Starting concentration for resources and consumers
typ = 2 # Functional response, Type I or II
K = 10 # Carrying capacity, kept constant with temperature



##### Intergrate system forward #####

def ass_temp_run(t, N, M, t_n,  Tref, Ma, k, ass, x0, t_fin, T_pk, Ea_D, typ):
    result_array = np.empty((0,N+M)) # Array to store data in for plotting
   

    for i in range(20, t_n):        # Run model at multiple temperatures, here set to just run at 20 C
        T = 273.15 + i # Temperature

        # Set up Ea (activation energy) and B0 (normalisation constant)
        # Based on Tom Smith's observations
        Ea_U = np.round(np.random.normal(1.5, 0.01, 10),3)[0:N] # Ea for uptake
        Ea_R = Ea_U - 0.8 # Ea for respiration, which should always be lower than Ea_U so 'peaks' later
        B_U = (10**(2.84 + (-4.96 * Ea_U))) + 4 # B0 for uptake - ** NB The '+4' term is added so B_U>> B_R, otherwise often the both consumers can die and the resources are consumed
        B_R = (10**(1.29 + (-1.25 * Ea_R))) # B0 for respiration

        #print('')
        #print('Temperature =',i)

        for j in range(ass):

            #print('')
            #print('Assembly = ', j)
            t = sc.linspace(0,t_fin-1,t_fin) # resetting 't' if steady state not reached (see below)

            #print('Uptake Ea (C1 - C5)' + str(Ea_U[0:M]))
            #print('Uptake Ea (C6 - C10)' + str(Ea_U[M:N]))
            #print('Metabolic ratio between competitors' + str(Meta_ratio))
            
            # Set up model
            U = par.params(N, M, T, k, Tref, T_pk, B_U, B_R,Ma, Ea_U, Ea_R, Ea_D)[0] # Uptake
            R = par.params(N, M, T, k, Tref, T_pk, B_U, B_R,Ma, Ea_U, Ea_R, Ea_D)[1] # Respiration
            l = par.params(N, M, T, k, Tref, T_pk, B_U, B_R,Ma, Ea_U, Ea_R, Ea_D)[2] # Leakage
            p = par.params(N, M, T, k, Tref, T_pk, B_U, B_R,Ma, Ea_U, Ea_R, Ea_D)[3] # Resource input
            l_sum = np.sum(l, axis=1)
            pars = (U, R,  l, p, l_sum, Ea_U, Ea_R, Ea_D, N, M, T, Tref, B_R, B_U, Ma, k, ass, typ, K) # Parameters to pass onto model


            # Run model
            pops = odeint(mod.metabolic_model, y0=x0, t=t, args = pars) # Integrate
            #pops = np.round(pops, 5)


            # Steady state test
            ss_test = np.round(abs((pops[t_fin-1,0:N]) - (pops[t_fin-50,0:N])),3) # Find difference between last timestep and timestep at t_fin - 50 (i.e. 150 if t_fin = 200)
            while True:
                if  np.any(ss_test > 0):    # If there is a difference then that consumer not yet reached steady state
                    t = sc.linspace(0,99,100) # reset t so shorter, only 100 timesteps
                    pops2 = odeint(mod.metabolic_model, y0=pops[t_fin-1,:], t=t, args=pars) # re-run model using last timestep concentrations of consumers and resources
                    pops = np.append(pops, pops2, axis=0) # append results of additional run to orginial run
                    t_fin = t_fin + 100 # adjust timesteps number
                    ss_test = np.round(abs((pops[t_fin-1,0:N]) - (pops[t_fin-50,0:N])),3) # Find again if consumers reached steady state now
                elif np.all(ss_test == 0):
                    break # Once no difference in consumer concentration then stop performing additional model runs 
                else:
                    pops=pops # If at steady state then nothing happens
            
            t_fin = 100

            pops = np.round(pops, 7)
            # if j == ass-1: 
            #     break

            ###Assembly###

            # Find which consumers have gone extinct
            rem_find = pops[t_fin-1,0:N] # Get the consumer concentrations from last timestep
            ext = np.where(rem_find<0.01) # Find which consumers have a concentration < 0.01 g/mL, i.e. are extinct
            rem_find = np.where(rem_find<0.01,0.1,rem_find) # Replace extinct consumers with a new concentration of 0.1
            x0 = np.concatenate((rem_find, pops[t_fin-1,N:N+M])) # Join new concentrations for consumers with those of resources
            
            # Create new consumers

            # New Ea_ and Ea_R
            Ea_tmp_U = np.round(np.random.normal(1.5, 0.01, 10),3)[0:len(ext[0])] # Ea for uptake cut to length(ext),i.e. the number of extinct consumers
            Ea_U[ext] = Ea_tmp_U # Replace removed Eas with new Eas
            Ea_R = Ea_U - 0.8

            # New B0
            B_U = 10**(2.84 + (-4.96 * Ea_U)) + 4
            B_R = 10**(1.29 + (-1.25 * Ea_R))

            #print(np.round(Ea_U, 3))

            result_array = np.append(result_array, pops, axis=0)

        x0 = np.concatenate((sc.full([N], (0.1)),sc.full([M], (0.1))))
     
  
    #### Plot output ####
    t_plot = sc.linspace(0,len(result_array),len(result_array))
    plt.plot(t_plot, result_array[:,0:N], 'g-', label = 'Consumers', linewidth=0.7)
    plt.plot(t_plot, result_array[:,N:N+M], 'b-', label = 'Resources', linewidth=0.7)
    plt.grid
    plt.ylabel('Population density')
    plt.xlabel('Time')
    plt.title('Consumer-Resource population dynamics')
    plt.legend([Line2D([0], [0], color='green', lw=2), Line2D([0], [0], color='blue', lw=2)], ['Consumer', 'Resources'])
    plt.show()


    U_out = (st.temp_growth(k, T, Tref, T_pk, N, B_U, Ma, Ea_U, Ea_D))

    return result_array, U_out, R

ass_temp_run(t, N, M, t_n,  Tref, Ma, k, ass, x0, t_fin, T_pk , Ea_D,  typ)




