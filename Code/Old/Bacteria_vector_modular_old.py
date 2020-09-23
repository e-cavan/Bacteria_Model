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

N = 20 # Number of species
M = 10 # Number of nutrients
#K = 10 # number of species (* 100)
#Fg = 5 # number of functional groups
#Fg_num = N/Fg # number of species in each functional group
k = 0.0000862
Tref = 273.15 # 0 degrees C
pk = 20 # Peak above Tref ('low' = 12 and 'high' = 20)
T_pk = Tref + pk
#B_g = 0.5 # B0 for growth
#B_r = 0.1 # B0 for respiration (maintence)
#B_g = np.repeat(np.array([0.3, 0.4, 0.5, 0.6, 0.7]),Fg_num)
#B_r = (0.5 * B_g) - 0.1
Ma = 1 # Mass
Ea_D = np.repeat(3.5,N) # Deactivation energy
#temp = np.array([5, 10,15,20,25])
t_n = 21 # number of temperatures (0-10 inclusive)
#t_n_output = (t_n * N) + N # number of rows in output array
#rep = math.ceil(N/M)
ass = 1 # assembly number
t_fin = 200
t = sc.linspace(0,t_fin-1,t_fin)
x0 = np.concatenate((sc.full([N], (0.1)),sc.full([M], (1.0))))
#Meta_ratio_tmp = np.repeat(np.array([1, 1.1, 1, 0.9, 1]), 10)
#Meta_ratio = np.array(random.choices(Meta_ratio_tmp, k=M))
typ = 2 # functional response
K = 50

# for U/R analysis
time = np.empty((0, 100))
Meta_ratio = np.empty((0,100))
counter = 0

##### Intergrate system forward #####

def ass_temp_run(t, N, M, t_n,  Tref, Ma, k, ass, x0, t_fin, T_pk, Ea_D, typ):
    result_array = np.empty((0,N+M))    
    #row = np.repeat(sc.linspace(1,ass,ass), N) 
    #B_n = sc.linspace(1,N,N)
    #output_tmp_lrg = np.empty((0,11))
    #N_n = np.tile(sc.linspace(1,M,M), rep)

    #while counter < 100:
        
        #print(counter)
    t_fin = 200
    x0 = np.concatenate((sc.full([N], (0.1)),sc.full([M], (1.0))))
    result_array = np.empty((0,N+M))    

    for i in range(20, t_n):
    #for i in range(0, len(temp)):
    #for i in range(1,3):
        print(i)
        T = 273.15 + i # temperature 
        Temp = np.repeat(T - 273.15, N)
        # Ea_U_tmp2 = np.repeat(np.array([0.6, 0.7, 0.8, 0.9,1.0]),N) # Activation energy (same for g and rm)
        # Ea_R_tmp2 = np.repeat(np.array([0.7, 0.8,  0.9, 1.0, 1.1]),N) # Activation energy (same for g and rm)
        # #Meta_ratio = np.array([1, 1.1, 1, 0.9, 1])
        # Ea_U_tmp3 = shuffle(Ea_U_tmp2, random_state=0) 
        # Ea_U = Ea_U_tmp3[0:N]
        # #print(Ea_U)       
        # Ea_R_tmp3 = shuffle(Ea_R_tmp2, random_state=0) 
        # Ea_R = Ea_R_tmp3[0:N]
        #Ea_R = shuffle(Ea_R_tmp2, random_state=0)        
        #print(Ea_R)
        #Ea_R=np.repeat(0.3, 10)
        # Varying B0
        #B_g = np.exp(2.5 + (-5 * Ea))
        #B_rm = 0.5 *(np.exp(2.5 + (-5 * Ea))) #(0.5 * (B_g)) #
        Ea_U = np.random.normal(1.5, 0.02, 100)
        Ea_U = Ea_U[0:N]
        Ea_R = Ea_U - 0.7
        #Ea_R = Ea_R[0:N]
        B_g = (10**(2.84 + (-4.96 * Ea_U))) + 4
        B_r = (10**(1.29 + (-1.25 * Ea_R))) + 1

        #output_tmp = np.empty((0,10))
        #print('')
        #print('Temperature =',i)

        for j in range(ass):
            #print('')
            print('Assembly = ', j)
            #print(t_fin)
            t = sc.linspace(0,t_fin-1,t_fin)
            #Ea_1_U = Ea_U # To fill output dataframe
            #Ea_1_R = Ea_R # To fill output dataframe
            #B_g_1 = B_g # To fill output dataframe when B0 varies
            #B_rm_1 = B_rm # To fill output dataframe when B0 varies
            #B_g_1 = B_g # To fill output dataframe when B0 constant
            #B_r_1 = B_r # To fill output dataframe when B0 constant

            #print('Uptake Ea (C1 - C5)' + str(Ea_U[0:M]))
            #print('Uptake Ea (C6 - C10)' + str(Ea_U[M:N]))
            #print('Metabolic ratio between competitors' + str(Meta_ratio))
            
            # Set up model
            U = par.params(N, M, T, k, Tref, T_pk, B_g, B_r,Ma, Ea_U, Ea_R, Ea_D, Meta_ratio)[0] # make less than 1
            R = par.params(N, M, T, k, Tref, T_pk, B_g, B_r,Ma, Ea_U, Ea_R, Ea_D, Meta_ratio)[1] # make less than 1
            l = par.params(N, M, T, k, Tref, T_pk, B_g, B_r,Ma, Ea_U, Ea_R, Ea_D, Meta_ratio)[2]
            p = par.params(N, M, T, k, Tref, T_pk, B_g, B_r,Ma, Ea_U, Ea_R, Ea_D, Meta_ratio)[3]
            l_sum = np.sum(l, axis=1)
            pars = (U, R,  l, p, l_sum, Ea_U, Ea_R, Ea_D, N, M, T, Tref, B_r, B_g, Ma, k, ass, Meta_ratio, typ, K)

            #print('Metabolic ratio (C1 - C5)' + str(np.round(np.diag(U[5:10])/R[5:10], 3)))
            #print('Metabolic ratio (C6 - C10)' + str(np.round(np.diag(U)/R[0:5],3)))
            #print(np.round(U, 3))
                #U_tmp = st.temp_growth(k, T, Tref, T_pk, N, B_g, Ma, Ea_U, Ea_D) ## for u/r 
            #print(U_tmp)
            #print(np.round(R, 3))
            #print(U_tmp/ R)
            #print((R * K)/((U_tmp * (1-0)) - R))
            # print(Ea_U)
            # print(Ea_R)
            #print(B_g)
            #print(B_r)
            #Meta_ind = U_tmp/R
            #print(Meta_ind[0]/Meta_ind[1])
            #Meta_ratio = np.append(Meta_ratio, Meta_ind[0]/Meta_ind[1])

            # Run model
            pops = odeint(mod.metabolic_model, y0=x0, t=t, args = pars)
            pops = np.round(pops, 5)
            #print(pops)
            # Steady state test
            ss_test = np.round(abs((pops[t_fin-1,0:N]) - (pops[t_fin-50,0:N])),3) 
            while True:
                if  np.any(ss_test > 0):
                    t = sc.linspace(0,99,100)
                    pops2 = odeint(mod.metabolic_model, y0=pops[t_fin-1,:], t=t, args=pars)
                    pops = np.append(pops, pops2, axis=0)
                    t_fin = t_fin + 100
                    ss_test = np.round(abs((pops[t_fin-1,0:N]) - (pops[t_fin-50,0:N])),5)
                elif np.all(ss_test == 0):
                    break
                else:
                    pops=pops
            print(t_fin)
            
            pops = np.round(pops, 5)
                #time = np.append(time, len(np.unique(pops, axis=0))) ## for u/r 
            #print(Meta_ratio)
            #print('Bacteria mass at equilibrium (C1 - C5)' + str(pops[len(pops)-1,0:M]))
            #print('Bacteria mass at equilibrium (C6 - C10)' + str(pops[len(pops)-1,M:N]))
            #print('Resource mass at equilibrium (S1 - S5)' + str(pops[len(pops)-1,N:N+M]))

            # Output for analysis in R
            #T_fin_B = pops[t_fin-1,0:N]
            #T_fin_N = np.tile(pops[t_fin-1,N:N+M],rep)
            #T_fin = sc.full([N], t_fin)
            #output_tmp = np.append(output_tmp, np.column_stack((T_fin_B, T_fin_N, Ea_1_U, Ea_1_R, Temp, B_n, N_n, T_fin, B_g_1, B_r_1)),  axis=0)
            
            # if j == ass-1: 
            #     break

            # #Assembly
            # rem_find = pops[t_fin-1,0:N] # final individuals numbers
            # ext = np.where(rem_find<0.01) # Eas to keep
            # rem_find = np.where(rem_find<0.01,0.1,rem_find) # replace extinct with 0.1
            # x0 = np.concatenate((rem_find, pops[t_fin-1,N:N+M])) # new x0 with nutrients
            
            # # Varying temp sensitivity 
            # Ea_tmp_U = np.repeat(np.array([0.6, 0.7, 0.8, 0.9, 1.0]),N) # Create large temporary Ea for new species
            # Ea_tmp_U = shuffle(Ea_U_tmp2) # randomise vector
            # Ea_tmp_U = Ea_tmp_U[0:len(ext[0])] # Cut vector so one Ea for each individual
            # Ea_U[ext] = Ea_tmp_U # Replace removed Eas with new Eas
            # Ea_tmp_R = np.repeat(np.array([0.6, 0.7, 0.8, 0.9, 1.0]),N) # Create large temporary Ea for new species
            # shuffle(Ea_R_tmp2, random_state=1) # randomise vector
            # Ea_tmp_R = Ea_R_tmp2[0:len(ext[0])] # Cut vector so one Ea for each individual
            # Ea_R[ext] = Ea_tmp_R # Replace removed Eas with new Eas

            # # Change relationship between old B and new one migrating in  
            # R_ext_tmp = np.array(ext)
            # R_ext_tmp[R_ext_tmp>M] = R_ext_tmp[R_ext_tmp>M] - M # - M so find which nut it had
            # R_ext = np.unique(R_ext_tmp) # remove duplicates, i.e. if both competitors die (wont happen)
            # R_func_new = np.repeat(np.array([0.8,0.9,1,1.1,1.2]), 50) # new U/Rm conditions
            # R_func_new = shuffle(R_func_new) # shuffle U/Rm conditions
            # Meta_ratio[R_ext] = R_func_new[R_ext]
            

            # B_g = 10**(2.84 + (-4.96 * Ea_U)) + 4
            # B_r = 10**(1.29 + (-1.25 * Ea_R))

            # Varying temperature sensitivity
            #B_g_tmp_2 = np.exp(2.5 + (-5 * Ea_tmp))
            #B_r_tmp_2 = 0.5 *(np.exp(2.5 + (-5 * Ea_tmp))) #(0.5 * B_g_tmp_2)     
            #B_g_tmp_2 = B_g_tmp_2[0:len(ext[0])] # Cut vector so one Ea for each individual
            #B_r_tmp_2 = B_r_tmp_2[0:len(ext[0])] # Cut vector so one Ea for each individual
            #B_g[ext] = B_g_tmp_2 # Replace removed Eas with new Eas
            #B_r[ext] = B_r_tmp_2 # Replace removed Eas with new Eas
            result_array = np.append(result_array, pops, axis=0)

        x0 = np.concatenate((sc.full([N], (0.1)),sc.full([M], (1.0))))
        #output = np.column_stack((row, output_tmp))
        #output_tmp_lrg = np.append(output_tmp_lrg, output,  axis=0)

    #print(result_array[len(result_array)-1,0:N]   )

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

        #counter = counter + 1


    #### Plot output ####
    t_plot = sc.linspace(0,len(result_array),len(result_array))
    plt.plot(t_plot, result_array[:,0:N], 'g-', label = 'Consumers', linewidth=0.7)
    plt.plot(t_plot, result_array[:,N:N+M], 'b-', label = 'Resources', linewidth=0.7)
    #plt.plot(t, pops[:,0:N], 'g-', label = 'Consumers', linewidth=0.7)
    #plt.plot(t, pops[:,N:N+M], 'b-', label = 'Resources', linewidth=0.7)
    plt.grid
    plt.ylabel('Population density')
    plt.xlabel('Time')
    plt.title('Bacteria-Resource population dynamics')
    plt.legend([Line2D([0], [0], color='green', lw=2), Line2D([0], [0], color='blue', lw=2)], ['Bacteria', 'Resources'])
    #plt.legend([Line2D([0], [0], color='green', lw=2)], ['Consumers'])
    #plt.savefig('Figure_ein_exist_2.png')
    #plt.ylim([0, 10])
    plt.show()

    # t_plot = sc.linspace(0,len(Meta_ratio),len(time[0:100]))
    # plt.plot(Meta_ratio, time[0:100], 'go')
    # #plt.plot(t_plot, result_array[:,N:N+M], 'b-', label = 'Resources', linewidth=0.7)
    # #plt.plot(t, pops[:,0:N], 'g-', label = 'Consumers', linewidth=0.7)
    # #plt.plot(t, pops[:,N:N+M], 'b-', label = 'Resources', linewidth=0.7)
    # plt.grid
    # plt.xlim(0.9,1.1)
    # plt.xlabel('Metabolic ratio between species (U/R:U/R)')
    # plt.ylabel('Time to mono dominance')
    # #plt.title('Bacteria-Resource population dynamics')
    # #plt.legend([Line2D([0], [0], color='green', lw=2), Line2D([0], [0], color='blue', lw=2)], ['Bacteria', 'Resources'])
    # #plt.legend([Line2D([0], [0], color='green', lw=2)], ['Consumers'])
    # #plt.savefig('Figure_ein_exist_2.png')
    # #plt.ylim([0, 10])
    # plt.show()

ass_temp_run(t, N, M, t_n,  Tref, Ma, k, ass, x0, t_fin, T_pk , Ea_D,  typ)




