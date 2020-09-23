#### Model equations

import numpy as np

def metabolic_model(pops,t, U, R, l, p, l_sum, Ea_U, Ea_R, Ea_D, N, M, T, Tref, B_r, B_g, Ma, k, ass, Meta_ratio):
    x = pops

    xc =  x[0:N] # consumer
    xr =  x[N:N+M] # resources

    ## Consumers
    # calculate 'middle'/ growth - Rg - leakeage term
    SL = (1 - l_sum) * xr
    #uptake rate and maintenance
    C = (np.sum(SL * U, axis=1)) - R
    #dCdt
    dCdt = xc * C
    
    ## Resources
    dSdt = p - np.multiply((xc @ U).transpose(), xr) + np.einsum('i,k,ik,kj->j', xc, xr, U, l)

    return np.array(np.concatenate((dCdt, dSdt)))
