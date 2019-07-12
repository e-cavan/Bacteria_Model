import scipy as sc
import numpy as np
from scipy.integrate import odeint
import matplotlib.pylab as pie

N = 100
M = 50

U = np.zeros([N,M])
np.fill_diagonal(U,1)
Rm = sc.full([N], (0.3))
Rg = sc.full([M], (0.1))
l = np.zeros([M,M])
for i in range(M-1):
    l[i,i+1] = 0.3
p = np.zeros(M)
t = sc.linspace(0,100,101)
x0 = np.concatenate((sc.full([N], (0.1)),sc.full([M], (1.0))))
u = np.ones(M)
l_sum = np.sum(l, axis=1)


def metabolic_model(pops,t):
    x = pops


    xc =  x[0:100] # consumer
    xr =  x[100:150] # resources

    ## Consumers
    # calculate 'middle'/ growth - Rg - leakeage term
    SL = (u - Rg - l_sum) * xr
    #uptake rate and maintenance
    C = (np.sum(SL * U, axis=1)) - Rm
    #dCdt
    dCdt = xc * C

    ## Resources
    dSdt = p - np.multiply((xc @ U).transpose(), xr) + np.einsum('i,k,ik,kj->j', xc, xr, U, l)

    return np.array(np.concatenate((dCdt, dSdt)))

pops = odeint(metabolic_model, y0=x0, t=t)

pie.plot(t, pops[:,0:100], 'g-', label = 'Resources', linewidth=0.7)
pie.plot(t, pops[:,100:150], 'b-', label = 'Consumers', linewidth=0.7)
pie.grid
pie.ylabel('Population density')
pie.xlabel('Time')
pie.title('Consumer-Resource population dynamics')
pie.show()