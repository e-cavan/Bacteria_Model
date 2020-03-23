import scipy as sc
import numpy as np
from scipy.integrate import odeint
import matplotlib.pylab as plt

###### Functions ########

# def assign_params():
# """ Creates parameter values from MTE """

# def update_network():
# """ Adds or deletes "species" (substrate or consumers) and returns updated system """

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

######### Main Code ###########

######## Set up parameters ###########

# int_time = sc.linspace(0,100,101)

N = 100
M = 50

# immig_freq = ?

t = sc.linspace(0,200,101)

U = np.zeros([N,M])
np.fill_diagonal(U,1)

Rm = sc.full([N], (0.03))
Rg = sc.full([M], (0.01))
l = np.zeros([M,M])
for i in range(M-1):
    l[i,i+1] = 0.3 # Can use list comprehensions for some things like this
p = np.zeros(M)
x0 = np.concatenate((sc.full([N], (0.1)),sc.full([M], (1.0))))
u = np.ones(M)
l_sum = np.sum(l, axis=1)

##### Intergrate system forward #####

pops = odeint(metabolic_model, y0=x0, t=t)

####### Plot results ##########
plt.plot(t, pops[:,0:100], 'g-', label = 'Consumers', linewidth=1)
plt.plot(t, pops[:,100:150], 'b-', label = 'Resources', linewidth=1)
# plt.legend('g-', label = 'Consumers',)
plt.grid
plt.ylabel('Population density')
plt.xlabel('Time')
plt.title('Consumer-Resource population dynamics')
plt.show()