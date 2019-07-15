import scipy as sc
import numpy as np
from scipy.integrate import odeint
import  matplotlib as plt
import matplotlib.pylab as pie

N = 100
M = 50

u1 = np.zeros([N,M])
np.fill_diagonal(u1,1)
Rm = sc.full([N], (0.3))
Rg = sc.full([M], (0.1))
l = np.zeros([M,M])
for i in range(M-1):
    l[i,i+1] = 0.3
p = np.zeros(M)
t = sc.linspace(0,100,101)
x0 = np.concatenate((sc.full([N], (0.1)),sc.full([M], (1.0))))


gt = 0
rgt = 0
et = 0
gRt = 0
eRt = 0


pars = (N, M, u1, Rm, Rg, l, p)

def metabolic_model(pops,N, M, u1, Rm, Rg, l, p, t):

    N = 100
    M = 50

    u1 = np.zeros([N,M])
    np.fill_diagonal(u1,1)
    Rm = sc.full([N], (0.3))
    Rg = sc.full([M], (0.1))
    l = np.zeros([M,M])
    for i in range(M-1):
        l[i,i+1] = 0.3
    p = np.zeros(M)
    t = sc.linspace(0,100,11)
    x0 = np.concatenate((sc.full([N], (0.1)),sc.full([M], (1.0))))

    Nt = np.array([])
    Mt = np.array([])

    x = pops

    for i in range(int(N)):
        #print(i)
        m = x[i] * Rm[i] # maintenance lost
        et = 0
        gt = 0
        rgt = 0
        for j in range(int(M)):
            g = u1[i,j] * x[i] * x[N+j] # growth
            gt = gt + g
            rg = u1[i,j] * x[i] * x[N+j] * Rg[j] # loss to growth
            rgt = rgt + rg
            for k in range(int(M)):
                e = u1[i,j] * x[i] * x[N+j] * l[j,k] # excretion
                et = et + e     
        xt = gt - rgt - m - et
        Nt = np.append(Nt, xt)

    for a in range(int(M)):
        #print(a)
        xR = 0 # external resrouces
        gRt = 0
        eRt = 0
        for b in range(int(N)):
            #print(b)
            gR = u1[b,a] * x[b] * x[N+a] # loss of resources to growth
            gRt = gRt + gR
            
            for c in range(int(M)):
                eR = u1[b,c] * x[b] * x[N+c] * l[c,a] # gain of resources from excretion
                eRt = eRt + eR
        rt = xR + eRt - gRt
        Mt = np.append(Mt, rt) 
    return np.array(np.concatenate((Nt, Mt)))

pops = odeint(metabolic_model, y0=x0, t=t, args=pars)


pie.plot(t, pops[:,0:100], 'g-', label = 'Resources')
pie.plot(t, pops[:,101:150], 'b-', label = 'Consumers')
pie.grid
pie.xlabel('Time')
pie.ylabel('Population density')
pie.title('Consumer-Resource population dynamics')
pie.show()