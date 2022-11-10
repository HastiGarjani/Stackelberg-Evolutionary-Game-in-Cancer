################################################################
#############              Figure 3               ##############
# Leader and Follower Best responses
# Quality of life level curves
# Stabilization region

import numpy as np
import pylab
from matplotlib import pyplot as plt
import matplotlib
import string
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FixedLocator

pylab.rcParams['figure.figsize'] = (11.25, 5.0)
# Parameters
K_max = 10000
r_max = 0.45
d = 0.01
g = 0.8
sigma  = 1
k = 2
b = 10
# Competition coefficients
aRS = 0.9
aSR = 0.1
# Quality of life coefficients
a1 = 0.54
a2 = 0.21
a3 = 0.25

# Initialization
m = np.linspace(0,1,201)
u = np.linspace(0,1,201)
x_S_hist = []
x_R_hist = []
Q_hist = []
####### Meshgrid to specify stabilization region and level curve value ########
for mm in m:
    x_S = []
    x_R = []
    Q = []
    for uu in u:
        x_R_temp = max(0,(K_max/(r_max*(1-aSR*aRS)))*(aSR*mm*np.exp(g*uu)/(k+b*uu)\
             - mm/k + aSR*d*np.exp(g*uu) + r_max -d - aSR*r_max))
        x_S_temp = max(0,(K_max/(r_max*(1-aSR*aRS)))*(-mm*np.exp(g*uu)/(k+b*uu) \
            + aRS*mm/k - d*np.exp(g*uu) + r_max - aRS*r_max + aRS*d))
        Q_temp = 1 - a1*(((x_R_temp+x_S_temp)/K_max)**2) - a2*(uu**2) - a3*(mm**2)
        x_S.append(x_S_temp)
        x_R.append(x_R_temp)
        Q.append(Q_temp)
    x_S_hist.append(x_S)
    x_R_hist.append(x_R)
    Q_hist.append(Q)

Y1, Y2 = np.meshgrid(m,u)
z = Q_hist
# Quality of life level curves
levels = MaxNLocator(nbins=10).tick_values(np.min(z), np.max(z))
cmap = matplotlib.cm.Blues(np.linspace(0,1,13))
cmap = matplotlib.colors.ListedColormap(cmap[4:,:-1])
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [4, 5]})

ax2.set_xlabel('Treatment dose', size = 10)
ax2.set_ylabel('Resistance rate', size = 10)
ax2.set_ylim((0,1))
ax2.set_xlim((0,1))
im = ax2.pcolormesh(Y1, Y2, np.transpose(z), cmap=cmap, norm=norm, shading='nearest')
fig.colorbar(im, ax=ax2, format='%.2f')
# Stabilization region
z2 = np.add(x_R_hist,x_S_hist)
levels2 = FixedLocator([z2.min(), 0, 7000, z2.max()], nbins=4).tick_values(z2.min(), z2.max())
colorsList2 = ['green', 'none', 'white']
cmap2 = matplotlib.colors.ListedColormap(colorsList2)
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)
im = ax2.pcolormesh(Y1, Y2, np.transpose(z2), cmap=cmap2, norm=norm2, shading='nearest')


# Best response curve of the leader
kk = (1/(1-aSR*aRS))**2
r_u = r_max*np.exp(-g*u)
m = (a1*kk*((aSR-1)/((k+b*u)*r_u)+(aRS-1)/(k*r_max))*\
    ((1-aRS)*(1-d/r_max)+(1-aSR)*(1-d/r_u)))/\
        (-a1*kk*((aSR-1)/((k+b*u)*r_u)+(aRS-1)/(k*r_max))*\
            ((aRS-1)/(k*r_max)+(aSR-1)/((k+b*u)*r_u))-a3)

# Illustrating best response curves
ax2.plot(m,u,'b--',linewidth=3)
mm1 = np.linspace(0,1,1001)
u_Stack_temp = -k/b - mm1/(2*b*d) + np.sqrt((mm1 * g)**2 + 4*mm1*g*b*d)/(2*b*d*g)
ax2.plot(mm1,u_Stack_temp,color='b',label="Follower's Best Response", linewidth=3)

# Stackelberg solution (mS_idx)
# mS_idx=mS_idx is calculated in Q_Stack file
mS_idx = 660
u_st = u_Stack_temp[mS_idx]
x_S_st = (K_max/(r_max*(1-aSR*aRS)))*(aSR*mm1[mS_idx]*np.exp(g*u_st)/(k+b*u_st) - mm1[mS_idx]/k \
    + aSR*d*np.exp(g*u_st) + r_max - d - aSR*r_max)
x_R_st = (K_max/(r_max*(1-aSR*aRS)))*(-mm1[mS_idx]*np.exp(g*u_st)/(k+b*u_st) + aRS*mm1[mS_idx]/k \
    - d*np.exp(g*u_st) + r_max - aRS*r_max + aRS*d)

# Quality of life at Stackelberg solution
Q_S = 1 - a1*(((x_R_st+x_S_st)/K_max)**2)- a2*(u_st**2) - a3*(mm1[mS_idx]**2)
ax2.scatter(mm1[mS_idx],u_Stack_temp[mS_idx],color='b', s=50)
ax2.text(mm1[mS_idx]-0.18,u_Stack_temp[mS_idx]+0.05,'Stackelberg',backgroundcolor='1')

# Nash solution (mN_idx)
# Nash solution is calculated at the intersection 
## of Leader and follower best response curves
mN_idx = 749
u_N = u_Stack_temp[mN_idx]
x_S_N = (K_max/(r_max*(1-aSR*aRS)))*(aSR*mm1[mN_idx]*np.exp(g*u_N)/(k+b*u_N) - mm1[mN_idx]/k \
    + aSR*d*np.exp(g*u_N) + r_max - d - aSR*r_max)
x_R_N = (K_max/(r_max*(1-aSR*aRS)))*(-mm1[mN_idx]*np.exp(g*u_N)/(k+b*u_N) + aRS*mm1[mN_idx]/k \
    - d*np.exp(g*u_N) + r_max - aRS*r_max + aRS*d)

# Quality of life at Nash solution
Q_N = 1 - a1*(((x_S_N+x_R_N)/K_max)**2)- a2*u_N**2 - a3*mm1[mN_idx]**2

# Illustrating MTD, Stackleberg, and Nash solutions
# MTD means Maximum tolerable dose which is 1 and happens at index 1000
mMTD_idx = 1000
ax2.scatter(mm1[mN_idx],u_Stack_temp[mN_idx],color='b', s=50)
ax2.text(mm1[mN_idx]+0.028,u_Stack_temp[mN_idx]-0.06,'Nash',backgroundcolor='1')
ax2.scatter(mm1[mMTD_idx],u_Stack_temp[mMTD_idx],color='b', s=50)
ax2.text(mm1[mMTD_idx]-0.07,u_Stack_temp[mMTD_idx]-0.07,'MTD',backgroundcolor='1') 

# Stabilization and progression regions
ax1.set_xlabel('Treatment dose', size = 10)
ax1.set_ylabel('Resistance rate', size = 10)
ax1.set_ylim((0,1))
ax1.set_xlim((0,1))
z = np.add(x_R_hist,x_S_hist)
levels = FixedLocator([z.min(), 0, 7000, z.max()], nbins=4).tick_values(z.min(), z.max())
colorsList = ['green', 'yellow', 'red']
cmap = matplotlib.colors.ListedColormap(colorsList)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
im = ax1.pcolormesh(Y1, Y2, np.transpose(z), cmap=cmap, norm=norm, shading='nearest')

# Leader and follower best response curves
ax1.plot(m,u,'b--',label="Leader's Best Response", linewidth=3)
ax1.plot(mm1,u_Stack_temp,color='b', linewidth=3)

# Illustrating MTD, Nash and Stackelberg solutions
ax1.scatter(mm1[mS_idx],u_Stack_temp[mS_idx],color='b', s=50)
ax1.text(mm1[mS_idx]-0.18,u_Stack_temp[mS_idx]+0.07,'Stackelberg',backgroundcolor='1')
ax1.text(mm1[mS_idx]-0.18,u_Stack_temp[mS_idx]+0.02,'Q='+str(round(Q_S,3)),backgroundcolor='1',size='x-small')
ax1.scatter(mm1[mN_idx],u_Stack_temp[mN_idx],color='b', s=50)
ax1.text(mm1[mN_idx]+0.028,u_Stack_temp[mN_idx]-0.05,'Nash',backgroundcolor='1',size='small')
ax1.text(mm1[mN_idx]+0.028,u_Stack_temp[mN_idx]-0.1,'Q='+str(round(Q_N,3)),backgroundcolor='1',size='x-small')
ax1.scatter(mm1[mMTD_idx],u_Stack_temp[mMTD_idx],color='b', s=50)
ax1.text(mm1[mMTD_idx]-0.09,u_Stack_temp[mMTD_idx]+0.03,'MTD',backgroundcolor='1',size='small')
ax1.set_xlim([0,1.001])
fig.legend()
ax1.text(-0.1, 1.05, string.ascii_uppercase[0], transform=ax1.transAxes, 
            size=20, weight='normal')
ax2.text(-0.1, 1.05, string.ascii_uppercase[1], transform=ax2.transAxes, 
            size=20, weight='normal')
plt.savefig("subplot_sep.pdf", dpi=100)
plt.show()