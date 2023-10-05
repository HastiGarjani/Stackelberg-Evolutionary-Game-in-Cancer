from tkinter import Scale
import numpy as np
import matplotlib.pyplot as plt
from cv2 import normalize
import numpy as np
import matplotlib.patches as mpatches
styles = mpatches.ArrowStyle.get_styles()
import matplotlib.patches as patches
from itertools import product

import string
K_max21 = 10000
r_max21 = 0.45
d = 0.01
g = 0.8
sigma = 1
a = 2
b = 10
a21 = 0.9
a12 = 0.15
m = 0.723
# m = 0.74
# xS_star = 0.0998
# xR_star = 0.5574
# u_star = 0.8895
xS_star = 882.0753429720928/K_max21
xR_star = 5749.127343149015/K_max21
u_star = 0.8866725747333786
def dif(xyz):
    xS, xR, uR = xyz
    r_u = r_max21*np.exp(-g*uR)
    dxS = xS*(r_max21*(1-a12*xR-xS) - d - m/a)
    dxR = xR*(r_u*(1-(xR+a21*xS)) - d - m/(a+b*uR))
    duR = sigma*(-g*r_u*(1-xR-a21*xS) + m*b/((a+b*uR)**2))
    return np.array([dxS, dxR, duR])


dt = 0.1
num_steps = 20000
center1 = 15
center2 = 35
center3 = 20+10
xyzs = np.empty((num_steps + 1, 3))  
#########################################     2D      ################################
from matplotlib import gridspec
import pylab
pylab.rcParams['figure.figsize'] = (11.25, 7)
plt.rcParams.update({'font.size': 16})

fig = plt.figure()
fig.tight_layout()
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1]) 
ax2 = fig.add_subplot(gs[1], aspect='equal')
fig.subplots_adjust(wspace=0, hspace=0)

########################################## Example 1 ###################################
xyzs[0] = (0.5, 0.1, 0.2)  
for i in range(num_steps):
    xyzs[i + 1] = xyzs[i] + dif(xyzs[i]) * dt
xyzs_2d = xyzs[:,0:2]
ax2.plot(*xyzs_2d.T, '-',lw=1.5,c='r')
# ax2.invert_xaxis()
ax2.quiver(xyzs[center1,0],xyzs[center1,1],\
    1*np.diff(xyzs[:,0])[center1+1],1*np.diff(xyzs[:,1])[center1+1],width=0.008,headwidth=8,headlength=6,headaxislength=3,color='red')

# plt.arrow(xyzs_2d[center1,0],xyzs_2d[center1,1],\
#     1*np.diff(xyzs_2d[:,0])[center1+1],1*np.diff(xyzs_2d[:,1])[center1+1],head_width=0.03,color='red')
ax2.set_xlabel("Sensitive population", size=18)
ax2.set_ylabel("Resistant population", size=18)
########################################## Example 2 ###################################
xyzs2 = np.empty((num_steps + 1, 3))  
xyzs2[0] = (0.3, 0.4, 0.5)  
for i in range(num_steps):
    xyzs2[i + 1] = xyzs2[i] + dif(xyzs2[i]) * dt
xyzs2_2d = xyzs2[:,0:2]
ax2.plot(*xyzs2_2d.T, '-',lw=1.5,c='b')
ax2.scatter(xS_star,xR_star,c='black')

ax2.quiver(xyzs2[center2,0],xyzs2[center2,1],\
    15*np.diff(xyzs2[:,0])[center2+1],15*np.diff(xyzs2[:,1])[center2+1],width=0.008,headwidth=8,headlength=6,headaxislength=3,color='blue')

# plt.arrow(xyzs2_2d[center2,0],xyzs2_2d[center2,1],\
#     0.07*np.diff(xyzs2_2d[:,0])[center2+1],0.1*np.diff(xyzs2_2d[:,1])[center2+1],head_width=0.027,color='blue')
########################################## Example 3 ###################################
xyzs3 = np.empty((num_steps + 1, 3))  
xyzs3[0] = (0.2, 0.2, 0.7)  
for i in range(num_steps):
    xyzs3[i + 1] = xyzs3[i] + dif(xyzs3[i]) * dt
xyzs3_2d = xyzs3[:,0:2]
ax2.plot(*xyzs3_2d.T, '-',lw=1.5,c='purple')
ax2.quiver(xyzs3[center3,0],xyzs3[center3,1],\
    1*np.diff(xyzs3[:,0])[center3+1],1*np.diff(xyzs3[:,1])[center3+1],width=0.008,headwidth=8,headlength=6,headaxislength=3,color='purple')

# patches.FancyArrowPatch((xyzs3_2d[center3,0],xyzs3_2d[center3,1]),\
#     (xyzs3_2d[center3+1000,0],xyzs3_2d[center3+1000,1]), arrowstyle='<->')
# plt.arrow(xyzs3_2d[center3,0],xyzs3_2d[center3,1],\
    # 1*np.diff(xyzs3_2d[:,0])[center3+1],1*np.diff(xyzs3_2d[:,1])[center3+1],head_width=0.03,color='purple')
# ax2.annotate('', xy=(xyzs3_2d[center3,0],xyzs3_2d[center3,1]), xytext=(xyzs3_2d[center3-10,0],xyzs3_2d[center3-10,1]),
#             arrowprops={'arrowstyle': '->', 'lw':1,'color': 'purple'})
plt.savefig("trajectory_2d.pdf")

#
#
#

#########################################     3D      ################################
# ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax1 = fig.add_subplot(gs[0], projection='3d')

########################################## Example 1 ###################################
ax1.view_init(elev=30., azim=+80)
ax1.plot(*xyzs.T, '-',lw=1.5,c='r')
ax1.quiver(xyzs[center1,0],xyzs[center1,1],xyzs[center1,2],\
    10*np.diff(xyzs[:,0])[center1+1],10*np.diff(xyzs[:,1])[center1+1],10*np.diff(xyzs[:,2])[center1+1],length = 1,arrow_length_ratio=1.2,color='red')
ax1.set_xlabel("\nSensitive population", size=18)
# ax2.invert_xaxis()
# ax1.set_ylabel("$x_R$", size=15)
ax1.set_ylabel("\nResistant population", size=18)
ax1.set_ylim([0.1,1])
ax1.set_xlim([0.1,1])
# ax1.set_zlim([0.1,1])

ax1.set_zlabel("\nResistance rate", size=18)
########################################## Example 2 ###################################
ax1.plot(*xyzs2.T, '-',lw=1.5,c='b')
ax1.scatter(xS_star,xR_star,u_star,"o",c='black')
u_all = np.linspace(0.1,1,10)
xs_all = np.linspace(0.1, 1, 10)
xR_all = np.linspace(0.1, 1, 10)
for xs, xR, u in product(xs_all, xR_all, u_all):
    if(xs+xR<1):
        ax1.plot(xs,xR,u, marker = ".", c='black', markersize = 2)

ax1.quiver(xyzs2[center2,0],xyzs2[center2,1],xyzs2[center2,2],\
    15*np.diff(xyzs2[:,0])[center2+1],15*np.diff(xyzs2[:,1])[center2+1],15*np.diff(xyzs2[:,2])[center2+1],length = 1,arrow_length_ratio=1.5,color='blue')

########################################## Example 3 ###################################
ax1.plot(*xyzs3.T, '-',lw=1.5,c='purple')
ax1.quiver(xyzs3[center3,0],xyzs3[center3,1],xyzs3[center3,2],\
    25*np.diff(xyzs3[:,0])[center3+1],25*np.diff(xyzs3[:,1])[center3+1],25*np.diff(xyzs3[:,2])[center3+1],length = 1,arrow_length_ratio=1.2,color='purple')
plt.tight_layout()

ax1.text(-4, -20, 15, string.ascii_uppercase[0], transform=ax1.transAxes, 
            size=20, weight='normal')
ax2.text(-0.1, 1.15, string.ascii_uppercase[1], transform=ax2.transAxes, 
            size=20, weight='normal')
plt.savefig("trajectory.pdf")
plt.tight_layout()


plt.savefig("trajectory.png", transparent=True )

plt.show()