################################################################
############# Code to Obtain Stackelberg Strategy ##############

import numpy as np

# Parameters
K_max = 10000
r_max = 0.45
T = 0.7*K_max
d = 0.01
g = 1
sigma = 1
theta = 0.1
k = 2
b = 10
m_eff = 0.5
# Competition coefficients
aRS = 0.9
aSR = 0.15
# Quality of life coefficients
# a1 = 0.54
# a2 = 0
# a3 = 0.25
a1 = 0.5
a2 = 0.25
a3 = 0.25
#Initialization
m = np.linspace(0,1,1001)

# Cancer cells best response curve
uR = -k/b - m/(2*b*d) + np.sqrt((m * g)**2 + 4*m*g*b*d)/(2*b*d*g)
uR[uR<0] = 0
# Cancer population at the equilibrium point
xR = (K_max/(r_max*(1-aSR*aRS)))*(-m*np.exp(g*uR)/(k+b*uR) + aRS*m/k - d*np.exp(g*uR)\
    + r_max - aRS*r_max + aRS*d)
xS = (K_max/(r_max*(1-aSR*aRS)))*(aSR*m*np.exp(g*uR)/(k+b*uR) - m/k + aSR*d*np.exp(g*uR) \
    + r_max -d - aSR*r_max)
xS[xS<0] = 0
# Quality of life function defined over stabilization region
Q = 1 - a1*(((xR+xS)/K_max)**2) - a2*(uR**2) - a3*(m**2)
Q[np.add(xS,xR)>7000] = 0

# Stackelberg strategy (mS)
m_star = m[np.argmax(Q)]
print(m_star)
