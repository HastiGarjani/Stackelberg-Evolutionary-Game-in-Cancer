#####################################################
################## Local stability ##################
import numpy as np
import matplotlib.pyplot as plt
K_max = 10000
r_max = 0.45
d = 0.01
g = 0.8
sigma = 1
a = 2
b = 10
############### growth rate ##############
epsilon = 0
m = 0.723
a21 = 0.9
a12 = 0.15
##########################
u = -a/b + (-m*g+np.sqrt((m*g)**2+4*m*g*d*b))/(2*b*d*g)
r_u = r_max*np.exp(-g*u)

h1 = K_max*(1 - d/r_max - m/(a*r_max))
h2 = K_max*(1 - d/r_u - m/((a+b*u)*r_u))
h3 = K_max*(1 - m*b/(g*r_u*((a+b*u)**2)))
#### Equilibrium points
x_R = (h2 - a21*h1)/(1 - a12*a21)
x_S = h1 - a12*x_R

f1_x_S = r_max*(1-(2*x_S+a12*x_R)/K_max) - d - m/a 
f2_x_S = -a21*x_R*(r_u/K_max)
f3_x_S = sigma*g*r_u*a21/K_max

f1_x_R = -a12*x_S*r_max/K_max
f2_x_R = r_u*(1-(a21*x_S+x_R)/K_max) - d - m/(a+b*u) - x_R*r_u/K_max
f3_x_R = sigma*g*r_u/K_max

f1_u = 0
f2_u = x_R*(-g*r_u*(1-(a21*x_S+x_R)/K_max) + (m*b)/((a+b*u)**2))
f3_u = sigma*((g**2)*r_u*(1-(a21*x_S+x_R)/K_max) - (2*m*(b**2))/((a+b*u)**3))
J = np.array([[f1_x_S, f1_x_R, f1_u],[f2_x_S, f2_x_R, f2_u],[f3_x_S, f3_x_R, f3_u]])
eigs, v = np.linalg.eig(J)
print(eigs)
