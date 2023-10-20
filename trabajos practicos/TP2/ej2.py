import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math

T_0 = 0
Y_0 = 11

T_FINAL = 15
H =0.5

def f_population(r,min,max,t,y):
        return r * y * (1 - (y/max)) * ((y/min) - 1)

def euler(r,A,K, t_i, y_i,h,f):
       return y_i + h*f(r,A,K, t_i,y_i)

def rk4(r,A,K, t_i,y_i,h,f):
       k1 = f(r,A,K, t_i,y_i)
       k2 = f(r,A,K, t_i + h/2,y_i + k1*h/2)
       k3 = f(r,A,K, t_i + h/2,y_i + k2*h/2)
       k4 = f(r,A,K, t_i + h, y_i + k3*h)
       return y_i + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


r = 0.5 #population growth rate
A = 10 #min population
K = 30 #max population


t = T_0

ts = [T_0]
ys_rk4 = [Y_0]
ys_euler = [Y_0]
ys_rk4_min = [9]
ys_euler_min = [9]
ys_rk4_max = [51]   
ys_euler_max = [51]

ys_rk4_r1 = [Y_0]
ys_rk4_r2 = [Y_0]
ys_euler_r1 = [Y_0]
ys_euler_r2 = [Y_0]

ys_rk4_h1 = [Y_0]
ys_rk4_h2 = [Y_0]
ys_euler_h1 = [Y_0]
ys_euler_h2 = [Y_0]

y_rk4 = Y_0
y_euler = Y_0
y_rk4_min = 9
y_euler_min = 9
y_rk4_max = 51
y_euler_max = 51

y_rk4_r1 = Y_0
y_euler_r1 = Y_0
y_rk4_r2 = Y_0
y_euler_r2 = Y_0

y_rk4_h1 = Y_0
y_euler_h1 = Y_0
y_rk4_h2 = Y_0
y_euler_h2 = Y_0


while t < T_FINAL:
       y_rk4 = rk4(r,A,K, t,y_rk4,H,f_population)
       y_euler = euler(r,A,K, t,y_euler,H,f_population)

       y_rk4_min = rk4(r,A,K, t,y_rk4_min,H,f_population)
       y_euler_min = euler(r,A,K, t,y_euler_min,H,f_population)
       y_rk4_max = rk4(r,A,K, t,y_rk4_max,H,f_population)
       y_euler_max = euler(r,A,K, t,y_euler_max,H,f_population) 

       y_rk4_r1 = rk4(0.25,A,K, t,y_rk4_r1,H,f_population)
       y_euler_r1 = euler(0.25,A,K, t,y_euler_r1,H,f_population)
       y_rk4_r2 = rk4(0.75,A,K, t,y_rk4_r2,H,f_population)
       y_euler_r2 = euler(0.75,A,K, t,y_euler_r2,H,f_population)

       y_rk4_h1 = rk4(r,A,K, t,y_rk4_r1,0.1,f_population)
       y_euler_h1 = euler(r,A,K, t,y_euler_r1,0.1,f_population)
       y_rk4_h2 = rk4(r,A,K, t,y_rk4_r2,1.5,f_population)
       y_euler_h2 = euler(r,A,K, t,y_euler_r2,1.5,f_population)

       t+= H

       ts.append(t)
       ys_rk4.append(y_rk4)
       ys_euler.append(y_euler)

       ys_rk4_min.append(y_rk4_min)
       ys_euler_min.append(y_euler_min)
       ys_rk4_max.append(y_rk4_max)
       ys_euler_max.append(y_euler_max)

       ys_rk4_r1.append(y_rk4_r1)
       ys_euler_r1.append(y_euler_r1)
       ys_rk4_r2.append(y_rk4_r2)
       ys_euler_r2.append(y_euler_r2)

       ys_rk4_h1.append(y_rk4_h1)
       ys_euler_h1.append(y_euler_h1)
       ys_rk4_h2.append(y_rk4_h2)
       ys_euler_h2.append(y_euler_h2)

#varianfo el N
plt.plot(ts, ys_euler, color = "red", label='Euler A<N<K')
plt.plot(ts, ys_rk4, color = "orange", label='RK4 A<N<K')

plt.plot(ts, ys_euler_min, color = "green", label='Euler N<A')
plt.plot(ts, ys_rk4_min, color = "yellow", label='RK4 N<A')

plt.plot(ts, ys_euler_max, color = "purple", label='Euler K<N')
plt.plot(ts, ys_rk4_max, color = "blue", label='RK4 K<N')

plt.xlabel('time')
plt.ylabel('population')
plt.legend()
plt.grid(True)
plt.show()

#Variando el r
plt.plot(ts, ys_euler, color = "red", label='Euler r=0.5')
plt.plot(ts, ys_rk4, color = "orange", label='RK4 r=0.5')

plt.plot(ts, ys_euler_r1, color = "green", label='Euler r=0.25')
plt.plot(ts, ys_rk4_r1, color = "yellow", label='RK4 r=0.25')

plt.plot(ts, ys_euler_r2, color = "purple", label='Euler r=0.75')
plt.plot(ts, ys_rk4_r2, color = "blue", label='RK4 r=0.75')

plt.xlabel('time')
plt.ylabel('population')
plt.legend()
plt.grid(True)
plt.show()

#Variando el h
plt.plot(ts, ys_euler, color = "red", label='Euler h=0.5')
plt.plot(ts, ys_rk4, color = "orange", label='RK4 h=0.5')

plt.plot(ts, ys_euler_h1, color = "green", label='Euler h=0.1')
plt.plot(ts, ys_rk4_h1, color = "yellow", label='RK4 h=0.1')

plt.plot(ts, ys_euler_h2, color = "purple", label='Euler h=1.5')
plt.plot(ts, ys_rk4_h2, color = "blue", label='RK4 h=1.5')

plt.xlabel('time')
plt.ylabel('population')
plt.legend()
plt.grid(True)