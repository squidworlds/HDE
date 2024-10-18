# Import packages
import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
#  defining the function in the RHS of the ODE given in the question

def dy_dt(y, t):
    '''Define our ODE in terms of y and t.'''
    return 4*t - 3*y

# Euler scheme
def ode_Euler(func, initialTime, finalTime, nSteps, y0):
    '''
    integrates the system of y' = func(y, t) using forward Euler method
    for the time steps in times and given initial condition y0
    ----------------------------------------------------------
    inputs:
        func: the RHS function in the system of ODE
        times: the points in time (or the span of independent variable in ODE)
        y0: initial condition (make sure the dimension of y0 and func are the same)
    output:
        y: the solution of ODE. 
        Each row in the solution array y corresponds to a value returned in column vector t
    '''

    # Initial conditions
    y0 = np.array(y0)

    # Dimension of ODE
    n = y0.size

    # Step size
    h = (finalTime - initialTime)/nSteps

    # Array of time values
    times = np.linspace(initialTime, finalTime, nSteps + 1)

    # Initialise array of y values, starting with y0
    y = np.zeros([nSteps + 1, n])
    y[0,:] = y0
    
    # Loop for timesteps from 0 to nSteps-1
    for k in range(nSteps):
        
        # Euler method
        y[k+1, :] = y[k, :] + h*func(y[k, :], times[k])
        
    return y, times

# Adams-Bashforth 2
def ode_AB2(func, initialTime, finalTime, nSteps, y0):
    '''
    integrates the system of y' = func(y, t) using Adams-Bashforth 2-step method
    for the time steps in times and given initial condition y0
    ----------------------------------------------------------
    inputs:
        func: the RHS function in the system of ODE
        initialTime: smallest time value
        finalTime: largest time value
        nSteps: difference between each time value
        y0: initial condition
    output:
        y: the solution of ODE
        times: array of times
        Each row in the solution array y corresponds to a value returned in column vector t
    '''
    # Initial conditions
    y0 = np.array(y0)

    # Dimension of ODE
    n = y0.size

    # Step size
    h = (finalTime - initialTime)/nSteps

    # Array of time values
    times = np.linspace(initialTime, finalTime, nSteps + 1)

    # Initialise array of y values, starting with y0
    y = np.zeros([nSteps + 1, n])
    y[0,:] = y0
    
    # First step using Euler
    y[1,:] = y[0,:] + h*func(y[0, :], times[0])
    
    # Now applying AB2 for the next t values
    for k in range(1, nSteps):
        
        # Applying formula
        y[k+1,:] = y[k,:] + h*(1.5*func(y[k, :], times[k])-0.5*func(y[k-1, :], times[k-1]))
        
    return y, times

# Adams-Bashforth 3
def ode_AB3(func, initialTime, finalTime, nSteps, y0):
    '''
    integrates the system of y' = func(y, t) using Adams-Bashforth 3-step method
    for the time steps in times and given initial condition y0
    ----------------------------------------------------------
    inputs:
        func: the RHS function in the system of ODE
        initialTime: smallest time value
        finalTime: largest time value
        nSteps: difference between each time value
        y0: initial condition
    output:
        y: the solution of ODE
        times: array of times
        Each row in the solution array y corresponds to a value returned in column vector t
    '''
    # Initial conditions
    y0 = np.array(y0)

    # Dimension of ODE
    n = y0.size

    # Step size
    h = (finalTime - initialTime)/nSteps

    # Array of time values
    times = np.linspace(initialTime, finalTime, nSteps + 1)

    # Initialise array of y values, starting with y0
    y = np.zeros([nSteps + 1, n])
    y[0,:] = y0
    
    # First step using Euler
    y[1,:] = y[0,:] + h*func(y[0, :], times[0])
    
    # Second step using AB2
    y[2, :] =  y[1,:] + h*(1.5*func(y[1, :], times[1])-0.5*func(y[0, :], times[0]))
    
    # Third step onwards using AB3
    for k in range(2, nSteps):

        # Applying formula
        y[k+1,:] = y[k,:] + h*((23/12)*func(y[k, :], times[k])-(16/12)*func(y[k-1, :], times[k-1])+(5/12)*func(y[k-2, :], times[k-2]))
        
    return y, times

# Define solutions for each method, and times
y_euler, times = ode_Euler(dy_dt, 0, 0.5, 10, 1)
y_AB2, times = ode_AB2(dy_dt, 0, 0.5, 10, 1)
y_AB3, times = ode_AB3(dy_dt, 0, 0.5, 10, 1)
# Create plot of all ODE solutions

# Define the three graphs
plt.plot(times, y_euler, 'b', label='Euler')
plt.plot(times, y_AB2, 'r', label='AB2')
plt.plot(times, y_AB3, 'g', label='AB3')

# Title
plt.title("Using numerical methods to solve y'=4t-3y with y(0)=1")

# Location of legend
plt.legend(loc='best')

# Axes and grid
plt.xlabel('t')
plt.ylabel('y')
plt.grid()

import pandas as pd
from pandas import DataFrame

# Define the three dataframes of solutions using methods
df_e = DataFrame(data = y_euler, index = times, columns = ["Euler"])
df_AB2 =  DataFrame(data = y_AB2, index = times, columns = ["AB2"])
df_AB3 = DataFrame(data = y_AB3, index = times, columns = ["AB3"])

# Join the three dataframes together
df = df_e.join(df_AB2, how='left')
df = df.join(df_AB3, how='left')
df
