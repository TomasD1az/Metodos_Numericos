# establich constants

# establish functions
def kinetic_function(theta):
    return

def potential_function(theta):
    return

def total_energy_function(theta, theta_dot):
    return

def pendulum_dinamic(theta):
    return

def position(theta):
    return

# numerical methods
def analitical(t):
    """
    The analitical function calculates the position of the pendulum at a given time using
    the analitical solution to the differential equation. It takes as input an array with 
    time values and returns an array with positions.
    
    :param t: Calculate the angle of the pendulum at each time step
    :return: The position and angle of the pendulum at a given time
    """
    return

def euler(f, h, x, y, z):
    """
    The euler function takes in a function f, a step size h, an initial value x0 and y0 for the 
    independent variable and dependent variable respectively. It then uses the euler method to approximate 
    the solution of the differential equation dy/dx = f(y) with initial condition y(x0) = y0. The function returns 
    an array containing all of these approximations.
    
    :param f: Define the function that is being used to calculate the values of x and y
    :param h: Determine the step size of the euler function
    :param x: Store the x values of the function
    :param y: Calculate the position of the ball
    :param z: Store the position of the object at each time step
    :return: The values of x and y
    """
    return

def runge_kuta_4(f): #reescribir
    """
    The runge_kuta_4 function is a numerical method for solving differential equations.
    It uses the Runge-Kutta 4th order method to approximate the solution of an ordinary 
    differential equation (ODE) with a given initial value. The function takes as input 
    the ODE and returns an array containing the approximated values of y(t). It also updates 
    the global variables rk4_ang, rk4_pos and angle_velocities.
    
    :param f: Pass the function that will be used in the runge-kutta method
    :return: The angle and the position of the pendulum at each time step
    :doc-author: Trelent
    """
    return

# evaluamos los metodos
def angle_error(angles, title,time): #reescribir
    """
    The angle_error function plots the angle error between the linearized solution and a given numerical method.
    
    :param angles: Plot the angles of the pendulum, and title is used to give a name to each graph
    :param title: Give a name to the plot and differentiate it from other plots
    :param time: Plot the angles in a graph, and the title is used to name the legend of that graph
    :return: A plot with the original solution and the numerical one
    """
    return

def energy(angles, method, time):
    """
    The energy function calculates the total, potential and kinetic energy of a pendulum.
        It takes as input:
            - angles: an array with the values of the angle at each time step.
            - method: a string indicating which method was used to calculate those angles (euler or rk4). 
            - time: an array with all times in which we calculated our angle values.
    
    :param angles: Calculate the energy function
    :param method: Plot the graph with the corresponding method
    :param time: Plot the energy function with the corresponding time
    :return: The total energy, potential and kinetic energies of the pendulum.
    """
    return