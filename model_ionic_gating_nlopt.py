# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Import libraries
# 
# This notebook requires matplotlib for plotting, NumPy for numerical operations, and SciPy for nonlinear data-fitting. The time and datetime modules are included only for timing the execution of the fitting operation.

# <codecell>

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optim
import scipy.linalg as LA
from scipy.integrate import odeint

import sys
import time
import datetime
import collections

# Use this to render inline graphics in ipython
%matplotlib inline
from mpltools import style
style.use('ggplot')

# <markdowncell>

# # Read the data files.
# 
# The data files I received contained the ionic current, and the integrated gating current (i.e, the gating charge).
# 
# To recover the gating current, we have to take the derivative of the gating charge.

# <codecell>

# Manually generate variable potential sweep
gv = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10]

# Manually generate time course data
dt = 0.00006
t = np.arange(0, 0.3, dt)

# Read in the data files (use PANDAS instead?)
# Take the gradient of the gating charge to recover current
ii = np.genfromtxt('ii.txt')
gc = np.genfromtxt('gi.txt')
gi = np.gradient(gc)[0]

# Neglect the first few points and last few points
gc = gc[10:-40]
gi = gi[10:-40]
ii = ii[10:-40]
t  =  t[10:-40]

def plotall(i, g):
    plt.figure()
    plt.title('Ionic Current')
    plt.plot(t, i)

    plt.figure()
    plt.title('Gating Current')
    plt.xlabel('Time (seconds)')
    plt.plot(t, g)
    return

# Show data
plotall(ii, gi)

# <markdowncell>

# # Define the Markov System
# 
# Here we use a modified Hodgkin-Huxley model to fit the data. We assume there are four closed states (the last two with partial conductance), an open conducting state, and an inactivated state. 
# 
# $
# C_0
# \begin{aligned}
# & 4\alpha \\
# & \rightleftharpoons \\
# & \beta
# \end{aligned} 
# C_1
# \begin{aligned}
# & 3\alpha \\
# & \rightleftharpoons \\
# & 2\beta
# \end{aligned} 
# C_2
# \begin{aligned}
# & 2\alpha \\
# & \rightleftharpoons \\
# & 3\beta
# \end{aligned} 
# C_3
# \begin{aligned}
# & \alpha \\
# & \rightleftharpoons \\
# & 4\beta
# \end{aligned} 
# C_4
# \begin{aligned}
# & k_{co} \\
# & \rightleftharpoons \\
# & k_{oc}
# \end{aligned} 
# Op
# \begin{aligned}
# & k_{oi} \\
# & \rightleftharpoons \\
# & k_{io}
# \end{aligned} 
# I
# $
# 
# 
# The forward and backward voltage-dependent kinetic rates can be described by the following formula:
# 
# $
# \begin{align}
# \alpha=\alpha_0 \exp(\frac{z_f e^- V}{k_B T}) \\
# \beta=\beta_0 \exp(\frac{z_b e^- V}{k_B T})
# \end{align}
# $
# 
# Before each experimental trace, the membrane is held at a negative "hyperpolarized" potential for an extended period of time until the system reaches equilibrium. We can impose this initial condition by noting that it is in steady state, and therefore $\frac{d C_0}{dt} = 0$

# <codecell>

# Rate parameters - memb potential in mV
def rate(c, z, mV):

    kB = 0.086173; # in e mV/K
    T = 300; # in K
 
    rate = c * np.exp( z * mV / kB / T )
    return rate


# Define the system
#
#        4a        3a        2a        1a       kco
#   C0 <====> C1 <====> C2 <====> C3 <====> C4 <===> Op
#        1b        2b        3b        4b       koc
#
#

def Mat(prm, V):
    #prm = np.abs(prm)
    c1, c2, zf, zb, kco, koc, kio, koi, Gm1, Gm2, Gm3, Gm4 = prm    
    a = rate(c1, zf, V)
    b = rate(c2, -zb, V)

    M = [[-4*a,      b,        0,      0,        0,         0,     0],
         [ 4*a, -b-3*a,      2*b,      0,        0,         0,     0],
         [   0,    3*a, -2*b-2*a,    3*b,        0,         0,     0],
         [   0,      0,      2*a, -3*b-a,      4*b,         0,     0],
         [   0,      0,        0,      a, -4*b-kco,       koc,     0],
         [   0,      0,        0,      0,      kco,  -koc-koi,   kio],
         [   0,      0,        0,      0,        0,       koi,  -kio]]

    return M


def steadystate_alt(prm, Vholding):
    # For given set of parameters, find the steady state condition
    # i.e. Mx=0, and sum x = 1
    c1, c2, zf, zb, kco, koc, kio, koi, Gm1, Gm2, Gm3, Gm4 = prm    
    a = rate(c1, zf, Vholding)
    b = rate(c2, -zb, Vholding)

    #C0i = 1 / (1 + 4 * (a/b)**1 + 6 * (a/b)**2 + 4 * (a/b)**3 + 1 * (a/b)**4 + (kco / koc) * (a/b)**4)
    C0i = (b**4 * koc) / (b**4*koc + 4*a*b**3*koc + 6*a**2*b**2*koc + 4*a**3*b*koc + a**4*koc + a**4*kco)
    C1i = 4 * a**1 / b**1 * C0i
    C2i = 6 * a**2 / b**2 * C0i
    C3i = 4 * a**3 / b**3 * C0i
    C4i = 1 * a**4 / b**4 * C0i
    Opi = 1 * a**4 / b**4 * C0i * kco / koc
    
    ybsl = [C0i, C1i, C2i, C3i, C4i, Opi, Ii]
    
    return ybsl


def f(y, t, prm, V):
    # This function defines the system of ordinary differential equations (ODE)
    #C0, C1, C2, C3, C4, Op, I = y
    
    # Matrix is unique at every voltage
    M = Mat(prm, V)
    
    # This represents the system of equations, where y_prime is the rate of change of each state
    y_prime = np.dot(M, y)

    return y_prime


def steadystate(prm, Vholding):
    # For given set of parameters, find the steady state condition
    # i.e. Mx=0, and sum x = 1

    m = Mat(prm, Vholding)

    # Add an extra row for steady state condition
    M = np.vstack((m, np.ones(len(m))))

    # Find initial populations which satisfy steady state condition
    ybsl, residuals, rank, s = LA.lstsq(M, [0,0,0,0,0,0,0,1])
    
    return ybsl


# Just for testing --- do not use
#pint repr(constraints(Pinit, -90))
#print "ASD"
##f([1,0,0,0,0,0,0], t, Pini, 88)
#print steadystate(Pinit, -90) -steadystate_alt(Pinit, -90)

# <markdowncell>

# # -----------------------------------------------------
# 
# # Initialize Parameters

# <codecell>

# organize parameters into a named tuple for easy reference
Params = collections.namedtuple('Params', 'index name value lo hi')
prms = ((0,  'c1',      1e+0,    1e-8,  1e4),
        (1,  'c2',      1e+0,    1e-8,  1e4),
        (2,  'zf',      1e+0,    0,       2),
        (3,  'zb',      1e+0,    0,       2),
        (4,  'kco',     1e+0,    1e-8,  1e4),
        (5,  'koc',     1e+0,    1e-8,  1e4),
        (6,  'kio',        0,    0,     0),
        (7,  'koi',        0,    0,     0),
        (8,  'Gm1',     1,       0,       1),
        (9,  'Gm2',     1,       0,       1),
        (10, 'Gm3',     0,       0,       1),
        (11, 'Gm4',     0,       0,       1))


# convert named tuple to two lists: pinit and bounds
def getprms(prms):
    Pinit=np.zeros(len(prms))
    BndLo=np.zeros(len(prms))
    BndHi=np.zeros(len(prms))

    for prm in map(Params._make, prms):
        # print "%s %s %s" % (p.index, p.name, p.value)
        Pinit[prm.index] = prm.value
        BndLo[prm.index] = prm.lo
        BndHi[prm.index] = prm.hi
    
    Bounds = zip(BndLo, BndHi)
    return Pinit, Bounds


Pinit, Bounds = getprms(prms)

# <markdowncell>

# # Evaluate the model
# 
# The model can be evaluated by solving the equations and seeing if they fit the data. The set of differential equations produce a time-dependent population of states, which give rise to ionic and gating currents.

# <codecell>

def evalmod(f, t, prm, mV, Vh):

    #prm = np.abs(prm)
    c1, c2, zf, zb, kco, koc, kio, koi, Gm1, Gm2, Gm3, Gm4 = prm
    a = rate(c1, zf, mV)
    b = rate(c2, -zb, mV)

    #solve the system of ODE
    y0 = steadystate(prm, Vh)
    soln = odeint(f, y0, t, args=(prm,mV))

    #these are the solved traces for each state    C0t = soln[:,0]
    C0t = soln[:,0]
    C1t = soln[:,1]
    C2t = soln[:,2]
    C3t = soln[:,3]
    C4t = soln[:,4]
    Opt = soln[:,5]

    # Ek is reversal potential 
    Ek = -88
    ionic = Gm1 * (mV - Ek) * (Opt + Gm3 * (C4t + Gm4 * (C3t + (0 * C2t))))

    #gating current
    C0C1 = (4*a*C0t - 1*b*C1t)
    C1C2 = (3*a*C1t - 2*b*C2t)
    C2C3 = (2*a*C2t - 3*b*C3t)
    C3C4 = (1*a*C3t - 4*b*C4t)
    gating = Gm2 * (zf + zb) * (C0C1 + C1C2 + C2C3 + C3C4)

    return ionic, gating


def evalmodgroup(f, t, prm, gv, Vh):
    ionic = np.zeros(np.shape(ii))
    gating = np.zeros(np.shape(gi))
    for i in np.arange(len(gv)):
        mp = gv[i]
        ionic[:,i], gating[:,i] = evalmod(f, t, prm, mp, Vh)
    return ionic, gating


# Evaluate with initial parameters as a test
Vh = -90
testi, testg = evalmodgroup(f, t, Pinit, gv, Vh)
plotall(testi, testg)

# <markdowncell>

# # Define the objective function
# 
# The objective function is a scoring function whose global minimum corresponds to the best fit of the data.

# <codecell>

def objTuple(prm, ii, gi, gv, Vh):

    sim_ionic, sim_gating = evalmodgroup(f, t, prm, gv, Vh)

    #noise_ionic = .005
    #noise_gating = .0002

    noise_ionic = 447e-5
    noise_gating = 9e-5

    err2_ionic  = np.square((ii - sim_ionic)/noise_ionic)
    err2_gating = np.square((gi - sim_gating)/noise_gating)

    err2_2d_arr = err2_ionic + err2_gating
    err2_1d_arr = np.sum(err2_2d_arr, axis=1)
    err2_sum = np.sum(err2_1d_arr)

    return (err2_sum, prm)


def objFunc(prm):
    return objTuple(prm, ii, gi, gv, Vh)[0]


def saveHistory(iter, history, current, tic, fid):

    # If we encounter a new lowest value, then update file and stdout and save to history
    if (current[0] < history[-1][0]):

        # numpy array not compatible with nlopt output it seems
        currentmod = (current[0], current[1].tolist())
        history.append(currentmod)

        # Get time in seconds since tic
        toc = time.time()
        tictoc = int(toc - tic)
        tocstr = str(datetime.timedelta(seconds = tictoc))
        
        # Write to file each time we reach a new low
        line = "%s %5s %.5e %s" % (tocstr, iter, current[0], repr(current[1]))
        fid.write(line+'\n')

        # Print useful information when every nth low is found
        if (len(history) % 500== 0):
            line = "%s Nfeval %4s %.5e %s" % (tocstr, iter, current[0], repr(current[1]))
            print line

    return history

# <markdowncell>

# # NLopt algorithm

# <codecell>

import nlopt


def nlopt_minimizer(prm, ii, gi, gv, Vh, alg_name):
    print "\nNLopt version %s.%s.%s" % (nlopt.version_major(),nlopt.version_minor(),nlopt.version_bugfix())
    print "Initial Parameters: %s" % prm

    # Create new file with unique name for saving output
    now = datetime.datetime.now()
    nowstr = now.strftime('%Y%m%d_%H%M%S')
    fname = alg_name + '_' + nowstr
    fid = open(fname, 'w', 1)
    
    # Create new instance
    algorithm = eval('nlopt.' + alg_name)
    myopt = nlopt.opt(algorithm, len(prm))
    print myopt.get_algorithm_name()
    sys.stdout.flush()

    #---------------------------------------------------------------------
    # This is a wrapper for the objective function + approximate gradient
    # The wrapper is used to include output to file and stdout
    global feval_iter
    feval_iter = 0
    def nlopt_f(x, grad):

        current = objTuple(x, ii, gi, gv, Vh)

        if grad.size > 0:
            # approximate gradients using finite differences
            epsilon = 1e-8
            gradpos = optim.approx_fprime(x, objFunc, +0.5 * epsilon)
            gradneg = optim.approx_fprime(x, objFunc, -0.5 * epsilon)
            grad[:] = 0.5 * (gradpos + gradneg)
            
        #### This is only for printing and saving to file
        global feval_iter
        saveHistory(feval_iter, hist, current, tic, fid)
        feval_iter += 1

        return current[0]
    #---------------------------------------------------------------------

    # Which objective function to use
    myopt.set_min_objective(nlopt_f)

    # Set bounds
    myopt.set_lower_bounds(np.asarray(Bounds)[:,0])
    myopt.set_upper_bounds(np.asarray(Bounds)[:,1])
    
    # Stopping criteria
    #myopt.set_maxtime(3)       # Stopping parameters - maxtime in seconds
    #myopt.set_maxeval(200)       # Stopping parameters - maxtime in seconds
    myopt.set_ftol_rel(1e-7)    # stop when obj function is converging-7
    myopt.set_xtol_rel(1e-4)    # stop when prms are not changing much-4    

    # Perform the optimization and time it
    tic = time.time()
    xopt = myopt.optimize(prm)
    toc = time.time()
    tocstr = str(datetime.timedelta(seconds = int(toc - tic)))
    print "Total time elapsed: %s" % tocstr
    fid.write("Total time elapsed: %s\n" % tocstr)


    # Retrieve the value from the last optimize call
    opt_val = myopt.last_optimum_value()
    result = myopt.last_optimize_result()

    print "minval: %s" % opt_val 
    print "exitcode: %s" % result 
    print "Optimized Parameters:"
    print repr(xopt)

    fid.write("minval: %s\n" % opt_val)
    fid.write("exitcode: %s\n" % result)
    fid.write("Optimized Parameters:")
    fid.write(repr(xopt))

    # Close the output file
    fid.close()

    return (xopt, opt_val, result)


sol_pre = np.array([  1.21567533e+01,   6.00404023e+00,   7.74151911e-01,
         8.01603948e-02,   1.06421292e+00,   5.76597001e-01,
         1.02931988e-02,   3.46634885e-05,   1.00000000e+00,
         1.00000000e+00])


hist = []; hist.append(tuple([u.tolist() for u in objTuple(Pinit, ii, gi, gv, Vh)]))

# Execute serial optimizations
#sol_pre, val, code = nlopt_minimizer(Pinit, ii, gi, gv, Vh, 'LD_LBFGS')

try:
    sol_pre, val, code = nlopt_minimizer(hist[-1][1], ii, gi, gv, Vh, 'LD_LBFGS')
except:
    pass

try:
    sol, val, code = nlopt_minimizer(hist[-1][1], ii, gi, gv, Vh, 'LN_BOBYQA')
except:
    pass

# <markdowncell>

# # Assess
# 
# Given a function and some parameters, I can calculate simulated data. Furthermore, given some experimental data, I can calculate residual error for assessing the quality of the parameters.

# <rawcell>

# '''
#     # This function is used to output stuff at every iteration
#     def callbackBH(x, f, accept):
#         
#         Xi = x
#         print Xi
#         print f
#         
#         # Save history of parameters to arrays for checking convergence
#         global output_prms
#         global output_objf
#         output_prms = np.vstack((output_prms, Xi))
#         output_objf = np.vstack((output_objf, objfun(Xi, ii, gi, gv, Vh)))
#         
#         # Print out useful messages to stdout every nth iterations
#         nfeval = len(output_objf)-1
#         if (nfeval % 1 == 0):
#             toc = time.time()
#             tocstr = str(datetime.timedelta(seconds = int(toc - tic)))
#             print "Time Elapsed %s   Nfeval %s   Objective %s" % (tocstr, nfeval, output_objf[nfeval])
#             sys.stdout.flush()
#             
#         return 
# '''
# 
# t2 = np.array([  1.44895673e+01,   2.57852354e+00,   6.97928419e-01,
#          6.36162325e-01,   9.38529529e-01,   7.79554436e-01,
#          6.11311351e-03,   1.52647194e-05,   1.00000000e+00,
#          1.00000000e+00])
# 
# Pinit = t2
# def assess(Pinit, ii, gi, gv):
#     
#     print "Initial parameters used for solver:"
#     print Pinit
#     print ""
#     
#     
#     # This function is used to output stuff at every iteration
#     def callbackF(Xi):
#         
#         # Save history of parameters to arrays for checking convergence
#         global output_prms
#         global output_objf
#         output_prms = np.vstack((output_prms, Xi))
#         output_objf = np.vstack((output_objf, objfun(Xi, ii, gi, gv, Vh)))
#         
#         # Print out useful messages to stdout every nth iterations
#         nfeval = len(output_objf)-1
#         if (nfeval % 20 == 0):
#             toc = time.time()
#             tocstr = str(datetime.timedelta(seconds = int(toc - tic)))
#             print "Time Elapsed %s   Nfeval %s   Objective %s" % (tocstr, nfeval, output_objf[nfeval])
#             sys.stdout.flush()
#             
#         return 
# 
#     
#     # Start timer
#     tic = time.time()
# 
#     # Sequential Least Squares Programming (or not)
#     slsqp = optim.minimize(objfun, Pinit, args=(ii, gi, gv, Vh), 
#                            method='L-BFGS-B', callback=callbackF)
#     minimizer_kwargs = {'method':'L-BFGS-B', 'args':(ii, gi, gv, Vh)}
#     #slsqp = optim.basinhopping(objfun, Pinit, minimizer_kwargs=minimizer_kwargs, callback=callbackBH, T=10, stepsize=1)
#     
#     # get best parameter solution
#     sol = slsqp.x
#     
#     # Output time elapsed to stdout
#     toc = time.time()
#     print str(datetime.timedelta(seconds = int(toc - tic)))
#         
#     print "--------------------------------------"
#     
#         
#     # Output best values    
#     print "\n\nBest parameters"
#     print repr(sol)
# 
#     print "Initial condition"
#     print steadystate(sol, Vh)
#     
#     print "\nTime of Execution (hh:mm:ss)"
#     toc = time.time()
#     print str(datetime.timedelta(seconds = int(toc - tic)))
# 
#     return
#     
# 
# #output_prms = Pinit
# #output_objf = objfun(Pinit, ii, gi, gv, Vh)
# #assess(Pinit, ii, gi, gv)
# #print output_prms[-1,:]

# <markdowncell>

# # Plot results and error

# <codecell>

# get curves with best parameters

sol = hist[-1][1]

def plot_sim_expt(f, t, prm, ii, gi, gv, Vh):

    # Get simulated data
    sim_i, sim_g = evalmodgroup(f, t, prm, gv, Vh)
    
    # Plot experimental and simulated data
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,2,1)
    plt.title('Ionic')
    plt.plot(t, ii)
    plt.plot(t, sim_i, 'k')

    plt.subplot(1,2,2)
    plt.title('Gating')
    plt.plot(t, gi)
    plt.plot(t, sim_g, 'k')
    
    # Plot residual error
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.title('Ionic Error')
    plt.ylim([-.04, .04])
    plt.plot(t, sim_i - ii)
    
    plt.subplot(1,2,2)
    plt.title('Gating Error')
    plt.ylim([-.001, .001])
    plt.plot(t, sim_g - gi)
    
    # Plot histograms of residuals    
    for i in np.arange(np.size(gv)):   
        
        plt.figure(5, figsize=(12,4))
        plt.suptitle('Residual error on ionic current')
        plt.subplot(2,5,i+1)
        plt.xlim([-.04,.04])
        plt.hist(sim_i[:,i]-ii[:,i], bins=20)
        #print np.std(sim_i[:,i]-ii[:,i])
 
        plt.figure(6, figsize=(12,4))
        plt.suptitle('Residual error on gating current')
        plt.subplot(2,5,i+1)
        plt.xlim([-.001,.001])
        plt.hist(sim_g[:,i]-gi[:,i], bins=20)
        #print np.std(sim_g[:,i]-gi[:,i])

    print "test"
    print np.std(sim_i[:,]-ii[:,])
    print np.std(sim_g-gi)


print sol
plot_sim_expt(f, t, sol, ii, gi, gv, Vh)

# <markdowncell>

# # Convergence of solution and initial condition

# <codecell>

# Plot history of output parameters and initial conditions to show convergence
output_objf = [u for u,v in hist]
output_prms = [v for u,v in hist]

def show_convergence(output_prms, output_objf):
    # Show convergence of parameters

    plt.figure(figsize=(12,8))
    plt.suptitle('Parameters')

    # Loop over each parameter --- should be 10
    for i in np.arange(np.shape(output_prms)[1]):
        plt.subplot(5,3,i+1)
        plt.plot([w[i] for w in output_prms], label='%s'%i)
        plt.legend()
        
    plt.figure(figsize=(12,4))
    #for i in np.arange(len(output_y0)):
    #    plt.subplot(1,7,i+1)
    #    plt.plot(output_y0[i,:])
    plt.title('Value of objective function')
    plt.semilogy(output_objf[5:-1])

show_convergence(output_prms, output_objf)
plt.semilogy(output_objf[5:-1])

# <codecell>


