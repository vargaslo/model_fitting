
# # Import libraries
# 
# This notebook requires matplotlib for plotting, NumPy for numerical operations, and SciPy for nonlinear data-fitting. The time and datetime modules are included only for timing the execution of the fitting operation.

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scipy.linalg as LA
from scipy.integrate import odeint

import sys
import time
import datetime
import collections

# Use this to render inline graphics in ipython



# # Read the data files.
# 
# The data files I received contained the ionic current, and the integrated gating current (i.e, the gating charge).
# 
# To recover the gating current, we have to take the derivative of the gating charge.

# In[2]:

# Manually generate variable potential sweep
gv = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10]

# Manually generate time course data
dt = 0.00006
t = np.arange(0, 0.3, dt)

# Read in the data files (use PANDAS instead?)
gi = np.genfromtxt('gi.txt')
ii = np.genfromtxt('ii.txt')

# Normalize ionic current
ii = ii / np.max(ii)

# Normalize gating charge
gc = gi / np.max(gi)

# Take the gradient of the integrated sweeps to recover current
gi = np.gradient(gi)[0] / np.max(np.gradient(gi)[0])

# Neglect the first few points and last few points
gc = gc[10:-40]
gi = gi[10:-40]
ii = ii[10:-40]
t  =  t[10:-40]

def plotall(i, g):
    plt.figure()
    plt.title('Ionic Current')
    plt.plot(i)

    plt.figure()
    plt.title('Gating Current')
    plt.plot(g)
    return

# Show data
plotall(ii, gi)


# Out[2]:

# image file:

# image file:

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

# In[3]:

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
    prm = np.abs(prm)
    c1, c2, zf, zb, kco, koc, kio, koi, Gm1, Gm2, Gm3, Gm4 = prm    
    a = rate(c1, zf, V)
    b = rate(c2, -zb, V)

    M = [[-4*a,      b,        0,      0,        0,        0,    0],
         [ 4*a, -b-3*a,      2*b,      0,        0,        0,    0],
         [   0,    3*a, -2*b-2*a,    3*b,        0,        0,    0],
         [   0,      0,      2*a, -3*b-a,      4*b,        0,    0],
         [   0,      0,        0,      a, -4*b-kco,      koc,    0],
         [   0,      0,        0,      0,      kco, -koc-koi,  kio],
         [   0,      0,        0,      0,        0,      koi, -kio]]

    return M


def f(y, t, prm, V):
    # This function defines the system of ordinary differential equations (ODE)
    #C0, C1, C2, C3, C4, Op, I = y
    
    # Matrix is unique at every voltage
    M = Mat(prm, V)

    # This represents the system of equations, where y_prime is the rate of change of each state
    y_prime = np.dot(M,y)

    return y_prime


def ybaseline(prm, Vholding):
    # For given set of parameters, find the steady state condition
    # i.e. Mx=0, and sum x = 1

    m = Mat(prm, Vholding)

    # Add an extra row for steady state condition
    M = np.vstack((m, np.ones(len(m))))

    # Find initial populations which satisfy steady state condition
    ybsl, residuals, rank, s = LA.lstsq(M, [0,0,0,0,0,0,0,1])
    
    return ybsl
    
# Just for testing --- do not use
#f([1,0,0,0,0,0,0], t, Pini, 88)
#ybaseline(Pini, -88)


# # -----------------------------------------------------
# 
# # Initialize Parameters

# In[4]:

'''
These parameters are very good if needed for checking

Best parameters
[  23.  2.  1.  0.5  5.  9.  0.  0.  0.5  0.035  0.05  1.]
Initial condition
[  4.83376850e-01   9.48550088e-02   6.98016932e-03   2.28291178e-04
   2.79990675e-06   2.75872048e-07   4.14556605e-01]
   
'''

# organize parameters into a named tuple for easy reference
Params = collections.namedtuple('Params', 'index name value lo hi')
prms = ((0,  'c1',      1,       0,  1e4),
        (1,  'c2',      1,       0,  1e4),
        (2,  'zf',      1,       0,    2),
        (3,  'zb',      0,       0,    2),
        (4,  'kco',     1,       0,  1e4),
        (5,  'koc',     1,       0,  1e4),
        (6,  'kio',     0,       0,  1e4),
        (7,  'koi',     0,       0,  1e4),
        (8,  'Gm1',     1,       0,  1e4),
        (9,  'Gm2',     1,       0,  1e4),
        (10, 'Gm3',     0,       0,    1),
        (11, 'Gm4',     0,       0,    1))


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

# Assume the whole population start in C0 state
C0i = 1
C1i = 0
C2i = 0
C3i = 0
C4i = 0
Opi = 0
Ii  = 0

y0 = [C0i, C1i, C2i, C3i, C4i, Opi, Ii]


# # Evaluate the model
# 
# The model can be evaluated by solving the equations and seeing if they fit the data. The set of differential equations produce a time-dependent population of states, which give rise to ionic and gating currents.

# In[5]:

def evalmod(f, y0, t, prm, mV):

    prm = np.abs(prm)
    c1, c2, zf, zb, kco, koc, kio, koi, Gm1, Gm2, Gm3, Gm4 = prm
    a = rate(c1, zf, mV)
    b = rate(c2, -zb, mV)

    #solve the system of ODE
    soln = odeint(f, y0, t, args=(prm,mV))

    #these are the solved traces for each state    C0t = soln[:,0]
    C0t = soln[:,0]
    C1t = soln[:,1]
    C2t = soln[:,2]
    C3t = soln[:,3]
    C4t = soln[:,4]
    Opt = soln[:,5]
    It  = soln[:,6]

    # Ek is reversal potential 
    Ek = -88
    ionic = Gm1 * (mV - Ek) * (Opt + Gm3 * (C4t + Gm4 * (C3t + (0 * C2t))))

    #gating current
    C0C1 = (4*a*C0t - 1*b*C1t)
    C1C2 = (3*a*C1t - 2*b*C2t)
    C2C3 = (2*a*C2t - 3*b*C3t)
    C3C4 = (1*a*C3t - 4*b*C4t)
    gating = Gm2 * (zf - zb) * (C0C1 + C1C2 + C2C3 + C3C4)

    return ionic, gating


def evalmodgroup(f, y0, t, prm, gv):
    ionic = np.zeros(np.shape(ii))
    gating = np.zeros(np.shape(gi))
    for i in np.arange(len(gv)):
        mp = gv[i]
        ionic[:,i], gating[:,i] = evalmod(f, y0, t, prm, mp)
    return ionic, gating


# Evaluate with initial parameters as a test
testi, testg = evalmodgroup(f, y0, t, Pinit, gv)
plotall(testi, testg)


# Out[5]:

# image file:

# image file:

# # Define the objective function
# 
# The objective function is (for now) just the residual error.

# In[6]:

def residualerr2arr(prm, y0, ii, gi, gv):

    sim_ionic, sim_gating = evalmodgroup(f, y0, t, prm, gv)

    err2_ionic  = np.square(ii - sim_ionic)
    err2_gating = np.square(gi - sim_gating)

    err2_arr = err2_ionic + err2_gating
    return (np.sum(err2_arr, axis=1))

def residualerr2sum(prm, y0, ii, gi, gv):
    err2_arr = residualerr2arr(prm, y0, ii, gi, gv)
    return np.sqrt(np.sum(err2_arr))

# test if function works
#print residualerr2arr(Pinit, y0, ii, gi, gv)
#print residualerr2sum(Pinit, y0, ii, gi, gv)


# # Assess
# 
# Given a function and some parameters, I can calculate simulated data. Furthermore, given some experimental data, I can calculate residual error for assessing the quality of the parameters.

# In[7]:

def assess(Pinit, y0, ii, gi, gv, iterations):
    
    print "Initial parameters used for solver:"
    print Pinit
    print "Initial conditions used for solver"
    print y0
    print ""

    # Store the values at each iteration to see how they behave
    output_prms = np.zeros((len(Pinit), iterations))
    output_y0   = np.zeros((len(y0), iterations))

    # Start timer
    tic = time.time()
    toc = time.time()
    print str(datetime.timedelta(seconds = int(toc - tic)))
    
    for i in np.arange(iterations):
    
        # Sequential Least Squares Programming
        slsqp = opt.minimize(residualerr2sum, Pinit, args=(y0, ii, gi, gv), 
                            method='SLSQP', bounds=Bounds)
        
        # get best parameter solution
        sol = slsqp.x
       
        # Save history of parameters and initial populations
        output_y0[:,i] = y0
        output_prms[:,i] = sol


        # Output helpful messages to stdout
        print "Iteration: %s/%s Message: %s" % (i+1, iterations, slsqp.message)
        print "Function Eval %s" % slsqp.fun
        print "Nit: %s" % slsqp.nit
        print "curr y0: %s" % repr(y0)
        print "curr prm:\n%s" % repr(sol)

        # find baseline y values --- should be at Vholding
        ybl = ybaseline(sol, -90)
        fract = 1
        ybl_mod = fract * ybl + (1 - fract) * np.asarray(y0)
        
        # reassign Pinit and y0 for next loop iteration
        Pinit = sol
        y0 = ybl_mod

        # Output time elapsed to stdout
        toc = time.time()
        print str(datetime.timedelta(seconds = int(toc - tic)))
        print "--------------------------------------"

        sys.stdout.flush()
        # End for loop

        
    # Output best values    
    print "\n\nBest parameters"
    print repr(sol)

    print "Initial condition"
    print repr(ybl)
    
    print "\nTime of Execution (hh:mm:ss)"
    toc = time.time()
    print str(datetime.timedelta(seconds = int(toc - tic)))

    return (output_prms, output_y0)
    

# Iterate 20 times exhibits good convergence
iterations = 20
output_prms, output_y0 = assess(Pinit, y0, ii, gi, gv, iterations)


# Out[7]:

#     Initial parameters used for solver:
#     [ 1.  1.  1.  0.  1.  1.  0.  0.  1.  1.  0.  0.]
#     Initial conditions used for solver
#     [1, 0, 0, 0, 0, 0, 0]
#     
#     0:00:00
#     Iteration: 1/20 Message: Optimization terminated successfully.
#     Function Eval 69.3152705371
#     Nit: 3
#     curr y0: [1, 0, 0, 0, 0, 0, 0]
#     curr prm:
#     array([  5.11812814e-14,   2.26852771e+01,   2.00000000e+00,
#              4.13002965e-14,   1.01114507e+00,   9.99511014e-01,
#              0.00000000e+00,  -6.49693796e-16,   1.01279007e+00,
#              5.19584376e-14,   2.36212812e-01,   8.86833648e-03])
#     0:00:01
#     --------------------------------------
#     Iteration: 2/20 Message: Iteration limit exceeded
#     Function Eval 13.3295637309
#     Nit: 101
#     curr y0: array([  5.00000000e-01,   1.25767452e-17,   3.29597460e-17,
#              6.93889390e-17,   7.37257477e-18,   1.27675648e-15,
#              5.00000000e-01])
#     curr prm:
#     array([  2.29578131e+01,   1.90396456e+00,   1.09037029e+00,
#              5.57441503e-01,   2.70970955e-17,   3.26546994e+01,
#              2.27524435e-01,   2.41803656e+01,   5.43696252e-02,
#              4.03623478e-02,   5.28283864e-01,   7.57374849e-01])
#     0:02:08
#     --------------------------------------
#     Iteration: 3/20 Message: Optimization terminated successfully.
#     Function Eval 13.2844477141
#     Nit: 39
#     curr y0: array([  8.58450781e-01,   1.33553615e-01,   7.79160922e-03,
#              2.02030138e-04,   1.96442940e-06,  -5.07406617e-17,
#             -3.77475828e-15])
#     curr prm:
#     array([  2.18601409e+01,   1.79315459e+00,   1.02158070e+00,
#              5.89248880e-01,   1.37355257e-16,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.28353469e-02,
#              2.59632989e-02,   3.59888547e-01,   6.87630185e-01])
#     0:03:02
#     --------------------------------------
#     Iteration: 4/20 Message: Optimization terminated successfully.
#     Function Eval 13.2743514725
#     Nit: 12
#     curr y0: array([  8.39441859e-01,   1.50179034e-01,   1.00753295e-02,
#              3.00418514e-04,   3.35911904e-06,   2.08166817e-17,
#              7.77156117e-16])
#     curr prm:
#     array([  2.18535498e+01,   1.81854999e+00,   1.02335346e+00,
#              5.71627500e-01,  -4.27568262e-19,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.16927660e-02,
#              2.50685709e-02,   3.72864896e-01,   6.74908802e-01])
#     0:03:17
#     --------------------------------------
#     Iteration: 5/20 Message: Optimization terminated successfully.
#     Function Eval 13.2715312047
#     Nit: 17
#     curr y0: array([  8.33480643e-01,   1.55323754e-01,   1.08545421e-02,
#              3.37134048e-04,   3.92667532e-06,  -1.12757026e-17,
#             -2.05391260e-15])
#     curr prm:
#     array([  2.17200722e+01,   1.84800820e+00,   1.02170941e+00,
#              5.62081027e-01,  -9.01097580e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.08076077e-02,
#              2.47959567e-02,   3.82145683e-01,   6.79051835e-01])
#     0:03:39
#     --------------------------------------
#     Iteration: 6/20 Message: Optimization terminated successfully.
#     Function Eval 13.2704217425
#     Nit: 5
#     curr y0: array([  8.30976621e-01,   1.57474841e-01,   1.11908950e-02,
#              3.53456476e-04,   4.18637699e-06,  -3.03576608e-18,
#              4.99600361e-16])
#     curr prm:
#     array([  2.17198840e+01,   1.84813214e+00,   1.02178037e+00,
#              5.60836242e-01,  -7.36221075e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.09931177e-02,
#              2.47500440e-02,   3.80797793e-01,   6.76905918e-01])
#     0:03:46
#     --------------------------------------
#     Iteration: 7/20 Message: Optimization terminated successfully.
#     Function Eval 13.2701601781
#     Nit: 2
#     curr y0: array([  8.30372656e-01,   1.57992792e-01,   1.12728253e-02,
#              3.57475072e-04,   4.25098934e-06,  -1.34441069e-17,
#             -5.05151476e-15])
#     curr prm:
#     array([  2.17198840e+01,   1.84813218e+00,   1.02178086e+00,
#              5.60835852e-01,  -7.36183190e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.09917241e-02,
#              2.47568936e-02,   3.80797040e-01,   6.76905656e-01])
#     0:03:49
#     --------------------------------------
#     Iteration: 8/20 Message: Optimization terminated successfully.
#     Function Eval 13.2701601836
#     Nit: 1
#     curr y0: array([  8.30372710e-01,   1.57992746e-01,   1.12728180e-02,
#              3.57474713e-04,   4.25098355e-06,   1.86482774e-17,
#              2.72004641e-15])
#     curr prm:
#     array([  2.17198833e+01,   1.84813284e+00,   1.02178740e+00,
#              5.60832468e-01,  -7.34592925e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.09922511e-02,
#              2.47559948e-02,   3.80797404e-01,   6.76900610e-01])
#     0:03:51
#     --------------------------------------
#     Iteration: 9/20 Message: Optimization terminated successfully.
#     Function Eval 13.270160908
#     Nit: 1
#     curr y0: array([  8.30374428e-01,   1.57991273e-01,   1.12725845e-02,
#              3.57463234e-04,   4.25079861e-06,  -1.90819582e-17,
#              1.66533454e-16])
#     curr prm:
#     array([  2.17198833e+01,   1.84813285e+00,   1.02178744e+00,
#              5.60832445e-01,  -7.34592925e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.09918412e-02,
#              2.47562193e-02,   3.80797338e-01,   6.76900573e-01])
#     0:03:52
#     --------------------------------------
#     Iteration: 10/20 Message: Optimization terminated successfully.
#     Function Eval 13.2701609118
#     Nit: 1
#     curr y0: array([  8.30374440e-01,   1.57991263e-01,   1.12725829e-02,
#              3.57463155e-04,   4.25079734e-06,   1.04083409e-17,
#             -9.99200722e-16])
#     curr prm:
#     array([  2.17198833e+01,   1.84813285e+00,   1.02178747e+00,
#              5.60832421e-01,  -7.34592925e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.09920069e-02,
#              2.47563225e-02,   3.80797357e-01,   6.76900557e-01])
#     0:03:54
#     --------------------------------------
#     Iteration: 11/20 Message: Optimization terminated successfully.
#     Function Eval 13.2701609115
#     Nit: 1
#     curr y0: array([  8.30374441e-01,   1.57991262e-01,   1.12725828e-02,
#              3.57463149e-04,   4.25079725e-06,  -9.49761103e-17,
#             -6.10622664e-15])
#     curr prm:
#     array([  2.17198832e+01,   1.84813285e+00,   1.02178748e+00,
#              5.60832412e-01,  -7.34592925e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.09919259e-02,
#              2.47563482e-02,   3.80797349e-01,   6.76900544e-01])
#     0:03:56
#     --------------------------------------
#     Iteration: 12/20 Message: Optimization terminated successfully.
#     Function Eval 13.2701609124
#     Nit: 1
#     curr y0: array([  8.30374444e-01,   1.57991259e-01,   1.12725823e-02,
#              3.57463127e-04,   4.25079688e-06,   1.99493200e-17,
#             -1.66533454e-16])
#     curr prm:
#     array([  2.17198832e+01,   1.84813286e+00,   1.02178758e+00,
#              5.60832344e-01,  -7.34592925e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.09919499e-02,
#              2.47564705e-02,   3.80797357e-01,   6.76900469e-01])
#     0:03:57
#     --------------------------------------
#     Iteration: 13/20 Message: Optimization terminated successfully.
#     Function Eval 13.2701609182
#     Nit: 1
#     curr y0: array([  8.30374458e-01,   1.57991247e-01,   1.12725804e-02,
#              3.57463035e-04,   4.25079539e-06,  -3.16587034e-17,
#             -3.21964677e-15])
#     curr prm:
#     array([  2.17198832e+01,   1.84813286e+00,   1.02178760e+00,
#              5.60832328e-01,  -7.34584813e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.09919099e-02,
#              2.47564279e-02,   3.80797354e-01,   6.76900445e-01])
#     0:03:59
#     --------------------------------------
#     Iteration: 14/20 Message: Optimization terminated successfully.
#     Function Eval 13.2701609203
#     Nit: 1
#     curr y0: array([  8.30374464e-01,   1.57991243e-01,   1.12725797e-02,
#              3.57462999e-04,   4.25079482e-06,   2.55871713e-17,
#              1.27675648e-15])
#     curr prm:
#     array([  2.17198832e+01,   1.84813286e+00,   1.02178762e+00,
#              5.60832312e-01,  -7.34570935e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.09919534e-02,
#              2.47564131e-02,   3.80797360e-01,   6.76900426e-01])
#     0:04:01
#     --------------------------------------
#     Iteration: 15/20 Message: Optimization terminated successfully.
#     Function Eval 13.2701609217
#     Nit: 1
#     curr y0: array([  8.30374467e-01,   1.57991240e-01,   1.12725792e-02,
#              3.57462977e-04,   4.25079446e-06,   4.25007252e-17,
#              3.21964677e-15])
#     curr prm:
#     array([  2.17198832e+01,   1.84813287e+00,   1.02178765e+00,
#              5.60832297e-01,  -7.34557057e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.09919154e-02,
#              2.47564051e-02,   3.80797357e-01,   6.76900404e-01])
#     0:04:03
#     --------------------------------------
#     Iteration: 16/20 Message: Optimization terminated successfully.
#     Function Eval 13.2701609236
#     Nit: 1
#     curr y0: array([  8.30374472e-01,   1.57991236e-01,   1.12725786e-02,
#              3.57462944e-04,   4.25079393e-06,   3.38271078e-17,
#              8.32667268e-16])
#     curr prm:
#     array([  2.17198832e+01,   1.84813287e+00,   1.02178768e+00,
#              5.60832279e-01,  -7.34564680e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.09919547e-02,
#              2.47564000e-02,   3.80797363e-01,   6.76900383e-01])
#     0:04:04
#     --------------------------------------
#     Iteration: 17/20 Message: Optimization terminated successfully.
#     Function Eval 13.2701609254
#     Nit: 1
#     curr y0: array([  8.30374476e-01,   1.57991232e-01,   1.12725780e-02,
#              3.57462915e-04,   4.25079347e-06,  -2.94902991e-17,
#              1.55431223e-15])
#     curr prm:
#     array([  2.17198832e+01,   1.84813287e+00,   1.02178770e+00,
#              5.60832263e-01,  -7.34557741e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.09919170e-02,
#              2.47563969e-02,   3.80797360e-01,   6.76900362e-01])
#     0:04:06
#     --------------------------------------
#     Iteration: 18/20 Message: Optimization terminated successfully.
#     Function Eval 13.2701609272
#     Nit: 1
#     curr y0: array([  8.30374481e-01,   1.57991228e-01,   1.12725774e-02,
#              3.57462885e-04,   4.25079298e-06,  -5.20417043e-18,
#              1.11022302e-16])
#     curr prm:
#     array([  2.17198832e+01,   1.84813287e+00,   1.02178773e+00,
#              5.60832245e-01,  -7.34542368e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.09919563e-02,
#              2.47563941e-02,   3.80797366e-01,   6.76900341e-01])
#     0:04:08
#     --------------------------------------
#     Iteration: 19/20 Message: Optimization terminated successfully.
#     Function Eval 13.270160929
#     Nit: 1
#     curr y0: array([  8.30374485e-01,   1.57991225e-01,   1.12725768e-02,
#              3.57462856e-04,   4.25079251e-06,   5.55111512e-17,
#              3.60822483e-15])
#     curr prm:
#     array([  2.17198832e+01,   1.84813288e+00,   1.02178775e+00,
#              5.60832230e-01,  -7.34535429e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.09919185e-02,
#              2.47563920e-02,   3.80797363e-01,   6.76900319e-01])
#     0:04:09
#     --------------------------------------
#     Iteration: 20/20 Message: Optimization terminated successfully.
#     Function Eval 13.2701609307
#     Nit: 1
#     curr y0: array([  8.30374490e-01,   1.57991221e-01,   1.12725762e-02,
#              3.57462825e-04,   4.25079202e-06,   7.97972799e-17,
#              5.10702591e-15])
#     curr prm:
#     array([  2.17198832e+01,   1.84813288e+00,   1.02178777e+00,
#              5.60832212e-01,  -7.34527954e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.09919568e-02,
#              2.47563899e-02,   3.80797369e-01,   6.76900299e-01])
#     0:04:11
#     --------------------------------------
#     
#     
#     Best parameters
#     array([  2.17198832e+01,   1.84813288e+00,   1.02178777e+00,
#              5.60832212e-01,  -7.34527954e-18,   3.26719778e+01,
#              3.96124587e-01,   2.41453248e+01,   4.09919568e-02,
#              2.47563899e-02,   3.80797369e-01,   6.76900299e-01])
#     Initial condition
#     array([  8.30374493e-01,   1.57991218e-01,   1.12725757e-02,
#              3.57462803e-04,   4.25079167e-06,  -1.73472348e-18,
#             -1.11022302e-15])
#     
#     Time of Execution (hh:mm:ss)
#     0:04:11
# 

# # Plot results and error

# In[8]:

# get curves with best parameters
sol = output_prms[:,-1]
ybl = output_y0[:,-1]


def plot_sim_expt(f, y0, t, prm, ii, gi, gv):

    # Get simulated data
    sim_i, sim_g = evalmodgroup(f, y0, t, prm, gv)
    
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
    plt.plot(t, sim_i - ii)
    
    plt.subplot(1,2,2)
    plt.title('Gating Error')
    plt.plot(t, sim_g - gi)
    
    # Plot histograms of residuals    
    for i in np.arange(np.size(gv)):   
        
        plt.figure(5, figsize=(12,4))
        plt.suptitle('Residual error on ionic current')
        plt.subplot(2,5,i+1)
        plt.xlim([-.05,.05])
        plt.hist(sim_i[:,i]-ii[:,i], bins=20)    
 
        plt.figure(6, figsize=(12,4))
        plt.suptitle('Residual error on gating current')
        plt.subplot(2,5,i+1)
        plt.xlim([-.5,.5])
        plt.hist(sim_g[:,i]-gi[:,i], bins=20)

    
plot_sim_expt(f, ybl, t, sol, ii, gi, gv)


# Out[8]:

# image file:

# image file:

# image file:

# image file:

# # Convergence of solution and initial condition

# In[9]:

# Plot history of output parameters and initial conditions to show convergence

def show_convergence(output_prms, output_y0):
    # Show convergence of parameters

    plt.figure(figsize=(12,4))
    plt.suptitle('Parameters')
    for i in np.arange(len(output_prms)):
        plt.subplot(4,4,i+1)
        plt.plot(output_prms[i,:])
        
    plt.figure(figsize=(12,4))
    #for i in np.arange(len(output_y0)):
    #    plt.subplot(1,7,i+1)
    #    plt.plot(output_y0[i,:])
    plt.title('Initial condition')
    plt.plot(np.transpose(output_y0))

show_convergence(output_prms, output_y0)


# Out[9]:

# image file:

# image file:
