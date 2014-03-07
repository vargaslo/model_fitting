
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
array([  2.17198832e+01,   1.84813288e+00,   1.02178777e+00,
         5.60832212e-01,  -7.34527954e-18,   3.26719778e+01,
         3.96124587e-01,   2.41453248e+01,   4.09919568e-02,
         2.47563899e-02,   3.80797369e-01,   6.76900299e-01])
Initial condition
array([  8.30374493e-01,   1.57991218e-01,   1.12725757e-02,
         3.57462803e-04,   4.25079167e-06,  -1.73472348e-18,
        -1.11022302e-15])
        
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
# The objective function is a scoring function whose global minimum corresponds to the best fit of the data.

# In[6]:

def objfun(prm, y0, ii, gi, gv):

    sim_ionic, sim_gating = evalmodgroup(f, y0, t, prm, gv)

    err2_ionic  = np.square(ii - sim_ionic)
    err2_gating = np.square(gi - sim_gating)

    err2_2d_arr = err2_ionic + err2_gating
    err2_1d_arr = np.sum(err2_2d_arr, axis=1)
    err2_sum = np.sum(err2_1d_arr)

    return np.sqrt(err2_sum)

# test if function works
print objfun(Pinit, y0, ii, gi, gv)


# Out[6]:

#     212.425619414
# 

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
        slsqp = opt.minimize(objfun, Pinit, args=(y0, ii, gi, gv), 
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
#     Function Eval 23.7277283651
#     Nit: 2
#     curr y0: [1, 0, 0, 0, 0, 0, 0]
#     curr prm:
#     array([ -2.77515255e-09,   2.75113506e+01,  -8.79730067e-10,
#              2.00000000e+00,   1.00395966e+00,   9.99799728e-01,
#              0.00000000e+00,  -3.90415781e-15,   1.00552559e+00,
#             -2.96725489e-09,   1.02748871e-01,   0.00000000e+00])
#     0:00:01
#     --------------------------------------
#     Iteration: 2/20 Message: Iteration limit exceeded
#     Function Eval 2.00787816809
#     Nit: 101
#     curr y0: array([  5.00000000e-01,   1.90729978e-13,  -1.15731382e-16,
#             -5.32923484e-17,  -8.62072017e-17,  -8.64752714e-13,
#              5.00000000e-01])
#     curr prm:
#     array([  3.34417994e+00,   1.75499526e+01,   4.37123177e-01,
#              7.53406094e-18,   7.66439945e+00,   3.74604105e+00,
#              1.42741912e-18,  -2.09581134e-17,   3.12826084e+00,
#              3.10927224e-04,   3.25416337e-01,   8.84376020e-01])
#     0:02:19
#     --------------------------------------
#     Iteration: 3/20 Message: Iteration limit exceeded
#     Function Eval 1.31761857601
#     Nit: 101
#     curr y0: array([  4.87743299e-01,   8.11635969e-02,   5.06480265e-03,
#              1.40469246e-04,   1.46093617e-06,   2.98907520e-06,
#              4.25883382e-01])
#     curr prm:
#     array([  1.31767160e+01,   7.54314491e+00,   7.54384426e-01,
#             -6.37898462e-16,  -1.24231183e-13,   6.21525437e+01,
#              4.50876669e-03,   9.81555161e+01,   3.67871436e-01,
#             -8.42559883e-21,   7.16182197e-02,   6.56901193e-01])
#     0:04:22
#     --------------------------------------
#     Iteration: 4/20 Message: Optimization terminated successfully.
#     Function Eval 0.908084074204
#     Nit: 51
#     curr y0: array([  6.21245780e-01,   3.14047053e-01,   5.95329304e-02,
#              5.01576616e-03,   1.58470551e-04,   1.63409596e-17,
#              6.83897383e-14])
#     curr prm:
#     array([  2.14678427e+01,   1.50906292e+00,   9.14223129e-01,
#              6.54939508e-01,   1.13489796e-19,   6.21672891e+01,
#              3.88628403e-01,   9.81564401e+01,   8.70368457e-02,
#              6.91915932e-05,   7.14703816e-02,   4.03949039e-01])
#     0:05:38
#     --------------------------------------
#     Iteration: 5/20 Message: Optimization terminated successfully.
#     Function Eval 0.839238182304
#     Nit: 25
#     curr y0: array([  7.91079407e-01,   1.90933653e-01,   1.72812897e-02,
#              6.95164046e-04,   1.04864797e-05,  -3.40439482e-17,
#              4.88498131e-15])
#     curr prm:
#     array([  1.96667988e+01,   2.08175676e+00,   8.86287279e-01,
#              5.62251577e-01,   1.03795280e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.03747831e-01,
#              5.47810489e-05,   6.07957778e-02,   5.99121755e-01])
#     0:06:12
#     --------------------------------------
#     Iteration: 6/20 Message: Optimization terminated successfully.
#     Function Eval 0.839595757763
#     Nit: 7
#     curr y0: array([  7.89165545e-01,   1.92499226e-01,   1.76084500e-02,
#              7.15864377e-04,   1.09136907e-05,  -5.00901404e-17,
#             -1.26010313e-14])
#     curr prm:
#     array([  1.96672911e+01,   2.08079715e+00,   8.87012982e-01,
#              5.61304592e-01,   1.20745977e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.04405639e-01,
#              5.49745975e-05,   6.04142103e-02,   5.98790396e-01])
#     0:06:21
#     --------------------------------------
#     Iteration: 7/20 Message: Optimization terminated successfully.
#     Function Eval 0.839639999112
#     Nit: 1
#     curr y0: array([  7.88937477e-01,   1.92685544e-01,   1.76476527e-02,
#              7.18360170e-04,   1.09655093e-05,   2.75387352e-17,
#              1.02695630e-14])
#     curr prm:
#     array([  1.96672911e+01,   2.08079720e+00,   8.87013290e-01,
#              5.61304744e-01,   1.20745977e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.04404213e-01,
#              5.49738467e-05,   6.04133852e-02,   5.98790137e-01])
#     0:06:23
#     --------------------------------------
#     Iteration: 8/20 Message: Optimization terminated successfully.
#     Function Eval 0.839639940637
#     Nit: 1
#     curr y0: array([  7.88937772e-01,   1.92685303e-01,   1.76476020e-02,
#              7.18356939e-04,   1.09654422e-05,  -6.07153217e-18,
#              7.38298311e-15])
#     curr prm:
#     array([  1.96672910e+01,   2.08079690e+00,   8.87013406e-01,
#              5.61304639e-01,   1.20737009e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.04404285e-01,
#              5.49720704e-05,   6.04134908e-02,   5.98789966e-01])
#     0:06:24
#     --------------------------------------
#     Iteration: 9/20 Message: Optimization terminated successfully.
#     Function Eval 0.839639942367
#     Nit: 1
#     curr y0: array([  7.88937755e-01,   1.92685318e-01,   1.76476050e-02,
#              7.18357134e-04,   1.09654463e-05,   6.17995238e-17,
#              1.77080572e-14])
#     curr prm:
#     array([  1.96672910e+01,   2.08079693e+00,   8.87013458e-01,
#              5.61304658e-01,   1.20736055e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.04404227e-01,
#              5.49718187e-05,   6.04133876e-02,   5.98789937e-01])
#     0:06:26
#     --------------------------------------
#     Iteration: 10/20 Message: Optimization terminated successfully.
#     Function Eval 0.839639919696
#     Nit: 1
#     curr y0: array([  7.88937803e-01,   1.92685279e-01,   1.76475968e-02,
#              7.18356608e-04,   1.09654353e-05,   3.64291930e-17,
#              1.19348975e-14])
#     curr prm:
#     array([  1.96672911e+01,   2.08079724e+00,   8.87016660e-01,
#              5.61306977e-01,   1.20718136e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.04404512e-01,
#              5.49647206e-05,   6.04138070e-02,   5.98789253e-01])
#     0:06:28
#     --------------------------------------
#     Iteration: 11/20 Message: Optimization terminated successfully.
#     Function Eval 0.83963921444
#     Nit: 1
#     curr y0: array([  7.88941319e-01,   1.92682406e-01,   1.76469920e-02,
#              7.18318081e-04,   1.09646349e-05,   2.79724161e-17,
#              5.66213743e-15])
#     curr prm:
#     array([  1.96672911e+01,   2.08079727e+00,   8.87016862e-01,
#              5.61307142e-01,   1.20717470e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.04404315e-01,
#              5.49644568e-05,   6.04135171e-02,   5.98789213e-01])
#     0:06:30
#     --------------------------------------
#     Iteration: 12/20 Message: Optimization terminated successfully.
#     Function Eval 0.839639169456
#     Nit: 1
#     curr y0: array([  7.88941553e-01,   1.92682215e-01,   1.76469517e-02,
#              7.18315515e-04,   1.09645816e-05,  -1.38777878e-17,
#             -1.55431223e-15])
#     curr prm:
#     array([  1.96672911e+01,   2.08079727e+00,   8.87016856e-01,
#              5.61307135e-01,   1.20717517e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.04404316e-01,
#              5.49644384e-05,   6.04135186e-02,   5.98789212e-01])
#     0:06:32
#     --------------------------------------
#     Iteration: 13/20 Message: Optimization terminated successfully.
#     Function Eval 0.839639169705
#     Nit: 1
#     curr y0: array([  7.88941545e-01,   1.92682222e-01,   1.76469532e-02,
#              7.18315609e-04,   1.09645835e-05,  -2.16840434e-17,
#             -9.38138456e-15])
#     curr prm:
#     array([  1.96672913e+01,   2.08079745e+00,   8.87017100e-01,
#              5.61307194e-01,   1.20709508e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.04404396e-01,
#              5.49628524e-05,   6.04136400e-02,   5.98789069e-01])
#     0:06:33
#     --------------------------------------
#     Iteration: 14/20 Message: Optimization terminated successfully.
#     Function Eval 0.839639127718
#     Nit: 1
#     curr y0: array([  7.88941750e-01,   1.92682054e-01,   1.76469178e-02,
#              7.18313353e-04,   1.09645367e-05,   1.92987987e-17,
#              1.66533454e-16])
#     curr prm:
#     array([  1.96672913e+01,   2.08079748e+00,   8.87017148e-01,
#              5.61307211e-01,   1.20708583e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.04404337e-01,
#              5.49626083e-05,   6.04135358e-02,   5.98789041e-01])
#     0:06:35
#     --------------------------------------
#     Iteration: 15/20 Message: Optimization terminated successfully.
#     Function Eval 0.839639114996
#     Nit: 1
#     curr y0: array([  7.88941794e-01,   1.92682019e-01,   1.76469104e-02,
#              7.18312880e-04,   1.09645268e-05,   2.84060969e-17,
#             -1.38777878e-15])
#     curr prm:
#     array([  1.96672916e+01,   2.08079852e+00,   8.87018194e-01,
#              5.61307698e-01,   1.20690428e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.04404493e-01,
#              5.49578147e-05,   6.04137554e-02,   5.98788605e-01])
#     0:06:37
#     --------------------------------------
#     Iteration: 16/20 Message: Optimization terminated successfully.
#     Function Eval 0.83963890324
#     Nit: 1
#     curr y0: array([  7.88942851e-01,   1.92681155e-01,   1.76467284e-02,
#              7.18301291e-04,   1.09642861e-05,   1.79977561e-17,
#              1.11022302e-16])
#     curr prm:
#     array([  1.96672916e+01,   2.08079855e+00,   8.87018238e-01,
#              5.61307714e-01,   1.20689565e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.04404396e-01,
#              5.49575869e-05,   6.04135849e-02,   5.98788577e-01])
#     0:06:38
#     --------------------------------------
#     Iteration: 17/20 Message: Optimization terminated successfully.
#     Function Eval 0.839638892762
#     Nit: 1
#     curr y0: array([  7.88942892e-01,   1.92681122e-01,   1.76467215e-02,
#              7.18300850e-04,   1.09642769e-05,  -5.35595873e-17,
#             -9.71445147e-15])
#     curr prm:
#     array([  1.96672911e+01,   2.08079712e+00,   8.87018436e-01,
#              5.61307111e-01,   1.20689565e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.04404518e-01,
#              5.49506110e-05,   6.04137223e-02,   5.98787946e-01])
#     0:06:40
#     --------------------------------------
#     Iteration: 18/20 Message: Optimization terminated successfully.
#     Function Eval 0.839638962868
#     Nit: 1
#     curr y0: array([  7.88942517e-01,   1.92681428e-01,   1.76467860e-02,
#              7.18304959e-04,   1.09643623e-05,   2.51534904e-17,
#              1.13242749e-14])
#     curr prm:
#     array([  1.96672910e+01,   2.08079712e+00,   8.87018445e-01,
#              5.61307092e-01,   1.20689565e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.04404436e-01,
#              5.49503881e-05,   6.04135790e-02,   5.98787919e-01])
#     0:06:42
#     --------------------------------------
#     Iteration: 19/20 Message: Optimization terminated successfully.
#     Function Eval 0.83963885327
#     Nit: 1
#     curr y0: array([  7.88942509e-01,   1.92681434e-01,   1.76467873e-02,
#              7.18305037e-04,   1.09643639e-05,  -5.22585447e-17,
#             -5.66213743e-15])
#     curr prm:
#     array([  1.96672705e+01,   2.08074461e+00,   8.87019145e-01,
#              5.61283763e-01,   1.20162032e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.04405299e-01,
#              5.47415067e-05,   6.04128790e-02,   5.98768825e-01])
#     0:06:43
#     --------------------------------------
#     Iteration: 20/20 Message: Optimization terminated successfully.
#     Function Eval 0.839642497924
#     Nit: 1
#     curr y0: array([  7.88923810e-01,   1.92696708e-01,   1.76500034e-02,
#              7.18509931e-04,   1.09686208e-05,   2.55871713e-17,
#             -5.21804822e-15])
#     curr prm:
#     array([  1.96672705e+01,   2.08074466e+00,   8.87019188e-01,
#              5.61283785e-01,   1.20162032e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.04405343e-01,
#              5.49045208e-05,   6.04129540e-02,   5.98768812e-01])
#     0:06:45
#     --------------------------------------
#     
#     
#     Best parameters
#     array([  1.96672705e+01,   2.08074466e+00,   8.87019188e-01,
#              5.61283785e-01,   1.20162032e-17,   6.21672963e+01,
#              3.89235756e-01,   9.81564146e+01,   1.04405343e-01,
#              5.49045208e-05,   6.04129540e-02,   5.98768812e-01])
#     Initial condition
#     array([  7.88923855e-01,   1.92696671e-01,   1.76499956e-02,
#              7.18509435e-04,   1.09686105e-05,   4.11996826e-18,
#              2.83106871e-15])
#     
#     Time of Execution (hh:mm:ss)
#     0:06:45
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
 
        plt.figure(6, figsize=(12,4))
        plt.suptitle('Residual error on gating current')
        plt.subplot(2,5,i+1)
        plt.xlim([-.001,.001])
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
