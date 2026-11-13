import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt


#
# This script provides the analytical solution for a rarefaction-shock wave
# with zero initial speeds.  This analytical solution can be found in many
# references; here we follow Lora-Clavijo et.al., arXiv:13013.3999
#
def riemann(pL, pR, rhoL, rhoR, Gamma, t, x0, xmin, xmax, n):
    Gp1 = Gamma + 1
    Gm1 = Gamma - 1
    #
    # first find pstar (from (48)
    #
    p1 = pL
    rho1 = rhoL
    p6 = pR
    rho6 = rhoR
    #
    v1 = 0.0
    v6 = 0.0
    #
    a1 = np.sqrt(Gamma * p1 / rho1)
    #
    A6 = 2. / rho6 / Gp1
    B6 = p6 * Gm1 / Gp1
    pstar = opt.brentq(eq48, p1, p6,
                       args=(p1, p6, rho1, rho6, Gamma, A6, B6))
    print("found pstar =", pstar)
    p3 = pstar
    p4 = pstar
    v3 = v1 - 2. * a1 / Gm1 * ((p3 / p1) ** (Gm1 / (2. * Gamma)) - 1.)
    v4 = v6 + (p4 - p6) * np.sqrt(A6 / (p4 + B6))  # = v3!!!
    rho3 = rho1 * (p3 / p1) ** (1. / Gamma)
    rho4 = rho6 * (p6 * Gm1 + p4 * Gp1) / (p4 * Gm1 + p6 * Gp1)
    print(rho3, rho4)
    #
    a3 = np.sqrt(Gamma * p3 / rho3)
    a6 = np.sqrt(Gamma * p6 / rho6)
    #
    v_head = - a1
    v_tail = v3 - a3
    v_cont = v3
    v_shock = v6 + a6 * np.sqrt(Gp1 * p4 / (2. * Gamma * p6) + Gm1 / (2. * Gamma))
    print(v_head, v_tail, v_cont, v_shock)
    #
    # create solution vectors
    #
    x = np.linspace(xmin, xmax, n)
    p = np.zeros(n)
    rho = np.zeros(n)
    v = np.zeros(n)
    for i in range(n):
        if (x[i] - x0 < t * v_head):
            # region 1:
            p[i] = p1
            rho[i] = rho1
            v[i] = v1
        elif (x[i] - x0 < t * v_tail):
            # region 2:
            term1 = 2. / Gp1
            term2 = Gm1 / (a1 * Gp1) * (v1 - (x[i] - x0) / t)
            p[i] = p1 * (term1 + term2) ** (2. * Gamma / Gm1)
            rho[i] = rho1 * (term1 + term2) ** (2. / Gm1)
            v[i] = 2 / Gp1 * (a1 + v1 * Gm1 / 2. + (x[i] - x0) / t)
        elif (x[i] - x0 < t * v_cont):
            # region 3:
            p[i] = p3
            rho[i] = rho3
            v[i] = v3
        elif (x[i] - x0 < t * v_shock):
            # region 4:
            p[i] = p4
            rho[i] = rho4
            v[i] = v4
        else:
            # region 6:
            p[i] = p6
            rho[i] = rho6
            v[i] = v6
    return x, p, rho, v

    #
    # find zero of eq. (48)...  CHECK: add v6 - v1 term if needed!!!
    #


def eq48(pstar, p1, p6, rho1, rho6, Gamma, A6, B6):
    a1 = np.sqrt(Gamma * p1 / rho1)
    term1 = (pstar - p6) * np.sqrt(A6 / (pstar + B6))
    exponent = (Gamma - 1.) / (2. * Gamma)
    term2 = 2. * a1 / (Gamma - 1.) * ((pstar / p1) ** exponent - 1.)
    return term1 + term2


pL = 1.0
pR = 0.1
rhoL = 1.0
rhoR = 0.125
Gamma = 1.4
x0 = 0.5
t = 0.2   # must match t_final in run_shocktube.py
xmin = 0.0
xmax = 1.0
n = 1001
x, p, rho, v = riemann(pL, pR, rhoL, rhoR, Gamma, t, x0, xmin, xmax, n)

filename = "Riemann.data"
f = open(filename, 'w')
f.write("# Results for Riemann problem at time t = {0}\n".format(t))
f.write("# {0:12} {1:14} {2:14} {3:14}\n".format("x", "rho", "eps", "v"))
for i in range(n):
    eps = p[i] / (rho[i] * (Gamma - 1.0))
    f.write("{0:.8e} {1:.8e} {2:.8e} {3:.8e}\n".format(x[i], rho[i], eps, v[i]))