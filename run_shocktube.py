# 9.3(c) and (d): Riemann shock-tube test and comparison with the
# analytical solution.
#
# Before running this, run ShockTube.py (with t set to the same t_final as
# below) to produce Riemann.data.

import numpy as np
import matplotlib.pyplot as plt
import integrator


n = 200              # number of grid cells
courant = 0.5
Gamma = 1.4
t_final = 0.2

rho_left, P_left = 1.0, 1.0
rho_right, P_right = 0.125, 0.1
x_partition = 0.5

# part c
delta_x = 1.0 / n
x = np.linspace(0.5 * delta_x, 1.0 - 0.5 * delta_x, n)

hydro = integrator.Integrator(x, Gamma, courant)
hydro.initialize(P_left, P_right, rho_left, rho_right, x_partition)
t, n_step = hydro.integrate(t_final)
print("finished at t =", t, "after", n_step, "steps")
numerical_file = "output_{0:}_{1:08}".format(n, n_step)

# part d
num = np.loadtxt(numerical_file)
ana = np.loadtxt("Riemann.data")

fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
labels = [r"$\rho$", r"$e$", r"$v$"]
for k, ax in enumerate(axes):
    ax.plot(ana[:, 0], ana[:, k + 1], "-", label="analytical")
    ax.plot(num[:, 0], num[:, k + 1], "o", label="numerical (n = {0})".format(n))
    ax.set_ylabel(labels[k], size=14)
    ax.grid(True)
axes[0].set_title("Riemann shock tube, t = {0:.2f}".format(t))
axes[0].legend()
axes[-1].set_xlabel("x", size=14)
plt.tight_layout()
plt.savefig("shocktube_t{0:.2f}.png".format(t), dpi=150)
plt.show()
