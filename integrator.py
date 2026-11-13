# Integrator for the 1D equations of hydrodynamics (Cartesian, flux-conservative
# form) with a Gamma-law EOS.

import numpy as np
import state
import gridfunction


class Integrator:

    def __init__(self, x, gamma, courant):
        self.gamma = gamma
        self.courant = courant
        # for RK4
        self.old = state.State(x, "old")
        self.k1 = state.State(x, "k1")
        self.k2 = state.State(x, "k2")
        self.k3 = state.State(x, "k3")
        self.k4 = state.State(x, "k4")
        self.inter = state.State(x, "intermediate")
        self.delta_x = x[1] - x[0]
        self.delta_t = 0.0          # set every step in integrate()
        self.n = np.size(x)
        self.x = x
        # primitive variables
        self.rho = gridfunction.GridFunction(x, "rho")
        self.e = gridfunction.GridFunction(x, "e")
        self.v = gridfunction.GridFunction(x, "v")
        # reconstructed primitives on the left (L) and right (R) side of each
        # interface; index i refers to interface i - 1/2
        self.rho_L = gridfunction.GridFunction(x, "rho_L")
        self.rho_R = gridfunction.GridFunction(x, "rho_R")
        self.e_L = gridfunction.GridFunction(x, "e_L")
        self.e_R = gridfunction.GridFunction(x, "e_R")
        self.v_R = gridfunction.GridFunction(x, "v_R")
        self.v_L = gridfunction.GridFunction(x, "v_L")
        # fluxes through interface i - 1/2
        self.f_rho = gridfunction.GridFunction(x, "f_rho")
        self.f_tau = gridfunction.GridFunction(x, "f_tau")
        self.f_S = gridfunction.GridFunction(x, "f_S")

# helper functions for calculations

    def P(self, rho, e):
        P = (self.gamma - 1) * e * rho
        return P

    def e_from_P(self, rho, P):
        e = P / ((self.gamma - 1) * rho)
        return e

    def sound_speed(self, rho, e):
        # c_s^2 = Gamma P / rho   (adiabatic sound speed, 9.1b)
        c_s = (self.gamma * self.P(rho, e) / rho)**(1 / 2)
        return c_s

# from notes on October 28th and November 2nd

    def conserved(self, rho, e, v):
        # computing conserved variables from primitive ones
        S = rho * v
        tau = rho * e + 0.5 * rho * v**2
        return rho, tau, S

    def recovery(self, state):
        # fills in the primitive variables rho, v, e from the conserved
        # variables stored in "state" (old during integrate, inter during RK4)
        for i in range(self.n):
            rho = state.rho[i]
            tau = state.tau[i]
            S = state.S[i]
            self.rho[i] = rho
            self.v[i] = S / rho
            self.e[i] = (tau - 0.5 * rho * self.v[i]**2) / rho



    def initialize(self, P_left, P_right, rho_left, rho_right, x_partition):
        # Riemann initial data: two constant states separated
        # by a partition at x_partition
        for i in range(self.n):
            if self.x[i] < x_partition:
                rho, P = rho_left, P_left
            else:
                rho, P = rho_right, P_right
            v = 0.0  # gas is at rest on both sides before the partition is removed
            e = self.e_from_P(rho, P)
            rho, tau, S = self.conserved(rho, e, v)
            self.old.rho[i] = rho
            self.old.tau[i] = tau
            self.old.S[i] = S



    def max_speed(self, state):
        # largest characteristic speed |u+|, |u-|, |u0| over the whole grid
        self.recovery(state)
        u_max = 0.0
        for i in range(self.n):
            c_s = self.sound_speed(self.rho[i], self.e[i])
            u_max = max(u_max, abs(self.v[i] + c_s), abs(self.v[i] - c_s),
                        abs(self.v[i]))
        return u_max

    def integrate(self, t_max, n_output=0):
        t = 0.0
        n_step = 0
        while t < t_max:
            u_max = self.max_speed(self.old)
            self.delta_t = self.courant * self.delta_x / u_max
            if t + self.delta_t > t_max:
                self.delta_t = t_max - t
            self.RK4()
            t += self.delta_t
            n_step += 1
            if n_output > 0 and n_step % n_output == 0:
                self.write_output(self.old, n_step, t)
        self.write_output(self.old, n_step, t)
        return t, n_step


    def minmod(self, a, b):
        # opposite signs (local max/min)
        if a * b <= 0.0:
            return 0.0
        # same sign: take the smaller slope
        elif abs(a) < abs(b):
            return a
        else:
            return b

    def mc(self, a, b):
        # monotonized central
        c = 0.5 * (a + b)
        return self.minmod(c, 2.0 * self.minmod(a, b))

    def reconstruct(self, rho, e, v):
        for i in range(self.n):
            im1 = rho.ind(i - 1)
            ip1 = rho.ind(i + 1)
            if i == 0 or i == self.n - 1:
                slope_rho = slope_e = slope_v = 0.0
            else:
                slope_rho = self.mc(rho[i] - rho[im1], rho[ip1] - rho[i])
                slope_e = self.mc(e[i] - e[im1], e[ip1] - e[i])
                slope_v = self.mc(v[i] - v[im1], v[ip1] - v[i])

            # for rho
            # left edge of cell i = right side of interface i - 1/2
            self.rho_R[i] = rho[i] - 0.5 * slope_rho
            # right edge of cell i = left side of interface i + 1/2
            self.rho_L[ip1] = rho[i] + 0.5 * slope_rho

            # for e
            self.e_R[i] = e[i] - 0.5 * slope_e
            self.e_L[ip1] = e[i] + 0.5 * slope_e

            # for v
            self.v_R[i] = v[i] - 0.5 * slope_v
            self.v_L[ip1] = v[i] + 0.5 * slope_v


    def side(self, rho, e, v):
        # helper function to solve for primitives, fluxes, sound speed, and characteristic speeds
        # on one side of interface
        rho, tau, S = self.conserved(rho, e, v)
        P = self.P(rho, e)
        c_s = self.sound_speed(rho, e)
        # fluxes (see 8.3)
        f_rho = rho * v
        f_S = rho * v**2 + P
        f_tau = (tau + P) * v
        # characteristic speeds (see 8.2)
        u_plus = v + c_s
        u_minus = v - c_s
        u_0 = v
        return rho, tau, S, f_rho, f_tau, f_S, u_plus, u_minus, u_0

    def compute_fluxes(self):
        for i in range(self.n):
            # reconstructed values on each side of interface i - 1/2
            rhoL, eL, vL = self.rho_L[i], self.e_L[i], self.v_L[i]
            rhoR, eR, vR = self.rho_R[i], self.e_R[i], self.v_R[i]

            (q_rhoL, q_tauL, q_SL, f_rhoL, f_tauL, f_SL,
             upL, umL, u0L) = self.side(rhoL, eL, vL)
            (q_rhoR, q_tauR, q_SR, f_rhoR, f_tauR, f_SR,
             upR, umR, u0R) = self.side(rhoR, eR, vR)

            # local maximum characteristic speed at this interface
            u_max = max(abs(upL), abs(umL), abs(u0L),
                        abs(upR), abs(umR), abs(u0R))

            # LLF flux
            self.f_rho[i] = 0.5 * (f_rhoL + f_rhoR - u_max * (q_rhoR - q_rhoL))
            self.f_tau[i] = 0.5 * (f_tauL + f_tauR - u_max * (q_tauR - q_tauL))
            self.f_S[i] = 0.5 * (f_SL + f_SR - u_max * (q_SR - q_SL))

    def compute_derivs(self, current, derivs):
        self.recovery(current)
        self.reconstruct(self.rho, self.e, self.v)
        self.compute_fluxes()
        for i in range(self.n):
            ip1 = self.rho.ind(i + 1)
            # flux in through left face minus flux out through right face
            derivs.rho[i] = -(self.f_rho[ip1] - self.f_rho[i]) / self.delta_x
            derivs.tau[i] = -(self.f_tau[ip1] - self.f_tau[i]) / self.delta_x
            derivs.S[i] = -(self.f_S[ip1] - self.f_S[i]) / self.delta_x
        # 9.3(a): all time derivatives are zero on the boundaries
        for b in (0, self.n - 1):
            derivs.rho[b] = 0.0
            derivs.tau[b] = 0.0
            derivs.S[b] = 0.0

    def RK4(self):
        # step 1:
        self.compute_derivs(self.old, self.k1)
        # step 2:
        self.inter.equals(self.old)
        self.inter.plus(self.delta_t / 2., self.k1)
        self.compute_derivs(self.inter, self.k2)
        # step 3:
        self.inter.equals(self.old)
        self.inter.plus(self.delta_t / 2., self.k2)
        self.compute_derivs(self.inter, self.k3)
        # step 4:
        self.inter.equals(self.old)
        self.inter.plus(self.delta_t, self.k3)
        self.compute_derivs(self.inter, self.k4)
        #
        self.old.plus(self.delta_t / 6., self.k1)
        self.old.plus(self.delta_t / 3., self.k2)
        self.old.plus(self.delta_t / 3., self.k3)
        self.old.plus(self.delta_t / 6., self.k4)



    def write_output(self, state, step_number, time):
        #part b function
        self.recovery(state)   # make sure rho, e, v belong to this state
        x = self.x
        rho = self.rho
        eps = self.e
        v = self.v
        output_filename = "output_{0:}_{1:08}".format(self.n, step_number)
        print("... writing output to file", output_filename)
        out = open(output_filename, 'w')
        #out.write("# with EOS {name}\n".format(name=self.eos.name()))
        out.write("# at time t = {0:.6f}\n".format(time))
        out.write("# {0:10} {1:16} {2:16} {3:16} \n".format("x", "rho", "eps", "v"))
        out.write("#=========================================")
        out.write("=========================================\n")
        for i in range(self.n):
            out.write("{0:.8f}   {1:.8e}   {2:.8e}   {3:.8e}\n".format(
                x[i], rho[i], eps[i], v[i]))
        out.close()
        return output_filename
