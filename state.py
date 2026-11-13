# Professor Baumgarte's state class for the traffic flow problem

import gridfunction


class State:

    def __init__(self, x, name):
        print("Setting up state", name)
        self.name = name
        self.x = x

        self.rho = gridfunction.GridFunction(x, "rho")
        self.tau = gridfunction.GridFunction(x, "tau")
        self.S = gridfunction.GridFunction(x, "S")

    def equals(self, state):
        self.rho.equals(state.rho)
        self.tau.equals(state.tau)
        self.S.equals(state.S)

    def plus(self, factor, state):
        self.rho.plus(factor, state.rho)
        self.tau.plus(factor, state.tau)
        self.S.plus(factor, state.S)

    def plot(self, t):
        self.rho.plot(t)
        self.tau.plot(t)
        self.S.plot(t)
        # self.psi.plot()
