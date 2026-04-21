import numpy as np

class AvellanedaStoikovAgent:
    def __init__(self, gamma=0.1, sigma=1.0, kappa=1.5, T=150, dt=15):
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa
        self.T     = T
        self.dt    = dt

    def compute_quotes(self, mid_price, inventory, t):
        time_remaining = max(self.T - t, 1e-6)
        reservation    = (mid_price - inventory * self.gamma
                          * (self.sigma ** 2) * time_remaining)
        spread = (self.gamma * (self.sigma ** 2) * time_remaining
                  + (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa))
        return reservation - spread / 2.0, reservation + spread / 2.0
