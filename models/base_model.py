import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base class for all cancer dynamics models"""

    def __init__(self):
        # Model parameters exactly as in original code
        # Growth rates (reduced tumor growth)
        self.r1, self.r2, self.r3, self.r4 = 0.4, 0.35, 0.3, 0.2
        # Carrying capacities
        self.K1, self.K2, self.K3 = 80.0, 100.0, 120.0
        # Tumor inhibition by immune cells
        self.a1 = 0.012
        # Immune stimulation by tumor
        self.a2 = 0.02
        # Other interaction constants
        self.a3, self.a4 = 0.004, 0.002
        # Tumor natural death rate
        self.h = 0.01
        # Saturation constant
        self.k = 0.1
        # Immune death rate
        self.d1 = 0.05
        # Nonlinear coupling constants
        self.c1, self.c2 = 0.005, 0.005

    @abstractmethod
    def system_dynamics(self, state, t):
        """Define the system dynamics"""
        pass

    def ensure_positive(self, state):
        """Ensure all state variables are positive"""
        return np.maximum(state, 1e-10)

    def gompertz_growth(self, tumor_cells):
        """Gompertz growth for tumor"""
        if tumor_cells > 1e-6:
            return self.r1 * tumor_cells * np.log(self.K1/tumor_cells)
        else:
            return self.r1 * tumor_cells * np.log(self.K1)

    def immune_killing(self, tumor_cells, immune_cells):
        """Improved immune-mediated tumor killing"""
        return self.a1 * tumor_cells * immune_cells / (1 + 0.01*tumor_cells)