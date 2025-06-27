import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base class for all cancer dynamics models with quantum effects"""

    def __init__(self):
        # Import configuration to get parameters
        from config.parameters import get_config
        config = get_config()
        params = config.model_params
        
        # Growth rates (r1, r2, r3, r4)
        self.r1 = params['r1']
        self.r2 = params['r2'] 
        self.r3 = params['r3']
        self.r4 = params['r4']
        
        # Carrying capacities (K1, K2, K3, K4)
        self.K1 = params['K1']
        self.K2 = params['K2']
        self.K3 = params['K3']
        self.K4 = params['K4']
        
        # Interaction coefficients (a1, a2, a3, a4)
        self.a1 = params['a1']
        self.a2 = params['a2']
        self.a3 = params['a3']
        self.a4 = params['a4']
        
        # Treatment/death rate
        self.h = params['h']
        
        # Saturation coefficient
        self.k = params['k']
        
        # Natural death rate
        self.d1 = params['d1']
        
        # Nonlinear coupling constants (c1, c2)
        self.c1 = params['c1']
        self.c2 = params['c2']
        
        # Quantum parameters
        self.p = params['p']
        self.q = params['q']
        self.quantum_threshold = params['quantum_threshold']
        
        # Stability parameters
        self.min_population = params['min_population']
        self.max_derivative = params['max_derivative']
        self.suppressor_quadratic = params['suppressor_quadratic']

    @abstractmethod
    def system_dynamics(self, state, t):
        """Define the system dynamics"""
        pass

    def ensure_positive(self, state):
        """Ensure all state variables are positive"""
        return np.maximum(state, self.min_population)

    def calculate_quantum_pressure_T(self, T):
        """
        Calculate Q_τ (tumor quantum pressure) from equation (13)
        Q_τ = -p²q/(2T) if T > 10^-5 and Q_τ = 0 if T ≤ 10^-5
        """
        if T > self.quantum_threshold:
            return -(self.p**2 * self.q) / (2 * T)
        else:
            return 0.0

    def calculate_quantum_pressure_I(self, I):
        """
        Calculate Q_i (immune quantum pressure) from equation (14)
        Q_i = -p²q/(2I) if I > 10^-5 and Q_i = 0 if I ≤ 10^-5
        """
        if I > self.quantum_threshold:
            return -(self.p**2 * self.q) / (2 * I)
        else:
            return 0.0

    def clip_derivatives(self, derivatives):
        """Apply stability constraints to derivatives"""
        return np.clip(derivatives, -self.max_derivative, self.max_derivative)

    def gompertz_growth(self, tumor_cells):
        """Gompertz growth for tumor (kept for compatibility)"""
        if tumor_cells > 1e-6:
            return self.r1 * tumor_cells * np.log(self.K1/tumor_cells)
        else:
            return self.r1 * tumor_cells * np.log(self.K1)

    def immune_killing(self, tumor_cells, immune_cells):
        """Improved immune-mediated tumor killing (kept for compatibility)"""
        return self.a1 * tumor_cells * immune_cells / (1 + 0.01*tumor_cells)

    def get_quantum_status(self, state):
        """Get quantum status for all cell types"""
        T, I, M, S = state
        return {
            'tumor_quantum_active': T <= self.quantum_threshold,
            'immune_quantum_active': I <= self.quantum_threshold,
            'tumor_pressure': self.calculate_quantum_pressure_T(T),
            'immune_pressure': self.calculate_quantum_pressure_I(I)
        }

    def __str__(self):
        return f"{self.__class__.__name__}(quantum_threshold={self.quantum_threshold:.0e}, p={self.p})"