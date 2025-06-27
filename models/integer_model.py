import numpy as np
from .base_model import BaseModel


class IntegerModel(BaseModel):
    """Integer order cancer dynamics model with quantum pressure terms"""

    def __init__(self):
        super().__init__()

    def system_dynamics(self, state, t):
        """
        Integer system dynamics from equations (1)-(4) with quantum pressure
        state = [T, I, M, S] where:
        T = Tumor cells
        I = Immune cells  
        M = Memory cells
        S = Suppressor cells
        """
        # Ensure positive values
        T, I, M, S = self.ensure_positive(state)

        # Calculate quantum pressure terms from equations (13) and (14)
        Q_tau = self.calculate_quantum_pressure_T(T)
        Q_i = self.calculate_quantum_pressure_I(I)

        # Equation (1): Tumor dynamics with quantum pressure
        # dT/dt = r1*T*(1 - T/K1) - a1*T*I + Q_τ + c1*sin(T*I) - h*T
        dT = (self.r1 * T * (1 - T/self.K1) - 
              self.a1 * T * I + Q_tau + 
              self.c1 * np.sin(T * I) - self.h * T)

        # Equation (2): Immune dynamics with quantum pressure  
        # dI/dt = r2*I*(1 - I/K2) - a2*T*I/(1+k*T) - d1*I + Q_i + c2*cos(I*M)
        dI = (self.r2 * I * (1 - I/self.K2) - 
              (self.a2 * T * I)/(1 + self.k * T) - 
              self.d1 * I + Q_i + 
              self.c2 * np.cos(I * M))

        # Equation (3): Memory cell dynamics
        # dM/dt = r3*M*(1 - M/K3) - a3*M*S + 0.02*T*(1 - M/K3)
        dM = (self.r3 * M * (1 - M/self.K3) - 
              self.a3 * M * S + 
              0.02 * T * (1 - M/self.K3))

        # Equation (4): Suppressor cell dynamics with quadratic term
        # dS/dt = r4*S*(1 - S/K4) + a4*T*S - 0.001*S*(S/K4)²
        dS = (self.r4 * S * (1 - S/self.K4) + 
              self.a4 * T * S - 
              self.suppressor_quadratic * S * (S/self.K4)**2)

        # Apply stability constraints
        derivatives = [dT, dI, dM, dS]
        return self.clip_derivatives(derivatives)

    def __call__(self, state, t):
        """Make the model callable for odeint"""
        return self.system_dynamics(state, t)

    def get_model_info(self):
        """Get information about the model"""
        return {
            'model_type': 'Integer Order with Quantum Effects',
            'equations': ['dT/dt with Q_τ', 'dI/dt with Q_i', 'dM/dt', 'dS/dt with quadratic'],
            'quantum_effects': True,
            'nonlinear_coupling': True,
            'parameters': {
                'growth_rates': [self.r1, self.r2, self.r3, self.r4],
                'carrying_capacities': [self.K1, self.K2, self.K3, self.K4],
                'quantum_momentum': self.p,
                'quantum_threshold': self.quantum_threshold
            }
        }