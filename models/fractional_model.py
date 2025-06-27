import numpy as np
from .base_model import BaseModel


class FractionalModel(BaseModel):
    """Fractional order cancer dynamics model with quantum pressure and memory effects"""

    def __init__(self):
        super().__init__()
        # Import configuration to get fractional parameters
        from config.parameters import get_config
        config = get_config()
        params = config.model_params
        
        # Fractional model specific parameters
        self.fractional_scaling_T = params['fractional_scaling_T']
        self.fractional_scaling_I = params['fractional_scaling_I']
        self.memory_influence = params['memory_influence']
        
        # Initialize history for fractional terms
        self.fractional_history = {'T': [], 'I': []}

    def system_dynamics(self, state, t):
        """
        Fractional system dynamics from equations (5)-(8) with quantum pressure
        state = [T, I, M, S] where:
        T = Tumor cells
        I = Immune cells
        M = Memory cells
        S = Suppressor cells
        """
        # Ensure positive values
        T, I, M, S = self.ensure_positive(state)

        # Update history for fractional terms
        self.fractional_history['T'] = self.fractional_history['T'][-2:] + [T]
        self.fractional_history['I'] = self.fractional_history['I'][-2:] + [I]

        # Calculate fractional memory terms T_frac and I_frac
        T_hist = np.array(self.fractional_history['T'])
        I_hist = np.array(self.fractional_history['I'])

        if len(T_hist) > 1:
            T_frac = self.fractional_scaling_T * (T - T_hist[-1])  # T_frac memory term
            I_frac = self.fractional_scaling_I * (I - I_hist[-1])  # I_frac memory term
        else:
            T_frac = 0
            I_frac = 0

        # Apply clipping to prevent extreme fractional values
        T_frac = np.clip(T_frac, -5, 5)
        I_frac = np.clip(I_frac, -5, 5)

        # Calculate quantum pressure terms from equations (13) and (14)
        Q_tau = self.calculate_quantum_pressure_T(T)
        Q_i = self.calculate_quantum_pressure_I(I)

        # Equation (5): Fractional tumor dynamics with quantum pressure
        # ᶜD_t^α T = r1*T*(1 - T/K1) - a1*T*I + Q_τ + c1*sin(T*I) - h*T + T_frac
        dT = (self.r1 * T * (1 - T/self.K1) - 
              self.a1 * T * I + Q_tau + 
              self.c1 * np.sin(T * I) - self.h * T + T_frac)

        # Equation (6): Fractional immune dynamics with quantum pressure
        # ᶜD_t^α I = r2*I*(1 - I/K2) - a2*T*I/(1+k*T) - d1*I + Q_i + c2*cos(I*M) + I_frac
        dI = (self.r2 * I * (1 - I/self.K2) - 
              (self.a2 * T * I)/(1 + self.k * T) - 
              self.d1 * I + Q_i + 
              self.c2 * np.cos(I * M) + I_frac)

        # Equation (7): Memory dynamics with fractional influence
        # ᶜD_t^α M = r3*M*(1 - M/K3) - a3*M*S + 0.02*T*(1 - M/K3) + 0.15*I_frac
        dM = (self.r3 * M * (1 - M/self.K3) - 
              self.a3 * M * S + 
              0.02 * T * (1 - M/self.K3) + 
              self.memory_influence * I_frac)

        # Equation (8): Suppressor dynamics with fractional influence
        # ᶜD_t^α S = r4*S*(1 - S/K4) + a4*T*S - 0.001*S*(S/K4)² + 0.15*I_frac
        dS = (self.r4 * S * (1 - S/self.K4) + 
              self.a4 * T * S - 
              self.suppressor_quadratic * S * (S/self.K4)**2 + 
              self.memory_influence * I_frac)

        # Apply stability constraints
        derivatives = [dT, dI, dM, dS]
        return self.clip_derivatives(derivatives)

    def __call__(self, state, t):
        """Make the model callable for odeint"""
        return self.system_dynamics(state, t)

    def reset_history(self):
        """Reset fractional history - useful for new simulations"""
        self.fractional_history = {'T': [], 'I': []}

    def get_fractional_info(self):
        """Get information about fractional memory effects"""
        T_hist = self.fractional_history['T']
        I_hist = self.fractional_history['I']
        
        return {
            'model_type': 'Fractional Order with Quantum Effects and Memory',
            'memory_length_T': len(T_hist),
            'memory_length_I': len(I_hist),
            'fractional_scaling': {
                'T_frac': self.fractional_scaling_T,
                'I_frac': self.fractional_scaling_I
            },
            'memory_influence': self.memory_influence,
            'current_history': {
                'T': T_hist[-3:] if len(T_hist) >= 3 else T_hist,
                'I': I_hist[-3:] if len(I_hist) >= 3 else I_hist
            }
        }

    def get_model_info(self):
        """Get comprehensive information about the fractional model"""
        base_info = {
            'model_type': 'Fractional Order with Quantum Effects',
            'equations': ['ᶜD_t^α T with Q_τ + T_frac', 'ᶜD_t^α I with Q_i + I_frac', 
                         'ᶜD_t^α M with I_frac', 'ᶜD_t^α S with I_frac'],
            'quantum_effects': True,
            'memory_effects': True,
            'nonlinear_coupling': True,
            'parameters': {
                'growth_rates': [self.r1, self.r2, self.r3, self.r4],
                'carrying_capacities': [self.K1, self.K2, self.K3, self.K4],
                'quantum_momentum': self.p,
                'quantum_threshold': self.quantum_threshold,
                'fractional_scaling': [self.fractional_scaling_T, self.fractional_scaling_I],
                'memory_influence': self.memory_influence
            }
        }
        
        # Add fractional-specific information
        fractional_info = self.get_fractional_info()
        base_info.update(fractional_info)
        
        return base_info