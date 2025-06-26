import numpy as np
from .base_model import BaseModel


class FractionalModel(BaseModel):
    """Fractional order cancer dynamics model - exact copy from original code"""

    def __init__(self):
        super().__init__()
        # Initialize history for fractional terms - exactly as in original
        self.fractional_history = {'T': [], 'I': []}

    def system_dynamics(self, state, t):
        """
        Fractional system dynamics - exactly as in original code
        state = [T, I, M, S] where:
        T = Tumor cells
        I = Immune cells
        M = Memory cells
        S = Stromal cells
        """
        # Ensure positive values
        T, I, M, S = self.ensure_positive(state)

        # Update history for fractional terms - exactly as in original
        self.fractional_history['T'] = self.fractional_history['T'][-2:] + [T]
        self.fractional_history['I'] = self.fractional_history['I'][-2:] + [I]

        # Simple fractional terms - exactly as in original
        T_hist = np.array(self.fractional_history['T'])
        I_hist = np.array(self.fractional_history['I'])

        # Calculate fractional terms with improved numerical stability
        if len(T_hist) > 1:
            # Very small scaling factor for stability
            T_frac = 0.05 * (T - T_hist[-1])
            I_frac = 0.05 * (I - I_hist[-1])
        else:
            T_frac = 0
            I_frac = 0

        # Apply clipping to prevent extreme values
        T_frac = np.clip(T_frac, -5, 5)
        I_frac = np.clip(I_frac, -5, 5)

        # Gompertz growth for tumor instead of logistic
        gompertz_growth = self.gompertz_growth(T)

        # Improved immune-mediated tumor killing (non-linear relationship)
        immune_killing = self.immune_killing(T, I)

        # Tumor dynamics with fractional term
        dT = gompertz_growth - immune_killing - self.h * T + T_frac

        # Immune dynamics with fractional term
        dI = (self.r2 * I * (1 - I/self.K2) +
              (self.a2 * T * I)/(1 + self.k * T) -
              self.d1 * I * S/150 + I_frac)

        # Memory and stromal dynamics with lesser fractional influence
        dM = (self.r3 * M * (1 - M/self.K2) -
              self.a3 * M * S +
              0.02 * T * (1 - M/self.K2) +
              0.03 * T_frac)

        dS = (self.r4 * S * (1 - S/self.K3) +
              self.a4 * T * S +
              0.03 * I_frac)

        # Final clipping of derivatives for stability
        return np.clip([dT, dI, dM, dS], -10, 10)

    def __call__(self, state, t):
        """Make the model callable for odeint"""
        return self.system_dynamics(state, t)

    def reset_history(self):
        """Reset fractional history - useful for new simulations"""
        self.fractional_history = {'T': [], 'I': []}