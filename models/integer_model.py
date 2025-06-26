from .base_model import BaseModel


class IntegerModel(BaseModel):
    """Integer order cancer dynamics model - exact copy from original code"""

    def __init__(self):
        super().__init__()

    def system_dynamics(self, state, t):
        """
        Integer system dynamics - exactly as in original code
        state = [T, I, M, S] where:
        T = Tumor cells
        I = Immune cells
        M = Memory cells
        S = Stromal cells
        """
        # Ensure positive values
        T, I, M, S = self.ensure_positive(state)

        # Gompertz growth for tumor instead of logistic
        gompertz_growth = self.gompertz_growth(T)

        # Improved immune-mediated tumor killing (non-linear relationship)
        immune_killing = self.immune_killing(T, I)

        # Tumor dynamics (with bounded immune killing)
        dT = gompertz_growth - immune_killing - self.h * T

        # Immune dynamics with tumor stimulation
        dI = (self.r2 * I * (1 - I/self.K2) +
              (self.a2 * T * I)/(1 + self.k * T) -
              self.d1 * I * S/150)

        # Memory cell dynamics
        dM = (self.r3 * M * (1 - M/self.K2) -
              self.a3 * M * S +
              0.02 * T * (1 - M/self.K2))

        # Stromal cell dynamics
        dS = self.r4 * S * (1 - S/self.K3) + self.a4 * T * S

        return [dT, dI, dM, dS]

    def __call__(self, state, t):
        """Make the model callable for odeint"""
        return self.system_dynamics(state, t)
        