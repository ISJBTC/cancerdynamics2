import numpy as np
from scipy.special import gamma
from .base_derivative import BaseFractionalDerivative, FractionalIntegrator


class HilferDerivative(BaseFractionalDerivative):
    """
    Hilfer fractional derivative implementation
    
    The Hilfer derivative is a generalization that interpolates between
    Riemann-Liouville and Caputo derivatives using a parameter β ∈ [0,1]
    
    When β = 0: Riemann-Liouville derivative
    When β = 1: Caputo derivative
    """

    def __init__(self, alpha=1.0, beta=0.5):
        """
        Initialize Hilfer derivative
        
        Args:
            alpha (float): Fractional order (0 < alpha <= 2)
            beta (float): Type parameter (0 <= beta <= 1)
        """
        super().__init__(alpha)
        if not 0 <= beta <= 1:
            raise ValueError("Beta must be between 0 and 1")
        self.beta = beta
        self.n = int(np.ceil(alpha))
        self.integrator = FractionalIntegrator(alpha)
        
    def compute_derivative(self, func, t, y, dt):
        """
        Compute Hilfer fractional derivative
        
        Args:
            func: Function to differentiate (cancer dynamics)
            t (float): Current time
            y: Current state [T, I, M, S]
            dt (float): Time step
            
        Returns:
            Fractional derivative value
        """
        self.update_history(t, y)
        
        if len(self.history) < 2:
            return func(y, t)
            
        # Hilfer derivative combines RL and Caputo approaches
        result = self._hilfer_integration(func, dt, t)
        return result
    
    def _hilfer_integration(self, func, dt, current_time):
        """
        Numerical integration for Hilfer derivative
        """
        if len(self.history) < 2:
            return np.array([0, 0, 0, 0])
            
        # For Hilfer derivative, we use a weighted combination approach
        # that interpolates between RL and Caputo methods
        
        # Riemann-Liouville component
        rl_component = self._compute_rl_component(dt, current_time)
        
        # Caputo component  
        caputo_component = self._compute_caputo_component(func, dt, current_time)
        
        # Hilfer combination: β * Caputo + (1-β) * RL
        result = self.beta * caputo_component + (1 - self.beta) * rl_component
        
        # Apply stability constraints
        result = np.clip(result, -50, 50)
        
        return result
    
    def _compute_rl_component(self, dt, current_time):
        """Compute Riemann-Liouville component"""
        n_points = len(self.history)
        result = np.zeros(4)
        
        history_array = np.array(self.history)
        
        for k in range(n_points - 1):
            t_k = k * dt
            t_diff = current_time - t_k
            
            if t_diff > 0:
                weight = (t_diff**(-self.alpha)) / gamma(1 - self.alpha)
                if np.isfinite(weight) and weight < 1e10:
                    result += history_array[k] * weight * dt
                    
        return result
    
    def _compute_caputo_component(self, func, dt, current_time):
        """Compute Caputo component"""
        derivatives = []
        for i in range(len(self.history)):
            if i < len(self.time_history):
                dy_dt = func(self.history[i], self.time_history[i])
                derivatives.append(dy_dt)
        
        if len(derivatives) < 2:
            return np.array([0, 0, 0, 0])
            
        n_points = len(derivatives)
        result = np.zeros(4)
        derivatives = np.array(derivatives)
        
        for k in range(n_points - 1):
            t_k = k * dt
            t_diff = current_time - t_k
            
            if t_diff > 0:
                weight = (t_diff**(-self.alpha)) / gamma(1 - self.alpha)
                if np.isfinite(weight) and weight < 1e10:
                    result += derivatives[k] * weight * dt
                    
        return result
    
    def compute_derivative_simple(self, func, t, y, dt):
        """
        Simplified Hilfer derivative for research purposes
        """
        self.update_history(t, y)
        
        if len(self.history) < 2:
            return func(y, t)
            
        # Classical derivative
        classical_deriv = np.array(func(y, t))
        
        # Hilfer memory effect
        if len(self.history) >= 2:
            current_state = np.array(y)
            previous_state = np.array(self.history[-2])
            
            # Caputo-style memory
            caputo_memory = self.alpha * 0.05 * (current_state - previous_state)
            
            # RL-style memory (weighted historical states)
            rl_memory = np.zeros(4)
            for i, hist_state in enumerate(self.history[:-1]):
                weight = (i + 1)**(-self.alpha/2) / gamma(1 - self.alpha/2)
                if np.isfinite(weight):
                    rl_memory += weight * np.array(hist_state)
            rl_memory *= 0.02 * self.alpha
            
            # Hilfer combination
            hilfer_memory = self.beta * caputo_memory + (1 - self.beta) * rl_memory
            hilfer_memory = np.clip(hilfer_memory, -5, 5)
            
            # Combine with classical derivative
            fractional_deriv = classical_deriv + hilfer_memory
        else:
            fractional_deriv = classical_deriv
            
        return np.clip(fractional_deriv, -10, 10)
    
    def __str__(self):
        return f"HilferDerivative(alpha={self.alpha}, beta={self.beta})"


# Convenience function
def hilfer_derivative(alpha=1.0, beta=0.5):
    """Create Hilfer derivative instance"""
    return HilferDerivative(alpha, beta)
    