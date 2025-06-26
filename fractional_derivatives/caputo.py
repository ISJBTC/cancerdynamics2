import numpy as np
from scipy.special import gamma
from .base_derivative import BaseFractionalDerivative, FractionalIntegrator


class CaputoDerivative(BaseFractionalDerivative):
    """
    Caputo fractional derivative implementation
    
    The Caputo derivative is defined as:
    D^α f(t) = 1/Γ(n-α) ∫[0,t] f^(n)(τ)/(t-τ)^(α-n+1) dτ
    
    where n = ceil(α)
    """

    def __init__(self, alpha=1.0):
        super().__init__(alpha)
        self.n = int(np.ceil(alpha))  # Order of classical derivative
        self.integrator = FractionalIntegrator(alpha)
        
    def compute_derivative(self, func, t, y, dt):
        """
        Compute Caputo fractional derivative
        
        Args:
            func: Function to differentiate (cancer dynamics)
            t (float): Current time
            y: Current state [T, I, M, S]
            dt (float): Time step
            
        Returns:
            Fractional derivative value
        """
        # Update history
        self.update_history(t, y)
        
        # For α ∈ (0,1), Caputo derivative reduces to:
        # D^α f(t) = 1/Γ(1-α) ∫[0,t] f'(τ)/(t-τ)^α dτ
        
        if len(self.history) < 2:
            # Not enough history, return classical derivative
            return func(y, t)
            
        # Compute classical derivatives at each time point
        derivatives = []
        for i in range(len(self.history)):
            if i < len(self.time_history):
                dy_dt = func(self.history[i], self.time_history[i])
                derivatives.append(dy_dt)
        
        if len(derivatives) < 2:
            return func(y, t)
            
        # Apply Caputo formula using numerical integration
        result = self._caputo_integration(derivatives, dt, t)
        
        return result
    
    def _caputo_integration(self, derivatives, dt, current_time):
        """
        Numerical integration for Caputo derivative
        """
        if len(derivatives) < 2:
            return derivatives[-1] if derivatives else [0, 0, 0, 0]
            
        n_points = len(derivatives)
        result = np.zeros(4)  # For [T, I, M, S]
        
        # Convert derivatives to numpy array
        derivatives = np.array(derivatives)
        
        # Caputo integration using rectangular rule
        for k in range(n_points - 1):
            t_k = k * dt
            t_diff = current_time - t_k
            
            if t_diff > 0:
                # Weight function: (t-τ)^(-α) / Γ(1-α)
                weight = (t_diff**(-self.alpha)) / gamma(1 - self.alpha)
                
                # Ensure weight is finite and reasonable
                if np.isfinite(weight) and weight < 1e10:
                    result += derivatives[k] * weight * dt
                    
        # Apply stability constraints
        result = np.clip(result, -50, 50)
        
        return result
    
    def compute_derivative_simple(self, func, t, y, dt):
        """
        Simplified Caputo derivative for research purposes
        Similar to the original code's fractional implementation
        """
        self.update_history(t, y)
        
        if len(self.history) < 2:
            return func(y, t)
            
        # Simple finite difference approximation with fractional scaling
        current_state = np.array(y)
        previous_state = np.array(self.history[-2]) if len(self.history) >= 2 else current_state
        
        # Classical derivative
        classical_deriv = np.array(func(y, t))
        
        # Fractional correction term (similar to original code)
        if len(self.history) >= 2:
            # Memory effect with fractional scaling
            memory_term = self.alpha * 0.05 * (current_state - previous_state)
            memory_term = np.clip(memory_term, -5, 5)
            
            # Combine classical and fractional parts
            fractional_deriv = classical_deriv + memory_term
        else:
            fractional_deriv = classical_deriv
            
        return np.clip(fractional_deriv, -10, 10)


# Convenience function for easy use
def caputo_derivative(alpha=1.0):
    """Create Caputo derivative instance"""
    return CaputoDerivative(alpha)