import numpy as np
from scipy.special import gamma
from .base_derivative import BaseFractionalDerivative


class GrunwaldLetnikovDerivative(BaseFractionalDerivative):
    """
    Grünwald-Letnikov fractional derivative implementation
    
    The Grünwald-Letnikov derivative is defined as:
    D^α f(t) = lim[h→0] h^(-α) Σ[k=0,∞] (-1)^k C(α,k) f(t-kh)
    
    where C(α,k) is the generalized binomial coefficient
    """

    def __init__(self, alpha=1.0):
        super().__init__(alpha)
        self.max_memory = 100  # Limit memory for computational efficiency
        
    def compute_derivative(self, func, t, y, dt):
        """
        Compute Grünwald-Letnikov fractional derivative
        
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
            
        result = self._grunwald_letnikov_sum(dt)
        return result
    
    def _grunwald_letnikov_sum(self, dt):
        """
        Compute the Grünwald-Letnikov sum
        """
        n_points = min(len(self.history), self.max_memory)
        
        if n_points < 2:
            return np.array([0, 0, 0, 0])
            
        result = np.zeros(4)  # For [T, I, M, S]
        
        # Get GL weights
        weights = self.get_weights(n_points)
        
        # Apply GL formula: h^(-α) Σ w_k f(t_n - k*h)
        for k in range(n_points):
            if k < len(self.history):
                state_k = np.array(self.history[-(k+1)])  # Reverse order
                result += weights[k] * state_k
                
        # Scale by h^(-α)
        result *= (dt**(-self.alpha))
        
        # Apply stability constraints
        result = np.clip(result, -50, 50)
        
        return result
    
    def get_weights(self, n):
        """
        Get Grünwald-Letnikov weights
        w_0 = 1, w_k = w_(k-1) * (α - k + 1) / k
        """
        weights = np.zeros(n)
        weights[0] = 1.0
        
        for k in range(1, n):
            weights[k] = weights[k-1] * (self.alpha - k + 1) / k
            
        return weights
    
    def compute_derivative_simple(self, func, t, y, dt):
        """
        Simplified Grünwald-Letnikov derivative for research purposes
        """
        self.update_history(t, y)
        
        if len(self.history) < 2:
            return func(y, t)
            
        # Classical derivative
        classical_deriv = np.array(func(y, t))
        
        # GL-style memory effect
        if len(self.history) >= 2:
            n_terms = min(len(self.history), 10)  # Use last 10 points
            memory_effect = np.zeros(4)
            
            # Simplified GL weights
            for k in range(n_terms):
                if k < len(self.history):
                    # Alternating weight pattern with fractional scaling
                    weight = ((-1)**k) * self.binomial_coefficient(self.alpha, k)
                    if np.isfinite(weight):
                        state_k = np.array(self.history[-(k+1)])
                        memory_effect += weight * state_k
                        
            # Scale memory effect
            memory_term = 0.05 * self.alpha * memory_effect * (dt**(-self.alpha))
            memory_term = np.clip(memory_term, -5, 5)
            
            # Combine with classical derivative
            fractional_deriv = classical_deriv + memory_term
        else:
            fractional_deriv = classical_deriv
            
        return np.clip(fractional_deriv, -10, 10)
    
    def binomial_coefficient(self, alpha, k):
        """Compute generalized binomial coefficient"""
        if k == 0:
            return 1.0
        elif k > 20:  # Avoid overflow for large k
            return 0.0
        else:
            try:
                return gamma(alpha + 1) / (gamma(k + 1) * gamma(alpha - k + 1))
            except (OverflowError, ZeroDivisionError):
                return 0.0


# Convenience function
def grunwald_letnikov_derivative(alpha=1.0):
    """Create Grünwald-Letnikov derivative instance"""
    return GrunwaldLetnikovDerivative(alpha)