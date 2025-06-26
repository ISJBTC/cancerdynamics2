import numpy as np
from scipy.special import gamma
from .base_derivative import BaseFractionalDerivative


class GutanDerivative(BaseFractionalDerivative):
    """
    Gutan fractional derivative implementation
    
    The Gutan derivative is a relatively new fractional derivative
    that uses a different kernel function for improved convergence
    properties in certain applications.
    """

    def __init__(self, alpha=1.0, tau=1.0):
        """
        Initialize Gutan derivative
        
        Args:
            alpha (float): Fractional order (0 < alpha <= 2)
            tau (float): Scale parameter (tau > 0)
        """
        super().__init__(alpha)
        if tau <= 0:
            raise ValueError("Tau must be positive")
        self.tau = tau
        
    def compute_derivative(self, func, t, y, dt):
        """
        Compute Gutan fractional derivative
        
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
            
        result = self._gutan_integration(func, dt, t)
        return result
    
    def _gutan_integration(self, func, dt, current_time):
        """
        Numerical integration for Gutan derivative
        """
        if len(self.history) < 2:
            return np.array([0, 0, 0, 0])
            
        n_points = len(self.history)
        result = np.zeros(4)  # For [T, I, M, S]
        
        # Gutan kernel with modified exponential decay
        for k in range(n_points - 1):
            t_k = k * dt
            t_diff = current_time - t_k
            
            if t_diff > 0:
                # Gutan kernel: exponential-power law combination
                kernel = self._gutan_kernel(t_diff)
                
                # Apply to function values (can be either state or derivatives)
                if hasattr(func, '__call__'):
                    # Use derivatives
                    if k < len(self.time_history):
                        func_value = np.array(func(self.history[k], self.time_history[k]))
                    else:
                        func_value = np.array(func(self.history[k], t_k))
                else:
                    # Use state values directly
                    func_value = np.array(self.history[k])
                
                if np.isfinite(kernel) and kernel < 1e10:
                    result += func_value * kernel * dt
                    
        # Apply stability constraints
        result = np.clip(result, -50, 50)
        
        return result
    
    def _gutan_kernel(self, t):
        """
        Gutan kernel function
        Combines exponential decay with power law
        """
        if t <= 0:
            return 0.0
            
        # Modified kernel: exp(-t/τ) * t^(α-1) / Γ(α)
        try:
            exponential_part = np.exp(-t / self.tau)
            power_part = t**(self.alpha - 1)
            normalization = 1.0 / gamma(self.alpha)
            
            kernel = exponential_part * power_part * normalization
            
            # Ensure finite value
            if not np.isfinite(kernel):
                return 0.0
                
            return kernel
            
        except (OverflowError, ZeroDivisionError):
            return 0.0
    
    def compute_derivative_simple(self, func, t, y, dt):
        """
        Simplified Gutan derivative for research purposes
        """
        self.update_history(t, y)
        
        if len(self.history) < 2:
            return func(y, t)
            
        # Classical derivative
        classical_deriv = np.array(func(y, t))
        
        # Gutan memory effect
        if len(self.history) >= 2:
            current_state = np.array(y)
            
            # Weighted memory with Gutan-style decay
            memory_effect = np.zeros(4)
            for i, hist_state in enumerate(self.history[:-1]):
                time_diff = (len(self.history) - i - 1) * dt
                
                # Gutan weight with exponential-power decay
                weight = self._gutan_kernel(time_diff)
                
                if np.isfinite(weight) and weight > 1e-15:
                    memory_effect += weight * np.array(hist_state)
                    
            # Scale memory effect
            memory_term = 0.1 * self.alpha * memory_effect
            memory_term = np.clip(memory_term, -5, 5)
            
            # Combine with classical derivative
            fractional_deriv = classical_deriv + memory_term
        else:
            fractional_deriv = classical_deriv
            
        return np.clip(fractional_deriv, -10, 10)
    
    def __str__(self):
        return f"GutanDerivative(alpha={self.alpha}, tau={self.tau})"


# Convenience function
def gutan_derivative(alpha=1.0, tau=1.0):
    """Create Gutan derivative instance"""
    return GutanDerivative(alpha, tau)
    