import numpy as np
from abc import ABC, abstractmethod
from scipy.special import gamma


class BaseFractionalDerivative(ABC):
    """Base class for all fractional derivative implementations"""

    def __init__(self, alpha=1.0):
        """
        Initialize fractional derivative
        
        Args:
            alpha (float): Fractional order (0 < alpha <= 2)
        """
        if not 0 < alpha <= 2:
            raise ValueError("Alpha must be between 0 and 2")
        self.alpha = alpha
        self.history = []
        self.time_history = []

    @abstractmethod
    def compute_derivative(self, func, t, y, dt):
        """
        Compute fractional derivative at time t
        
        Args:
            func: Function to differentiate
            t (float): Current time
            y: Current state
            dt (float): Time step
            
        Returns:
            Fractional derivative value
        """
        pass

    def reset_history(self):
        """Reset history for new simulation"""
        self.history = []
        self.time_history = []

    def update_history(self, t, y):
        """Update history with new point"""
        self.history.append(y)
        self.time_history.append(t)

    def get_weights(self, n):
        """
        Get Grünwald-Letnikov weights for fractional derivative
        
        Args:
            n (int): Number of points
            
        Returns:
            numpy.array: Weights for GL approximation
        """
        weights = np.zeros(n)
        weights[0] = 1.0
        
        for k in range(1, n):
            weights[k] = weights[k-1] * (self.alpha - k + 1) / k
            
        return weights

    def binomial_coefficient(self, n, k):
        """Compute binomial coefficient for fractional order"""
        if k == 0:
            return 1.0
        elif k > n:
            return 0.0
        else:
            return gamma(n + 1) / (gamma(k + 1) * gamma(n - k + 1))

    def power_law_kernel(self, t):
        """Power law kernel for fractional derivatives"""
        if t <= 0:
            return 0.0
        return t**(self.alpha - 1) / gamma(self.alpha)

    def mittag_leffler_kernel(self, t, beta=1.0):
        """Mittag-Leffler kernel approximation"""
        if t <= 0:
            return 0.0
        # Simple approximation for small t
        if t < 1:
            return t**(self.alpha - 1) / gamma(self.alpha)
        else:
            return np.exp(-t**(self.alpha/2)) * t**(self.alpha - 1)

    def __str__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha})"

    def __repr__(self):
        return self.__str__()


class FractionalIntegrator:
    """Helper class for fractional integration"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def riemann_liouville_integral(self, func_values, dt):
        """Compute Riemann-Liouville fractional integral"""
        n = len(func_values)
        if n == 0:
            return 0.0
            
        integral = 0.0
        for k in range(n):
            t_k = (n - k) * dt
            if t_k > 0:
                integral += func_values[k] * (t_k**(self.alpha - 1))
                
        return integral * dt / gamma(self.alpha)
    
    def caputo_integral(self, func_values, dt):
        """Compute integral part for Caputo derivative"""
        if len(func_values) < 2:
            return 0.0
            
        # Use trapezoidal rule for integration
        integral = 0.0
        for k in range(1, len(func_values)):
            t_k = k * dt
            weight = (t_k**(1 - self.alpha)) / gamma(2 - self.alpha)
            integral += 0.5 * (func_values[k] + func_values[k-1]) * weight * dt
            
        return integral


def create_fractional_derivative(derivative_type, alpha=1.0):
    """
    Factory function to create fractional derivative instances
    
    Args:
        derivative_type (str): Type of derivative ('caputo', 'riemann_liouville', etc.)
        alpha (float): Fractional order
        
    Returns:
        BaseFractionalDerivative: Instance of the requested derivative type
    """
    derivative_map = {
        'caputo': 'CaputoDerivative',
        'riemann_liouville': 'RiemannLiouvilleDerivative', 
        'grunwald_letnikov': 'GrunwaldLetnikovDerivative',
        'hilfer': 'HilferDerivative',
        'gutan': 'GutanDerivative'
    }
    
    if derivative_type not in derivative_map:
        available = ', '.join(derivative_map.keys())
        raise ValueError(f"Unknown derivative type: {derivative_type}. "
                        f"Available types: {available}")
    
    # Import here to avoid circular imports
    module_name = f"fractional_derivatives.{derivative_type}"
    class_name = derivative_map[derivative_type]
    
    try:
        module = __import__(module_name, fromlist=[class_name])
        derivative_class = getattr(module, class_name)
        return derivative_class(alpha)
    except ImportError as e:
        raise ImportError(f"Could not import {derivative_type}: {e}")
 