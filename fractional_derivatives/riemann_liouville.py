import numpy as np
from .base_derivative import BaseFractionalDerivative

class RiemannLiouvilleDerivative(BaseFractionalDerivative):
    def __init__(self, alpha=1.5):
        super().__init__(alpha)
        self.max_history = 5
        
    def update_history(self, t, y):
        super().update_history(t, y)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            self.time_history = self.time_history[-self.max_history:]
    
    def compute_derivative_simple(self, func, t, y, dt):
        self.update_history(t, y)
        if len(self.history) < 2:
            return func(y, t)
        classical_deriv = np.array(func(y, t))
        if len(self.history) >= 2:
            current = np.array(y)
            previous = np.array(self.history[-2])
            memory_term = 0.01 * (self.alpha - 1.0) * (current - previous)
            memory_term = np.clip(memory_term, -1, 1)
            result = classical_deriv + memory_term
        else:
            result = classical_deriv
        return np.clip(result, -10, 10)

    def compute_derivative(self, func, t, y, dt):
        return self.compute_derivative_simple(func, t, y, dt)

def riemann_liouville_derivative(alpha=1.0):
    return RiemannLiouvilleDerivative(alpha)
