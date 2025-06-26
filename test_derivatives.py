import numpy as np
from fractional_derivatives.base_derivative import create_fractional_derivative
from fractional_derivatives.caputo import CaputoDerivative
from config.parameters import get_alpha_values

# Test function (simple exponential)
def test_function(y, t):
    """Simple test function: dy/dt = -y"""
    return -np.array(y)

print("Fractional Derivatives Test:")
print("-" * 40)

# Test alpha values
alpha_vals = get_alpha_values()
print(f"Testing with {len(alpha_vals)} alpha values: {alpha_vals[:5]}...")

# Test Caputo derivative
print("\n1. Testing Caputo Derivative:")
caputo = CaputoDerivative(alpha=0.5)
print(f"   Created: {caputo}")

# Test with simple state
test_state = [1.0, 1.0, 1.0, 1.0]
test_time = 0.1
dt = 0.01

result = caputo.compute_derivative_simple(test_function, test_time, test_state, dt)
print(f"   Result: {result}")

# Test factory function
print("\n2. Testing Factory Function:")
try:
    caputo2 = create_fractional_derivative('caputo', alpha=0.8)
    print(f"   Factory created: {caputo2}")
except Exception as e:
    print(f"   Factory test will work after implementing other derivatives: {e}")

print("\nDerivatives base test completed!")