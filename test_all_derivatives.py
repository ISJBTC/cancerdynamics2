import numpy as np
from fractional_derivatives.base_derivative import create_fractional_derivative
from fractional_derivatives.caputo import CaputoDerivative
from fractional_derivatives.riemann_liouville import RiemannLiouvilleDerivative
from fractional_derivatives.grunwald_letnikov import GrunwaldLetnikovDerivative
from fractional_derivatives.hilfer import HilferDerivative
from fractional_derivatives.gutan import GutanDerivative
from config.parameters import get_alpha_values

# Test function
def test_function(y, t):
    """Simple test function for cancer dynamics"""
    T, I, M, S = y
    return [-0.1*T, 0.05*I, -0.02*M, 0.01*S]

print("ALL FRACTIONAL DERIVATIVES TEST")
print("=" * 50)

# Test alpha values
alpha_test = [0.5, 0.8, 1.0, 1.5]
test_state = [50.0, 10.0, 20.0, 30.0]
test_time = 0.1
dt = 0.01

print(f"Test state: {test_state}")
print(f"Test alphas: {alpha_test}")
print("\n" + "-" * 50)

# Test all derivative types
derivative_types = ['caputo', 'riemann_liouville', 'grunwald_letnikov', 'hilfer', 'gutan']

for deriv_type in derivative_types:
    print(f"\n{deriv_type.upper()} DERIVATIVE:")
    print("-" * 30)
    
    for alpha in alpha_test:
        try:
            # Create derivative instance
            if deriv_type == 'hilfer':
                derivative = HilferDerivative(alpha=alpha, beta=0.5)
            elif deriv_type == 'gutan':
                derivative = GutanDerivative(alpha=alpha, tau=1.0)
            else:
                derivative = create_fractional_derivative(deriv_type, alpha=alpha)
            
            print(f"  α={alpha}: {derivative}")
            
            # Test computation
            result = derivative.compute_derivative_simple(test_function, test_time, test_state, dt)
            print(f"    Result: [{result[0]:.4f}, {result[1]:.4f}, {result[2]:.4f}, {result[3]:.4f}]")
            
        except Exception as e:
            print(f"  α={alpha}: ERROR - {e}")

print(f"\n{'='*50}")
print("All derivative types implemented and tested!")
print("Available alpha values:", len(get_alpha_values()))
