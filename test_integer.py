import numpy as np
from scipy.integrate import odeint
from models.integer_model import IntegerModel

# Test the integer model
model = IntegerModel()

# Initial conditions from original code
initial_state = [50, 10, 20, 30]  # [T, I, M, S]
t = np.linspace(0, 2, 21)

# Solve the system
solution = odeint(model, initial_state, t)

print("Integer Model Test:")
print(f"Initial state: {initial_state}")
print(f"Final state: {solution[-1]}")
print("Test completed successfully!")
