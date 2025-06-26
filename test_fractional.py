import numpy as np
from scipy.integrate import odeint
from models.fractional_model import FractionalModel

# Test the fractional model
model = FractionalModel()

# Initial conditions from original code
initial_state = [50, 10, 20, 30]  # [T, I, M, S]
t = np.linspace(0, 2, 21)

# Reset history before simulation
model.reset_history()

# Solve the system
solution = odeint(model, initial_state, t)

print("Fractional Model Test:")
print(f"Initial state: {initial_state}")
print(f"Final state: {solution[-1]}")
print(f"History length T: {len(model.fractional_history['T'])}")
print(f"History length I: {len(model.fractional_history['I'])}")
print("Test completed successfully!")
