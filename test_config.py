from config.parameters import get_config, get_alpha_values, get_initial_conditions

# Test the configuration
config = get_config()

print("Configuration Test:")
print("-" * 30)

# Test alpha values
alpha_vals = get_alpha_values()
print(f"Alpha values: {len(alpha_vals)} values")
print(f"First 5: {alpha_vals[:5]}")
print(f"Last 5: {alpha_vals[-5:]}")

# Test initial conditions
init_conds = get_initial_conditions()
print(f"\nInitial conditions: {len(init_conds)} sets")
for i, cond in enumerate(init_conds):
    print(f"  Set {i+1}: {cond}")

# Test derivative types
print(f"\nFractional derivative types: {config.fractional_derivative_types}")

# Print full summary
print("\n")
config.print_config_summary()

print("\nConfiguration test completed successfully!")
