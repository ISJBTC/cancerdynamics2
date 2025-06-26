import torch
import numpy as np
from neural_networks.dynamics_net import (
    DynamicsNet, AdaptiveDynamicsNet, FractionalDynamicsNet, 
    EnsembleDynamicsNet, create_network, get_model_info
)
from neural_networks.trainer import DynamicsTrainer
from models.integer_model import IntegerModel
from models.fractional_model import FractionalModel

print("NEURAL NETWORKS TEST")
print("=" * 50)

# Test device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Test different network types
print("\n1. TESTING NETWORK ARCHITECTURES:")
print("-" * 40)

network_configs = [
    {'type': 'standard', 'name': 'Standard DynamicsNet'},
    {'type': 'adaptive', 'name': 'Adaptive Network', 'hidden_sizes': [64, 32], 'dropout_rate': 0.1},
    {'type': 'fractional', 'name': 'Fractional Network', 'memory_size': 10, 'alpha': 0.8},
    {'type': 'ensemble', 'name': 'Ensemble Network', 'num_networks': 3}
]

networks = {}
for config in network_configs:
    net_type = config.pop('type')
    net_name = config.pop('name')
    
    try:
        network = create_network(net_type, **config)
        networks[net_type] = network
        info = get_model_info(network)
        
        print(f"{net_name}:")
        print(f"  Parameters: {info['parameters']:,}")
        print(f"  Type: {info['type']}")
        
        # Test forward pass
        test_input = torch.randn(5, 4)  # Batch of 5, input size 4
        
        if net_type == 'fractional':
            # Test with memory
            memory = torch.randn(5, 10, 4)  # Batch of 5, memory size 10, input size 4
            alpha_tensor = torch.full((5, 1), 0.8)
            output = network(test_input, memory, alpha_tensor)
        elif net_type == 'ensemble':
            output, weights = network(test_input)
            print(f"  Ensemble weights shape: {weights.shape}")
        else:
            output = network(test_input)
            
        print(f"  Output shape: {output.shape}")
        print(f"  ✓ Forward pass successful\n")
        
    except Exception as e:
        print(f"{net_name}: ERROR - {e}\n")

# Test trainer
print("2. TESTING TRAINER:")
print("-" * 40)

try:
    # Create models
    integer_model = IntegerModel()
    fractional_model = FractionalModel()
    
    # Create neural networks
    integer_net = DynamicsNet()
    fractional_net = DynamicsNet()
    
    # Create trainer
    trainer = DynamicsTrainer(device=device)
    
    print(f"Trainer created successfully")
    print(f"Device: {trainer.device}")
    
    # Test data generation (small sample)
    print("\nTesting data generation...")
    X_int, y_int = trainer.generate_training_data(
        integer_model, num_trajectories=5, model_type='integer'
    )
    print(f"Integer data: X shape {X_int.shape}, y shape {y_int.shape}")
    
    X_frac, y_frac = trainer.generate_training_data(
        fractional_model, num_trajectories=5, model_type='fractional'
    )
    print(f"Fractional data: X shape {X_frac.shape}, y shape {y_frac.shape}")
    
    # Test training (very short)
    print("\nTesting training (5 epochs)...")
    trained_int = trainer.train_model(
        integer_net, X_int, y_int, 'integer', epochs=5, verbose=False
    )
    print("✓ Integer model training successful")
    
    trained_frac = trainer.train_model(
        fractional_net, X_frac, y_frac, 'fractional', epochs=5, verbose=False
    )
    print("✓ Fractional model training successful")
    
    # Test prediction
    print("\nTesting prediction...")
    scaler_X, scaler_y = trainer.get_scalers('integer')
    test_state = [50.0, 10.0, 20.0, 30.0]
    
    # Scale the test data first
    test_scaled = scaler_X.transform([test_state])
    delta = trainer.predict_step(trained_int, test_state, scaler_X, scaler_y)
    print(f"Prediction delta: {delta}")
    print("✓ Prediction successful")
    
except Exception as e:
    print(f"Trainer test failed: {e}")

print(f"\n{'='*50}")
print("Neural networks module test completed!")
