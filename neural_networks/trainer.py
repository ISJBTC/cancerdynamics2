import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
from sklearn.preprocessing import StandardScaler
from scipy.integrate import odeint

from .dynamics_net import DynamicsDataset
from config.parameters import get_config

warnings.filterwarnings('ignore')


class DynamicsTrainer:
    """
    Trainer class for cancer dynamics neural networks - based on original code
    """
    
    def __init__(self, device=None):
        self.config = get_config()
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Scalers for data normalization
        self.integer_scaler_X = StandardScaler()
        self.integer_scaler_y = StandardScaler()
        self.fractional_scaler_X = StandardScaler()
        self.fractional_scaler_y = StandardScaler()
        
        # Models will be set during training
        self.integer_model = None
        self.fractional_model = None
        
    def generate_training_data(self, model, num_trajectories=None, model_type='integer'):
        """
        Generate training data from model trajectories - exact copy from original
        
        Args:
            model: Cancer dynamics model (integer or fractional)
            num_trajectories: Number of trajectories to generate
            model_type: 'integer' or 'fractional'
            
        Returns:
            X, y: Training features and targets
        """
        if num_trajectories is None:
            num_trajectories = self.config.nn_params['num_trajectories']
            
        t = self.config.get_time_array()
        X, y = [], []
        
        for _ in range(num_trajectories):
            # Random initial conditions similar to original code
            init_state = [
                np.random.uniform(30, 70),    # Tumor
                np.random.uniform(5, 20),     # Immune
                np.random.uniform(10, 40),    # Memory
                np.random.uniform(20, 50)     # Stromal
            ]
            
            if model_type == 'fractional':
                # Reset fractional history
                if hasattr(model, 'reset_history'):
                    model.reset_history()
                elif hasattr(model, 'fractional_history'):
                    model.fractional_history = {'T': [], 'I': []}
                    
            try:
                trajectory = odeint(model, init_state, t)
                
                # Create training pairs: current state -> next state change
                for i in range(len(trajectory)-1):
                    X.append(trajectory[i])
                    y.append(trajectory[i+1] - trajectory[i])
                    
            except Exception as e:
                print(f"Integration error: {e}")
                continue
                
        return np.array(X), np.array(y)
    
    def train_model(self, model, X, y, model_type='integer', epochs=None, 
                   learning_rate=None, verbose=True):
        """
        Train a neural network model - exact copy from original code
        
        Args:
            model: Neural network model
            X: Training features
            y: Training targets
            model_type: 'integer' or 'fractional'
            epochs: Number of training epochs
            learning_rate: Learning rate
            verbose: Print training progress
        """
        # Get scalers
        scaler_X = (self.integer_scaler_X if model_type == 'integer' 
                   else self.fractional_scaler_X)
        scaler_y = (self.integer_scaler_y if model_type == 'integer' 
                   else self.fractional_scaler_y)
        
        # Scale data
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        # Create dataset and dataloader
        dataset = DynamicsDataset(X_scaled, y_scaled)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.nn_params['batch_size'], 
            shuffle=True
        )
        
        # Training parameters
        if epochs is None:
            epochs = self.config.nn_params['epochs']
        if learning_rate is None:
            learning_rate = (self.config.nn_params['learning_rate_integer'] 
                           if model_type == 'integer' 
                           else self.config.nn_params['learning_rate_fractional'])
        
        # Setup training
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        max_grad_norm = self.config.nn_params['max_grad_norm']
        
        # Training loop - exact copy from original
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                
                # Check for NaN loss and skip this batch if found
                if torch.isnan(loss).item():
                    if verbose:
                        print(f"Warning: NaN loss detected at epoch {epoch+1}, skipping batch")
                    continue
                    
                loss.backward()
                
                # Apply gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
            # Print progress
            if verbose and (epoch + 1) % 20 == 0:
                avg_loss = total_loss/batch_count if batch_count > 0 else float('nan')
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
                
        return model
    
    def train_models(self, integer_model, fractional_model, 
                    integer_dynamics, fractional_dynamics, verbose=True):
        """
        Train both integer and fractional models - exact copy from original
        
        Args:
            integer_model: Neural network for integer dynamics
            fractional_model: Neural network for fractional dynamics
            integer_dynamics: Integer dynamics model
            fractional_dynamics: Fractional dynamics model
            verbose: Print training progress
        """
        if verbose:
            print("Training integer model...")
        X_std, y_std = self.generate_training_data(integer_dynamics, model_type='integer')
        if verbose:
            print(f"Generated {len(X_std)} training samples for integer model")
        self.integer_model = self.train_model(
            integer_model, X_std, y_std, 'integer', verbose=verbose
        )
        
        if verbose:
            print("\nTraining fractional model...")
        X_frac, y_frac = self.generate_training_data(fractional_dynamics, model_type='fractional')
        if verbose:
            print(f"Generated {len(X_frac)} training samples for fractional model")
        self.fractional_model = self.train_model(
            fractional_model, X_frac, y_frac, 'fractional', verbose=verbose
        )
        
        return self.integer_model, self.fractional_model
    
    def predict_step(self, model, state, scaler_X, scaler_y):
        """
        Predict single step - exact copy from original
        """
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(scaler_X.transform([state])).to(self.device)
            delta = scaler_y.inverse_transform(model(x).cpu().numpy())
        return delta[0]
    
    def predict_trajectory(self, model, scaler_X, scaler_y, init_state, t):
        """
        Predict full trajectory - exact copy from original
        """
        trajectory = [init_state]
        current_state = init_state
        
        for i in range(1, len(t)):
            delta = self.predict_step(model, current_state, scaler_X, scaler_y)
            next_state = np.maximum(current_state + delta, 1e-10)
            trajectory.append(next_state)
            current_state = next_state
            
        return np.array(trajectory)
    
    def get_scalers(self, model_type='integer'):
        """Get scalers for a specific model type"""
        if model_type == 'integer':
            return self.integer_scaler_X, self.integer_scaler_y
        else:
            return self.fractional_scaler_X, self.fractional_scaler_y
    
    def save_models(self, integer_path='integer_model.pth', 
                   fractional_path='fractional_model.pth'):
        """Save trained models"""
        if self.integer_model is not None:
            torch.save(self.integer_model.state_dict(), integer_path)
            print(f"Integer model saved to {integer_path}")
            
        if self.fractional_model is not None:
            torch.save(self.fractional_model.state_dict(), fractional_path)
            print(f"Fractional model saved to {fractional_path}")
    
    def load_models(self, integer_model, fractional_model,
                   integer_path='integer_model.pth', 
                   fractional_path='fractional_model.pth'):
        """Load pre-trained models"""
        try:
            integer_model.load_state_dict(torch.load(integer_path, map_location=self.device))
            self.integer_model = integer_model.to(self.device)
            print(f"Integer model loaded from {integer_path}")
        except FileNotFoundError:
            print(f"Integer model file {integer_path} not found")
            
        try:
            fractional_model.load_state_dict(torch.load(fractional_path, map_location=self.device))
            self.fractional_model = fractional_model.to(self.device)
            print(f"Fractional model loaded from {fractional_path}")
        except FileNotFoundError:
            print(f"Fractional model file {fractional_path} not found")


class AdvancedTrainer(DynamicsTrainer):
    """
    Advanced trainer with additional features for research
    """
    
    def __init__(self, device=None):
        super().__init__(device)
        self.training_history = {'integer': [], 'fractional': []}
        
    def train_with_validation(self, model, X, y, model_type='integer', 
                            validation_split=0.2, early_stopping=True, 
                            patience=10, verbose=True):
        """
        Train model with validation and early stopping
        """
        # Split data
        n_val = int(len(X) * validation_split)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        
        # Scale data
        scaler_X = (self.integer_scaler_X if model_type == 'integer' 
                   else self.fractional_scaler_X)
        scaler_y = (self.integer_scaler_y if model_type == 'integer' 
                   else self.fractional_scaler_y)
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        X_val_scaled = scaler_X.transform(X_val)
        y_val_scaled = scaler_y.transform(y_val)
        
        # Create datasets
        train_dataset = DynamicsDataset(X_train_scaled, y_train_scaled)
        val_dataset = DynamicsDataset(X_val_scaled, y_val_scaled)
        
        train_loader = DataLoader(train_dataset, 
                                batch_size=self.config.nn_params['batch_size'], 
                                shuffle=True)
        val_loader = DataLoader(val_dataset, 
                              batch_size=self.config.nn_params['batch_size'], 
                              shuffle=False)
        
        # Training setup
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), 
                             lr=self.config.nn_params['learning_rate_integer'])
        criterion = nn.MSELoss()
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        epochs = self.config.nn_params['epochs']
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    pred = model(batch_X)
                    loss = criterion(pred, batch_y)
                    if not torch.isnan(loss):
                        val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Early stopping check
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Store training history
        self.training_history[model_type] = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        return model
        