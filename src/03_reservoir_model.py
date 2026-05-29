import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns

# Style settings for plots
sns.set_theme(style="whitegrid")
SEED = 42
np.random.seed(SEED)

class ReservoirModel:
    """
    Implementation of a Reservoir Computing model with Euler integration.
    Default parameters mirror the baseline Gaussian network described in the report.
    """
    def __init__(self, n_neurons=250, spectral_radius=0.8, input_scaling=0.1, input_dim=3, dt=0.01):
        self.N = n_neurons
        self.J = spectral_radius
        self.input_scaling = input_scaling
        self.input_dim = input_dim
        self.dt = dt
        
        # Initialize internal weights W ~ N(0, J^2/N)
        std_dev = self.J / np.sqrt(self.N)
        self.W = np.random.normal(0, std_dev, (self.N, self.N))
        
        # Initialize input weights W_in with standard Gaussian scaled by input_scaling
        self.W_in = np.random.normal(0, 1, (self.N, self.input_dim)) * self.input_scaling
        
        self.W_readout = None

    def harvest_states(self, input_signal, washout=200):
        """
        Simulates the reservoir using Teacher Forcing to collect internal states.
        
        Parameters:
        - input_signal: numpy array of shape (T, input_dim) representing the ground-truth signal.
        - washout: number of initial time steps to discard to remove zero-state transient effects.
        
        Returns:
        - R: State matrix of shape (T - washout, N).
        - valid_signal: The corresponding target signal after washout.
        """
        T = input_signal.shape[0]
        R = np.zeros((T, self.N))
        
        # Initial reservoir state r(0) = 0
        r = np.zeros(self.N)
        
        for t in range(T):
            u = input_signal[t]
            # Continuous-time update via Euler discretization
            # dr/dt = -r(t) + tanh(W * r(t) + W_in * h(t))
            dr = -r + np.tanh(self.W @ r + self.W_in @ u)
            r = r + self.dt * dr
            R[t] = r
            
        # Discard transient states (washout)
        return R[washout:], input_signal[washout:]

    def train_readout(self, R, target_signal, lambda_reg=1e-8):
        """
        Trains the readout weights using Ridge Regression.
        The model predicts the signal at time t+1 using the state at time t.
        """
        # Temporal shift: predict t+1 from state at t
        X = R[:-1]          # States at time t
        Y = target_signal[1:] # Target values at time t+1
        
        # Augment the state matrix with a constant bias term (column of 1s)
        X_biased = np.hstack([X, np.ones((X.shape[0], 1))])
        
        # Solve regularized least-squares (Ridge Regression)
        # alpha corresponds to lambda regularization parameter
        ridge = Ridge(alpha=lambda_reg, fit_intercept=False)
        ridge.fit(X_biased, Y)
        
        self.W_readout = ridge.coef_
        predictions = ridge.predict(X_biased)
        
        # Calculate Root Mean Square Error (RMSE) for training evaluation
        rmse = np.sqrt(np.mean((Y - predictions)**2, axis=0))
        return predictions, Y, rmse

if __name__ == "__main__":
    # Example Usage Pipeline
    print("Initializing baseline Gaussian Reservoir (N=250, J=0.8)...")
    model = ReservoirModel(n_neurons=250, spectral_radius=0.8, input_scaling=0.1)
    
    # In a real execution, you would pass the generated Lorenz data here:
    # dummy_signal = np.random.randn(5000, 3) 
    # R, valid_signal = model.harvest_states(dummy_signal, washout=200)
    # preds, targets, rmse = model.train_readout(R, valid_signal, lambda_reg=1e-8)
    # print(f"Training RMSE for (x, y, z): {rmse}")
    
    print("Reservoir Model module ready to use!")
