import numpy as np

class SimulationWorld:
    """
    Defining our simulated "world".
    It "knows" the Ground Truth and provides observations y_t based on C_t.
    """
    def __init__(
        self,
        K: int,
        n: int,
        mu_signal: float,
        Sigma: np.ndarray
    ):
        """
        Initialize the world.
        K: Total streams
        n: Number of anomalous streams
        mu_signal: Mean of anomalous signals
        Sigma: KxK covariance matrix
        """
        self.K = K
        self.n = n
        self.mu_signal = mu_signal
        self.Sigma = Sigma

        # 1. Secretly select n anomalous streams as the ground truth.
        self.S_true = np.random.choice(K, n, replace=False)
        
        # 2. Build the full mean vector mu_true
        self.mu_true = np.zeros(K)
        self.mu_true[self.S_true] = mu_signal
        
        print(f"World created. Ground Truth anomalies (S*) are at indices: {self.S_true}")

    def get_observation(self, C_t) -> float:
        """
        Based on the given observation vector C_t, return a scalar observation y_t.
        y_t = C_t' * X_t
        """
        # Based on the properties of linear combinations of normal distributions:
        # y_t ~ N(mean, var)
        
        # 1. Compute the projected mean
        projected_mean = C_t.dot(self.mu_true)
        
        # 2. Compute the projected variance
        # var = C_t' * Sigma * C_t
        projected_var = C_t.dot(self.Sigma).dot(C_t)
        
        # Ensure variance is positive (numerical stability)
        if projected_var < 1e-9:
            projected_var = 1e-9
            
        projected_std = np.sqrt(projected_var)
        
        # 3. Sample a point from this distribution.
        y_t = np.random.normal(loc=projected_mean, scale=projected_std)
        
        return y_t