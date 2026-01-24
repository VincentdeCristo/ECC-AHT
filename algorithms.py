import numpy as np
import cvxpy as cp
from scipy.stats import norm
from scipy.special import expit
from typing import Optional, List
from tree_utils import build_tree, get_tree_action_vectors

# --- Registry System ---
ALGORITHM_REGISTRY = {}

def register_algorithm(name: str):
    """Decorator to register a new algorithm class."""
    def decorator(cls):
        ALGORITHM_REGISTRY[name] = cls
        return cls
    return decorator

# --- Base Class ---

class BaseAlgorithm:
    """
    Base class for model-driven active hypothesis testing algorithms.
    Supports both centralized (mu_0=0) and non-centralized (mu_0!=0) scenarios.
    """
    def __init__(
        self, 
        K: int, 
        n: int, 
        Sigma: np.ndarray, 
        B: float, 
        mu_signal: Optional[np.ndarray] = None, 
        mu_0: Optional[np.ndarray] = None, 
        delta_signal: Optional[np.ndarray] = None
    ):
        self.K = K  # Number of streams/arms
        self.n = n  # Number of anomalies to identify
        self.Sigma = Sigma
        self.B = B  # L1-norm budget
        
        # Phase compatibility: mu_0 is the mean under H0 (null hypothesis)
        self.mu_0 = mu_0 if mu_0 is not None else np.zeros(K)
        
        # signal_strength (H1 - H0)
        self.delta_signal = delta_signal if delta_signal is not None else mu_signal
        
        if self.delta_signal is None:
            raise ValueError("Must provide mu_signal or delta_signal for the likelihood update.")

        # Belief: Marginal probabilities based on Gaussian model
        initial_prob = n / K
        self.p_t = np.full(K, initial_prob)
        initial_odds = initial_prob / (1 - initial_prob)
        self.log_odds = np.full(K, np.log(initial_odds))
        
        self.t = 0 # internal timestep
        self.T_s = np.zeros(K) # counter for base arm pulls (if applicable)

    def select_action(self) -> np.ndarray:
        """Select a sensing vector C_t. To be implemented by subclasses."""
        raise NotImplementedError

    def update(self, C_t: np.ndarray, y_t: float):
        """
        Update log-odds and beliefs using Gaussian likelihood ratio.
        y_t is the raw observation: C_t' * X_t
        """
        self.t += 1
        
        # Calculate variance under projection C_t
        projected_var = C_t.dot(self.Sigma).dot(C_t)
        projected_std = np.sqrt(max(projected_var, 1e-9))

        # Under H0: y_t ~ N(C_t' * mu_0, projected_var)
        mean_0 = C_t.dot(self.mu_0)
        
        # Under H1(k): y_t ~ N(C_t' * mu_0 + C_t[k] * delta_signal[k], projected_var)
        mean_1_vec = mean_0 + C_t * self.delta_signal
        
        # Log-Likelihood calculation
        log_L_0 = norm.logpdf(y_t, loc=mean_0, scale=projected_std)
        log_L_1 = norm.logpdf(y_t, loc=mean_1_vec, scale=projected_std)
        
        # Log-odds update
        self.log_odds += (log_L_1 - log_L_0)
        self.p_t = expit(self.log_odds)

        # Track base arm pulls (C_t = unit vector e_k)
        if np.sum(C_t > 0) == 1 and np.isclose(np.sum(C_t), 1.0):
             k = np.argmax(C_t)
             self.T_s[k] += 1

    def get_S_hat(self) -> np.ndarray:
        """Return the indices of the n highest probability streams."""
        return np.argsort(self.p_t)[-self.n:]
    
    def is_terminated(self) -> bool:
        """
        Theoretical GLR-style stopping rule.
        Practical implementations may use approximate or surrogate criteria.
        """
        # Placeholder: full GLR over hypothesis sets is computationally expensive
        return False

# --- Core Algorithm ---

@register_algorithm("ECC_AHT")
class ECC_AHT(BaseAlgorithm):
    """
    ECC-AHT: Efficient Champion-Challenger Active Hypothesis Testing.
    The proposed algorithm maximizing the KL-divergence via QP optimization.
    """
    def __init__(self, K, n, Sigma, B, mu_signal=None, mu_0=None, delta_signal=None):
        super().__init__(K=K, n=n, Sigma=Sigma, B=B, mu_signal=mu_signal, mu_0=mu_0, delta_signal=delta_signal)
        # Pre-compile the CVXPY problem for speed
        self.C_var = cp.Variable(K)
        self.delta_param = cp.Parameter(K)
        self.prob = None 

    def _build_problem(self):
        """Initialize the Quadratic Programming problem."""
        Sigma_reg = cp.psd_wrap(self.Sigma + 1e-6 * np.eye(self.K))
        objective = cp.Minimize(cp.quad_form(self.C_var, Sigma_reg))
        constraints = [
            self.C_var @ self.delta_param == 1,
            cp.norm1(self.C_var) <= self.B
        ]
        self.prob = cp.Problem(objective, constraints)

    def select_action(self) -> np.ndarray:
        if self.prob is None:
            self._build_problem()
            
        # Identify Champion (i_star) and Challenger (j_star)
        sorted_indices = np.argsort(self.p_t)
        S_hat = sorted_indices[-self.n:]
        S_hat_complement = sorted_indices[:-self.n]
        
        i_star = S_hat[np.argmin(self.p_t[S_hat])]
        j_star = S_hat_complement[np.argmax(self.p_t[S_hat_complement])]
        
        # Build difference vector
        delta_t = np.zeros(self.K)
        delta_t[i_star] = self.delta_signal
        delta_t[j_star] = -self.delta_signal
        
        try:
            self.delta_param.value = delta_t
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            
            if self.C_var.value is None:
                return self.fallback_action(delta_t)
                
            C_t = self.C_var.value
            c_norm = np.linalg.norm(C_t, 1)
            return (C_t / c_norm * self.B) if c_norm > 1e-9 else self.fallback_action(delta_t)

        except cp.error.SolverError:
            return self.fallback_action(delta_t)

    def fallback_action(self, delta_t: np.ndarray) -> np.ndarray:
        """Fallback to simple difference vector if solver fails."""
        return (delta_t / (np.linalg.norm(delta_t, 1) + 1e-9)) * self.B

# --- Ablation & Variants ---

@register_algorithm("ECC_AHT_Diagonal")
class ECC_AHT_Diagonal(ECC_AHT):
    """Ablation: Force Sigma to be diagonal."""
    def __init__(self, K, n, Sigma, B, mu_signal=None, mu_0=None, delta_signal=None):
        Sigma_diag_only = np.diag(np.diag(Sigma))
        super().__init__(K, n, Sigma_diag_only, B, mu_signal=mu_signal, mu_0=mu_0, delta_signal=delta_signal)

@register_algorithm("RandomSparseProjection")
class RandomSparseProjection(BaseAlgorithm):
    """Baseline: Random Sparse Projection (RSP)."""
    def __init__(self, K, n, Sigma, B, mu_signal=None, mu_0=None, delta_signal=None):
        super().__init__(K, n, Sigma, B, mu_signal=mu_signal, mu_0=mu_0, delta_signal=delta_signal)
        self.num_active = int(np.ceil(self.B))
        
    def select_action(self) -> np.ndarray:
        indices = np.random.choice(self.K, self.num_active, replace=False)
        C_t = np.zeros(self.K)
        C_t[indices] = np.random.randn(self.num_active) 
        
        C_t_norm = np.linalg.norm(C_t, 1)
        if C_t_norm > 1e-9:
            C_t = (C_t / C_t_norm) * self.B
        return C_t

@register_algorithm("RoundRobin")
class RoundRobin(BaseAlgorithm):
    """Baseline: Classical Round-Robin (sequential polling)."""
    def __init__(self, K, n, Sigma, B, mu_signal=None, mu_0=None, delta_signal=None):
        super().__init__(K=K, n=n, Sigma=Sigma, B=B, mu_signal=mu_signal, mu_0=mu_0, delta_signal=delta_signal)
        self.current_stream = 0

    def select_action(self) -> np.ndarray:
        C_t = np.zeros(self.K)
        C_t[self.current_stream] = 1.0
        
        self.current_stream = (self.current_stream + 1) % self.K
        
        return C_t

@register_algorithm("ECC_AHT_SimpleDiff")
class ECC_AHT_SimpleDiff(BaseAlgorithm):
    """Ablation: Uses the difference vector directly without QP optimization."""

    def __init__(self, K, n, Sigma, B, mu_signal=None, mu_0=None, delta_signal=None):
        super().__init__(K=K, n=n, Sigma=Sigma, B=B, mu_signal=mu_signal, mu_0=mu_0, delta_signal=delta_signal)

    def select_action(self) -> np.ndarray:
        S_hat_indices = np.argsort(self.p_t)
        S_hat = S_hat_indices[-self.n:]
        
        i_star = S_hat[np.argmin(self.p_t[S_hat])]
        S_hat_complement = S_hat_indices[:-self.n]
        j_star = S_hat_complement[np.argmax(self.p_t[S_hat_complement])]
        
        delta_t = np.zeros(self.K)
        delta_t[i_star] = self.delta_signal
        delta_t[j_star] = -self.delta_signal
        
        norm_delta = np.linalg.norm(delta_t, 1)
        if norm_delta < 1e-9:
            C_t = np.random.randn(self.K)
            C_t = (C_t / (np.linalg.norm(C_t, 1) + 1e-9)) * self.B
        else:
            C_t = (delta_t / norm_delta) * self.B
            
        return C_t

@register_algorithm("ECC_AHT_CostFree")
class ECC_AHT_CostFree(BaseAlgorithm):
    """
    Benchmark: Unconstrained optimal sensing (No sparsity/budget limit).
    It always calculates and uses the theoretically optimal dense vector C_t ~ Sigma_inv * delta_t
    """
    def __init__(self, K, n, Sigma, B, mu_signal=None, mu_0=None, delta_signal=None):
        super().__init__(K=K, n=n, Sigma=Sigma, B=B, mu_signal=mu_signal, mu_0=mu_0, delta_signal=delta_signal)
        # Pre-calculate the inverse of Sigma, because we will be using it repeatedly.
        try:
            self.Sigma_inv = np.linalg.inv(self.Sigma)
            print("Cost-Free: Sigma inverse computed.")
        except np.linalg.LinAlgError:
            print("Cost-Free: Sigma is singular. Using pseudo-inverse.")
            self.Sigma_inv = np.linalg.pinv(self.Sigma)

    def select_action(self) -> np.ndarray:
        S_hat_indices = np.argsort(self.p_t)
        S_hat = S_hat_indices[-self.n:]
        
        i_star = S_hat[np.argmin(self.p_t[S_hat])]
        S_hat_complement = S_hat_indices[:-self.n]
        j_star = S_hat_complement[np.argmax(self.p_t[S_hat_complement])]
        
        delta_t = np.zeros(self.K)
        delta_t[i_star] = self.delta_signal
        delta_t[j_star] = -self.delta_signal
        
        # [Benchmark] Calculate the unconstrained optimal C_t
        # C_t_optimal ~ Sigma_inv * delta_t
        # (Normalization is not needed because KL divergence is independent of the scale of C_t)
        C_t = self.Sigma_inv.dot(delta_t)
        
        return C_t

@register_algorithm("ECC_AHT_Restricted")
class ECC_AHT_Restricted(BaseAlgorithm):
    """
    Comparative experimental algorithms (key comparisons):
    Using ECC-AHT's "brain" (KL divergence maximization),
    However, it was forced to use HDS's "discrete action space" (tree node vectors).
    """
    def __init__(self, K, n, Sigma, B, mu_signal=None, mu_0=None, delta_signal=None):
        super().__init__(K=K, n=n, Sigma=Sigma, B=B, mu_signal=mu_signal, mu_0=mu_0, delta_signal=delta_signal)

        # 1. Constructing the action space of HDS
        if (self.K & (self.K - 1) != 0) or self.K == 0:
            raise ValueError(f"ECC-AHT-Restricted requires K to be a power of 2, but K={self.K}")

        nodes_dict, _ = build_tree(self.K)
        # self.tree_actions is a dictionary of the form {node_id: C_vector}
        self.tree_actions = get_tree_action_vectors(nodes_dict, self.K)
        print(f"ECC-AHT-Restricted: The action space is restricted to {len(self.tree_actions)} tree vectors.")

    def select_action(self) -> np.ndarray:
        """
        Select action: Do not use the QP solver.
        Instead, brute-force search HDS's action space to find the action that maximizes KL divergence.
        """
        S_hat_indices = np.argsort(self.p_t)
        S_hat = S_hat_indices[-self.n:]
        
        i_star = S_hat[np.argmin(self.p_t[S_hat])]
        S_hat_complement = S_hat_indices[:-self.n]
        j_star = S_hat_complement[np.argmax(self.p_t[S_hat_complement])]
        
        delta_t = np.zeros(self.K)
        delta_t[i_star] = self.delta_signal
        delta_t[j_star] = -self.delta_signal
        
        # [Ablation Point] Action space of brute-force search HDS
        best_C_v = None
        max_kl_metric = -1.0
        
        for C_v in self.tree_actions.values():
            # Calculating the generalized Rayleigh quotient (the core of KL divergence)
            numerator = (C_v.dot(delta_t))**2
            denominator = C_v.dot(self.Sigma).dot(C_v)
            
            if denominator < 1e-9:
                kl_metric = 0.0
            else:
                kl_metric = numerator / denominator
                
            if kl_metric > max_kl_metric:
                max_kl_metric = kl_metric
                best_C_v = C_v
        
        if best_C_v is None:
            # If no match is found (e.g., all kl=0), return a random action.
            return self.tree_actions[np.random.choice(list(self.tree_actions.keys()))]
            
        return best_C_v 

@register_algorithm("HDS_Gafni")
class HDS_Gafni:
    """
    We strictly reproduced the HDS algorithm from Gafni et al. (2023).
    This is a stateful algorithm, completely different from our previous BaseAlgorithm.
    """
    def __init__(self, K, n, mu_signal, Sigma, B=None, K_l=5, Theta1_hypotheses=None):
        if (K & (K - 1) != 0) or K == 0:
            raise ValueError(f"HDS can only be used when K is a power of 2, but K={K}")
        if Sigma is not None and not np.allclose(Sigma, np.eye(K)):
            print("Warning: The reproduction of HDS (Gafni '23) assumes Sigma=I.")
            
        self.K = K
        self.n = n # Algorithm 2 from the HDS paper
        self.mu_signal_true = mu_signal # Actual signal strength (for the environment)
        self.Sigma = np.eye(K) # Assume Sigma = I
        
        # HDS Specific Parameters
        self.K_l = K_l  # Fixed number of samples for internal node testing
        self.L = int(np.log2(K))
        self.c = 1e-5 # Bayesian risk cost (for leaf node threshold)
        self.leaf_threshold = np.log(self.L / self.c)
        
        # Composite hypothesis space
        self.Theta0 = 0.0 # Null hypothesis mean [cite: 720]
        self.Theta1 = Theta1_hypotheses if Theta1_hypotheses else {mu_signal / 2, mu_signal, mu_signal * 1.5}
        
        # State machine
        self.nodes, self.root = build_tree(K)
        self.current_node = self.root
        self.state = "INTERNAL" # 'INTERNAL', 'INTERNAL_SAMPLING', 'LEAF_TEST', 'TERMINATED'
        
        # Action/Sample buffers
        self.action_buffer = []
        self.samples_collected = {'L': [], 'R': []}
        self.leaf_samples = []
        self.leaf_n = 0
        
        # result
        self.S_hat = [] # Anomalies found
        self.n_found = 0

    def get_S_hat(self) -> List[int]:
        return self.S_hat

    def is_terminated(self) -> bool:
        return self.state == "TERMINATED"

    def _get_node_params(self, node) -> float:
        """(HDS assumption) Calculate the variance of the observations. Variance = Number of aggregated leaf nodes."""
        num_leaves = len(node.leaf_indices)
        return np.sqrt(num_leaves) # std_dev

    def _log_pdf(self, y, node, theta) -> float:
        """Calculate the log-pdf of y under N(theta, num_leaves)."""
        std_dev = self._get_node_params(node)
        return norm.logpdf(y, loc=theta, scale=std_dev)

    def _mle(self, samples, node, theta_set) -> float:
        """Finding the Maximum Likelihood Estimator (MLE) on a finite set."""
        if not samples:
            return list(theta_set)[0] 
            
        best_theta = None
        max_log_like = -np.inf
        
        for theta in theta_set:
            log_like = np.sum([self._log_pdf(y, node, theta) for y in samples])
            if log_like > max_log_like:
                max_log_like = log_like
                best_theta = theta
        return best_theta

    def _compute_GLLR(self, samples, node) -> float:
        """Calculate the GLLR statistic for internal nodes."""
        log_L_0 = np.sum([self._log_pdf(y, node, self.Theta0) for y in samples])
        
        # Find the MLE \hat{\theta}_1
        theta_hat_1 = self._mle(samples, node, self.Theta1)
        log_L_1 = np.sum([self._log_pdf(y, node, theta_hat_1) for y in samples])
        
        return log_L_1 - log_L_0

    def _compute_ALLR_step(self, y, node) -> float:
        """Calculate the single-step ALLR for leaf nodes."""
        # \hat{\theta}_1(i-1)
        theta_hat_delayed = self._mle(self.leaf_samples, node, self.Theta1)
        
        log_L_0 = self._log_pdf(y, node, self.Theta0)
        log_L_1 = self._log_pdf(y, node, theta_hat_delayed)
        
        self.leaf_samples.append(y) # Update History
        return log_L_1 - log_L_0

    def _get_action_vec_for_node(self, node) -> Optional[np.ndarray]:
        """Create a C_t vector for aggregated observations of HDS."""
        C_t = np.zeros(self.K)
        # Check if this node has already been pruned (because its child nodes have been found)
        active_leaves = node.leaf_indices - set(self.S_hat)
        if not active_leaves:
            return None # This node doesn't need to be tested.
        C_t[list(active_leaves)] = 1.0
        return C_t

    def select_action(self) -> Optional[np.ndarray]:
        """HDS state machine: determines what to observe next."""
        if self.is_terminated():
            return None # It's already over.

        # 1. Check the action buffer.
        if self.action_buffer:
            return self.action_buffer.pop(0)

        # 2. State machine logic
        # Status: At an internal node, the decision is made whether to descend further or backtrack.
        if self.state == "INTERNAL":
            # If it's a leaf node, switch to leaf node testing.
            if self.current_node.level == 0:
                self.state = "LEAF_TEST"
                self.leaf_samples = [] # Reset
                self.leaf_n = 0
                self.allr_sum = 0.0
                return self.select_action() # Recall
            
            # Otherwise, start internal node testing.
            self.state = "INTERNAL_SAMPLING"
            self.samples_collected = {'L': [], 'R': []}
            
            # Create sampling actions for the left and right child nodes.
            child_L, child_R = self.current_node.children
            
            vec_L = self._get_action_vec_for_node(child_L)
            vec_R = self._get_action_vec_for_node(child_R)

            #  "Measure K_l-1 samples from *each* child node"
            if vec_L is not None:
                self.action_buffer.extend([vec_L] * self.K_l)
            if vec_R is not None:
                self.action_buffer.extend([vec_R] * self.K_l)
                
            if not self.action_buffer:
                # Both child nodes have been pruned.
                self.state = "INTERNAL"
                self.current_node = self.current_node.parent if self.current_node.parent else self.root
                return self.select_action()

            # Save the observed target so that during the update process, we know which entity y belongs to.
            self.internal_targets = {tuple(vec_L): 'L', tuple(vec_R): 'R'} if (vec_L is not None and vec_R is not None) else \
                                    {tuple(vec_L): 'L'} if vec_L is not None else \
                                    {tuple(vec_R): 'R'}
            
            return self.action_buffer.pop(0)

        # Status: At the leaf node, sequential testing.
        if self.state == "LEAF_TEST":
            # "Draw y(n)"
            self.leaf_n += 1
            return self.select_action_for_node(self.current_node)

    def select_action_for_node(self, node) -> Optional[np.ndarray]:
        """Helper function: Generates an action vector for a specified node (for HDS)"""
        C_t = np.zeros(self.K)
        active_leaves = node.leaf_indices - set(self.S_hat)
        if not active_leaves:
            return None # The node has been pruned.
        C_t[list(active_leaves)] = 1.0
        return C_t

    def update(self, C_t, y_t):
        """HDS state machine: updates the state based on observation y_t"""
        if self.is_terminated():
            return

        # Status: Collecting K_l samples from internal nodes.
        if self.state == "INTERNAL_SAMPLING":
            # Identify whether y_t comes from the left child node or the right child node.
            target_key = tuple(C_t)
            if target_key in self.internal_targets:
                child_label = self.internal_targets[target_key]
                self.samples_collected[child_label].append(y_t)
            
            if not self.action_buffer: # Sampling completed.
                self.state = "INTERNAL_DECISION"
                self._process_internal_decision()
            return

        # Status: Leaf node testing in progress
        if self.state == "LEAF_TEST":
            # Calculate ALLR
            step_allr = self._compute_ALLR_step(y_t, self.current_node)
            self.allr_sum += step_allr
            
            # Check the termination conditions.
            if self.allr_sum > self.leaf_threshold:
                leaf_id = list(self.current_node.leaf_indices)[0]
                self.S_hat.append(leaf_id) # Declared as abnormal.
                self.n_found += 1
                
                if self.n_found == self.n:
                    self.state = "TERMINATED" #
                else:
                    # "Remove detected anomalous leaf node from tree"
                    # We achieve this by going back to the root and restarting the search.
                    self.current_node = self.root
                    self.state = "INTERNAL"
            
            # Check the backtracking conditions.
            elif self.allr_sum < 0: # The original paper is < 0
                self.current_node = self.current_node.parent
                self.state = "INTERNAL"
            
            # Otherwise, continue sampling at the same leaf node.
            else:
                self.state = "LEAF_TEST" # Maintain the status quo.
            return

    def _process_internal_decision(self):
        """After collecting K_l*2 samples, the decision is made about where to go next."""
        # "Compute GLLR for each child"
        child_L, child_R = self.current_node.children
        
        GLLR_L = -np.inf
        if self.samples_collected['L']:
            GLLR_L = self._compute_GLLR(self.samples_collected['L'], child_L)
            
        GLLR_R = -np.inf
        if self.samples_collected['R']:
            GLLR_R = self._compute_GLLR(self.samples_collected['R'], child_R)

        # "if Both GLLRs are negative then..."
        if GLLR_L <= 0 and GLLR_R <= 0: # The paper uses 'negative', we use <= 0.
            self.current_node = self.current_node.parent if self.current_node.parent else self.root
        
        # "Invoke... on child with larger GLLR"
        elif GLLR_L > GLLR_R:
            self.current_node = child_L
        else:
            self.current_node = child_R
            
        self.state = "INTERNAL" # Reset the state; the next `select_action` call will make a new decision.

@register_algorithm("BaseArm_CombGapE")
class BaseArm_CombGapE(BaseAlgorithm):
    """
    hybrid baselines (action space restriction):
    Using the "brain" of ECC-AHT (deterministic i*, j* identification),
    but the actions are limited to C_t = e_k (base arm pulling model).
    It uses the CombGapE rule to select which arm to pull.
    """
    def __init__(self, K, n, Sigma, B, mu_signal=None, mu_0=None, delta_signal=None):
        super().__init__(K=K, n=n, Sigma=Sigma, B=B, mu_signal=mu_signal, mu_0=mu_0, delta_signal=delta_signal)
        # T_s is a counter used internally by this algorithm for decision-making.
        # It only calculates pulls of the type C_t = e_k.
        self.T_s = np.zeros(self.K) 

    def select_action(self) -> np.ndarray:
        S_hat_indices = np.argsort(self.p_t)
        S_hat = S_hat_indices[-self.n:]
        
        i_star = S_hat[np.argmin(self.p_t[S_hat])]
        S_hat_complement = S_hat_indices[:-self.n]
        j_star = S_hat_complement[np.argmax(self.p_t[S_hat_complement])]
        
        delta_t = np.zeros(self.K)
        delta_t[i_star] = 1.0 # Weights are not important, only the index matters.
        delta_t[j_star] = -1.0
        
        # [SOTA Rules] Apply the arm selection rule of CombGapE.
        # p_t = arg max_{s} (\pi_s^k - \pi_s^l)^2 / (T_s(t)(T_s(t)+1))
        # In our example, s is chosen only between i_star and j_star.
        
        # Add a small epsilon to the case where T_s = 0 to avoid division by zero.
        T_s_i = self.T_s[i_star]
        T_s_j = self.T_s[j_star]

        # CombGapE Gap Measurement
        gap_metric_i = (delta_t[i_star]**2) / ((T_s_i + 1e-9) * (T_s_i + 1 + 1e-9))
        gap_metric_j = (delta_t[j_star]**2) / ((T_s_j + 1e-9) * (T_s_j + 1 + 1e-9))
        
        if gap_metric_i > gap_metric_j:
            k_to_pull = i_star
        else:
            k_to_pull = j_star
            
        # Update the internal counter and return the action.
        self.T_s[k_to_pull] += 1
        
        C_t = np.zeros(self.K)
        C_t[k_to_pull] = 1.0
        return C_t

@register_algorithm("TTTS_Challenger")
class TTTS_Challenger(BaseAlgorithm):
    """
    hybrid baselines (action selection strategy):
    Using the "continuous action model" (QP solver) of ECC-AHT,
    but use the "randomized" brain of TTTS to select (i*, j*).
    """
    def __init__(self, K, n, Sigma, B, mu_signal=None, mu_0=None, delta_signal=None):
        super().__init__(K=K, n=n, Sigma=Sigma, B=B, mu_signal=mu_signal, mu_0=mu_0, delta_signal=delta_signal)

        self.C_var = cp.Variable(self.K)
        self.delta_param = cp.Parameter(self.K)
        self.prob = None

    def _build_problem(self):
        Sigma_reg = cp.psd_wrap(self.Sigma + 1e-6 * np.eye(self.Sigma.shape[0]))
        objective = cp.Minimize(cp.quad_form(self.C_var, Sigma_reg))
        constraints = [
            self.C_var @ self.delta_param == 1,
            cp.norm1(self.C_var) <= self.B
        ]
        self.prob = cp.Problem(objective, constraints)

    def select_action(self) -> np.ndarray:
        if self.prob is None:
            self._build_problem()
            
        # [SOTA Strategy] Thompson Sampling
        N_t = max(self.t, 2)
        a_k = self.p_t * N_t + 1
        b_k = (1 - self.p_t) * N_t + 1
        p_tilde = np.random.beta(a_k, b_k)
        
        S_hat_indices_tilde = np.argsort(p_tilde)
        S_hat_tilde = S_hat_indices_tilde[-self.n:]
        
        i_star_tilde = S_hat_tilde[np.argmin(p_tilde[S_hat_tilde])]
        S_hat_complement_tilde = S_hat_indices_tilde[:-self.n]
        j_star_tilde = S_hat_complement_tilde[np.argmax(p_tilde[S_hat_complement_tilde])]
        
        delta_t = np.zeros(self.K)
        delta_t[i_star_tilde] = self.delta_signal
        delta_t[j_star_tilde] = -self.delta_signal
        
        try:
            self.delta_param.value = delta_t
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            
            if self.C_var.value is None:
                return self.fallback_action(delta_t)
                
            C_t = self.C_var.value
            c_norm = np.linalg.norm(C_t, 1)
            if c_norm > 1e-9:
                return C_t / c_norm * self.B
            else:
                return self.fallback_action(delta_t)

        except cp.error.SolverError:
            return self.fallback_action(delta_t)

    def fallback_action(self, delta_t):
        C_t = delta_t / (np.linalg.norm(delta_t, 1) + 1e-9)
        return C_t * self.B

# --- MAB-based SOTA Baselines ---

@register_algorithm("CombGapE")
class CombGapE:
    """
    The true "model-agnostic" SOTA baseline (CombGapE).
    It does not inherit from BaseAlgorithm because it does not use a Gaussian model.
    It only cares about empirical means.
    """
    def __init__(self, K, n, Sigma=None, B=None, mu_signal=None, mu_0=None, delta_signal=None):
        self.K = K
        self.n = n
        # (Note: It completely ignores Sigma, B, mu_signal, mu_0, and delta_signal)
        
        # Beliefs: Model-independent empirical means and number of pulls.
        self.T_s = np.zeros(K) 
        self.emp_mean = np.zeros(K)
        
        self.t = 0 # Internal Time

    def is_terminated(self) -> bool:
        return False # Never terminate prematurely

    def get_S_hat(self) -> np.ndarray:
        """Return the n arms with the highest current average reward."""
        return np.argsort(self.emp_mean)[-self.n:]

    def select_action(self) -> np.ndarray:
        """
        Select a base arm (C_t = e_k) to pull.
        """
        self.t += 1
        
        # 1. Initialization Phase: Ensure each arm is pulled at least once.
        if self.t <= self.K:
            k_to_pull = self.t - 1
            C_t = np.zeros(self.K)
            C_t[k_to_pull] = 1.0
            return C_t

        #2. Determining the champion and challenger (based on average experience)
        S_hat_indices = np.argsort(self.emp_mean)
        S_hat = S_hat_indices[-self.n:]
        
        # Prevent errors during the initialization phase (when the mean is 0)
        if len(S_hat) == 0:
            k_to_pull = self.t % self.K
            C_t = np.zeros(self.K)
            C_t[k_to_pull] = 1.0
            return C_t
        
        i_star = S_hat[np.argmin(self.emp_mean[S_hat])]
        S_hat_complement = S_hat_indices[:-self.n]
        
        if len(S_hat_complement) == 0:
            k_to_pull = i_star # Only one option
        else:
            j_star = S_hat_complement[np.argmax(self.emp_mean[S_hat_complement])]
        
            # 3. Constructing difference vectors (only the indices matter)
            delta_t = np.zeros(self.K)
            delta_t[i_star] = 1.0 
            delta_t[j_star] = -1.0
            
            # 4. [SOTA Rules] Apply the CombGapE arm selection rule.
            T_s_i = self.T_s[i_star]
            T_s_j = self.T_s[j_star]

            gap_metric_i = (delta_t[i_star]**2) / ((T_s_i + 1e-9) * (T_s_i + 1 + 1e-9))
            gap_metric_j = (delta_t[j_star]**2) / ((T_s_j + 1e-9) * (T_s_j + 1 + 1e-9))
            
            if gap_metric_i > gap_metric_j:
                k_to_pull = i_star
            else:
                k_to_pull = j_star
            
        # 5. Return Action
        C_t = np.zeros(self.K)
        C_t[k_to_pull] = 1.0
        return C_t

    def update(self, C_t, y_t):
        """
        Model-independent update: Only the empirical mean is updated.
        """
        # 1. Locate the arm being pulled, k
        k_arr = np.where(C_t > 0)[0]
        if len(k_arr) == 0:
            return # Not a single arm pulling
            
        k = k_arr[0]
        
        # 2. Update the number of pulls
        self.T_s[k] += 1
        T_k = self.T_s[k]
        
        # 3. Updating the empirical mean (Welford's algorithm)
        self.emp_mean[k] = self.emp_mean[k] + (y_t - self.emp_mean[k]) / T_k

@register_algorithm("TTTS")
class TTTS:
    """
    A true implementation of Top-Two Thompson Sampling (TTTS).
    
    Features:
    1. Model-Free: Does not use Sigma, mu_0 or delta_signal.
    2. Data-Driven: Only based on observed historical data (counts, emp_mean).
    3. Assumption: Rewards follow a Gaussian distribution (optimized for sensor readings).
    """
    def __init__(self, K, n, Sigma=None, B=None, mu_signal=None, mu_0=None, delta_signal=None):
        self.K = K
        self.n = n
        
        # Status Tracking
        self.counts = np.zeros(K)    # The number of times each arm is pulled, T_k
        self.emp_mean = np.zeros(K)  # The empirical mean of each arm, mu_k_hat

        self.beta = 0.5 # TTTS's hyperparameter, typically fixed at 0.5 (determines pulling the champion or challenger)
        self.t = 0      # Time step

    def select_action(self) -> np.ndarray:
        self.t += 1
        
        # 1. Initialization Phase: Ensure each arm is pulled at least once.
        # (avoiding infinite variance in the posterior)
        if self.t <= self.K:
            k = self.t - 1
            return self._one_hot(k)

        # 2. Sampling step (Thompson Sampling)
        # We use a Gaussian posterior approximation.
        # Mean = empirical mean
        # Standard deviation = 1 / sqrt(counts)  (this is a heuristic for standard TS)
        # Add 1e-9 to prevent division by zero
        sigma_proxy = 1.0 / np.sqrt(self.counts + 1e-9)
        
        # --- First sampling: Searching for the "leader" (i*) ---
        # Sample theta from the posterior distribution of each arm.
        theta_1 = np.random.normal(self.emp_mean, sigma_proxy)
        i_star = np.argmax(theta_1)

        # --- Resampling loop: Finding the "Challenger" j* ---
        # Repeat the sampling process until a winner *different from* the current champion is found.
        # This winner is the most likely person to challenge the reigning champion.
        j_star = i_star
        max_resamples = 100 # Security limitations to prevent infinite loops
        
        for _ in range(max_resamples):
            theta_2 = np.random.normal(self.emp_mean, sigma_proxy)
            winner = np.argmax(theta_2)
            
            if winner != i_star:
                j_star = winner
                break
        
        # If after many sampling attempts no one can beat i_star (i_star has a significant advantage),
        # then we revert to selecting the second-place candidate based on the empirical mean as the challenger.
        if j_star == i_star:
            sorted_indices = np.argsort(self.emp_mean)
            # If the first place winner is i_star, take the second place winner; otherwise, take the first place winner.
            if sorted_indices[-1] == i_star:
                j_star = sorted_indices[-2]
            else:
                j_star = sorted_indices[-1]

        # 3. Decision-making steps
        # With a probability of beta (0.5), the current champion i* is selected; with a probability of 1-beta, the challenger j* is selected.
        # This aims to balance "validating the current best" and "exploring potential best".
        if np.random.rand() < self.beta:
            k_to_pull = i_star
        else:
            k_to_pull = j_star
            
        return self._one_hot(k_to_pull)

    def update(self, C_t, y_t):
        """
        Model-independent update: only update the statistics (Welford's algorithm)
        """
        k = np.argmax(C_t) # Find the index of the arm that was pulled.
        
        # Update counts
        self.counts[k] += 1
        n = self.counts[k]
        
        # Update the average experience (incremental update, resulting in more stable values)
        # new_mean = old_mean + (new_value - old_mean) / n
        self.emp_mean[k] += (y_t - self.emp_mean[k]) / n

    def get_S_hat(self) -> np.ndarray:
        """Return the n arms with the highest current average reward."""
        # `argsort` sorts in ascending order and then takes the last n elements.
        return np.argsort(self.emp_mean)[-self.n:]

    def is_terminated(self) -> bool:
        # Pure exploration algorithms typically do not stop on their own and are controlled by an external budget.
        return False

    def _one_hot(self, k) -> np.ndarray:
        """Helper function: Generate one-hot vectors"""
        v = np.zeros(self.K)
        v[k] = 1.0
        return v