# utils/analysis.py
import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# utils/analysis.py
class JacobianAnalyzer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.input_size = model.input_size

    def compute_state_jacobian(self, hidden_state):
        """
        Compute the state-to-state Jacobian matrix for a given hidden state.
        """
        hidden_state.requires_grad_(True)

        # Create a dummy input tensor
        dummy_input = torch.zeros(1, 1, self.input_size).to(self.device)  # batch_size=1, seq_len=1

        # Forward pass with the current hidden state
        with torch.enable_grad():
            # Create initial states
            c_state = torch.zeros_like(hidden_state)
            init_states = (hidden_state.unsqueeze(0), c_state.unsqueeze(0))

            # Run one step
            next_state, _ = self.model(dummy_input, init_states)
            next_hidden = next_state.squeeze(0).squeeze(0)  # Remove batch and seq dimensions

        # Compute Jacobian
        jacobian = torch.zeros(self.model.hidden_size, self.model.hidden_size).to(self.device)

        for i in range(self.model.hidden_size):
            if hidden_state.grad is not None:
                hidden_state.grad.zero_()

            # Compute gradient of i-th output with respect to input
            if next_hidden[i].requires_grad:
                next_hidden[i].backward(retain_graph=True)
                if hidden_state.grad is not None:
                    jacobian[i] = hidden_state.grad

        return jacobian

    def analyze_spectral_properties(self, num_samples=100):
        """
        Analyze spectral properties of the state-to-state Jacobian.
        """
        singular_values_squared = []

        for _ in range(num_samples):
            # Generate random hidden state
            hidden_state = torch.randn(self.model.hidden_size).to(self.device)

            # Compute Jacobian
            jacobian = self.compute_state_jacobian(hidden_state)

            # Compute singular values
            try:
                U, S, V = torch.svd(jacobian)
                singular_values_squared.extend((S ** 2).cpu().detach().numpy())
            except RuntimeError as e:
                print(f"SVD computation failed: {e}")
                continue

        if not singular_values_squared:
            raise ValueError("Failed to compute any singular values")

        singular_values_squared = np.array(singular_values_squared)

        # Compute mean and variance as per paper
        mJJT_1 = np.mean(singular_values_squared)
        sigma_JJT = np.var(singular_values_squared)

        return mJJT_1, sigma_JJT, singular_values_squared

    def plot_singular_value_distribution(self, singular_values_squared):
        """
        Plot histogram of squared singular values
        """
        plt.figure(figsize=(10, 6))
        plt.hist(singular_values_squared, bins=50, density=True, alpha=0.7)

        # Fit normal distribution
        try:
            mu, std = stats.norm.fit(singular_values_squared)
            x = np.linspace(min(singular_values_squared), max(singular_values_squared), 100)
            p = stats.norm.pdf(x, mu, std)
            plt.plot(x, p, 'r-', lw=2, label=f'Normal fit\nμ={mu:.2f}, σ={std:.2f}')
        except:
            print("Warning: Could not fit normal distribution")

        plt.axvline(x=1, color='k', linestyle='--', label='Ideal (x=1)')
        plt.title('Distribution of Squared Singular Values')
        plt.xlabel('Squared Singular Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return mu, std if 'mu' in locals() else (None, None)


def compute_convergence_timescale(model, sequence_length=1000):
    """
    Compute the convergence timescale ξ
    """
    device = next(model.parameters()).device
    correlations = []

    # Generate random input sequence
    x = torch.randn(1, sequence_length, model.input_size).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        hidden_seq, _ = model(x)
        hidden_states = hidden_seq.squeeze(0)  # Remove batch dimension

    # Compute correlations between consecutive states
    for t in range(1, len(hidden_states)):
        h1 = hidden_states[t - 1].view(-1)
        h2 = hidden_states[t].view(-1)
        correlation = torch.corrcoef(torch.stack([h1, h2]))[0, 1]
        correlations.append(correlation.item())

    # Compute χᶜ*ₛ as the limiting correlation
    chi_cs = np.mean(correlations[-100:])  # Use last 100 timesteps
    xi = -1 / np.log(abs(chi_cs))  # Use absolute value to handle negative correlations

    return xi, correlations